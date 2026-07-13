"""사용자 질문을 DB 조회, 웹 수집, 최종 답변으로 조율하는 서비스."""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent.prompts.commander import QA_COMMANDER_SYSTEM_PROMPT
from agent.application.run_context import (
    emit_run_event,
    invoke_with_metrics,
    measure_step,
    run_context,
)
from agent.application.run_contracts import RunEventSink, RunPhase, RunStatus
from agent.tools.recipe_learning import review_recipe_candidates
from agent.tools.realtime_scraping import realtime_scraping
from agent.tools.site_registry import get_collection_site_profile, list_collection_sites
from agent.tools.sqlite_query import sqlite_query
from agent.utils.logger import logger


def validate_citations(answer: str, valid_ids: list[int]) -> str:
    """답변의 job_id 인용이 실제 DB 조회 결과에 포함됐는지 검증한다."""

    valid = {str(job_id) for job_id in valid_ids}

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1) in valid else "[출처 확인 불가]"

    return re.sub(r"\[job_id:(\d+)\]", replace, answer)


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return content if isinstance(content, str) else str(content)


class ChatService:
    """DB 우선 조회와 필요 시 수집을 수행하는 로컬 Agent Orchestrator."""

    def __init__(self, llm_with_tools: Any = None, max_turns: int = 14):
        self._llm_with_tools = llm_with_tools
        self.max_turns = max_turns
        self._tools = {
            "sqlite_query": sqlite_query,
            "list_collection_sites": list_collection_sites,
            "get_collection_site_profile": get_collection_site_profile,
            "realtime_scraping": realtime_scraping,
            "review_recipe_candidates": review_recipe_candidates,
        }

    def _get_llm_with_tools(self):
        if self._llm_with_tools is None:
            from agent.application.model_clients import get_google_chat_model

            llm = get_google_chat_model("gemini-3.5-flash", temperature=0.0)
            self._llm_with_tools = llm.bind_tools(list(self._tools.values()))
        return self._llm_with_tools

    @staticmethod
    def _result(
        answer: str,
        *,
        context: Any,
        started: float,
        status: RunStatus = RunStatus.COMPLETED,
    ) -> dict[str, Any]:
        duration = max(0.0, time.perf_counter() - started)
        metrics = context.snapshot()
        metrics["duration_sec"] = round(duration, 6)
        return {
            "run_id": context.run_id,
            "run_status": status.value,
            "last_action_result": answer,
            "is_finished": status in {RunStatus.COMPLETED, RunStatus.FAILED},
            "duration_sec": duration,
            "step_durations": [{"node": "chat_orchestrator", "duration": duration}],
            "metrics": metrics,
            "llm_usage": metrics.get("llm", {}),
        }

    def run(
        self,
        query: str,
        *,
        run_id: str | None = None,
        event_sink: RunEventSink | None = None,
    ) -> dict[str, Any]:
        """질문을 처리하고 기존 호환 상태 형태로 결과를 반환한다."""

        query = str(query or "").strip()
        with run_context(
            run_id=run_id,
            query=query,
            event_sink=event_sink,
            prefix="chat",
        ) as (context, _created):
            started = time.perf_counter()
            if not query:
                emit_run_event(
                    "run_failed",
                    RunPhase.FAILED,
                    "질문이 비어 있습니다.",
                    status=RunStatus.FAILED,
                )
                return self._result(
                    "질문이 비어있습니다.",
                    context=context,
                    started=started,
                    status=RunStatus.FAILED,
                )

            logger.info("Executing ChatService orchestrator loop")
            emit_run_event("planning_started", RunPhase.PLANNING, "질문을 분석하고 있습니다.")
            messages = [
                SystemMessage(content=QA_COMMANDER_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
            valid_ids: list[int] = []

            try:
                for turn in range(self.max_turns):
                    logger.info("Chat orchestrator turn", turn=turn + 1)
                    with measure_step("chat_orchestrator_reasoning", turn=turn + 1):
                        response = invoke_with_metrics(
                            self._get_llm_with_tools(),
                            messages,
                            "chat_orchestrator",
                        )
                    messages.append(response)
                    if not response.tool_calls:
                        emit_run_event(
                            "answering_started",
                            RunPhase.ANSWERING,
                            "수집된 근거로 답변을 정리하고 있습니다.",
                        )
                        answer = validate_citations(
                            _message_text(response.content),
                            list(set(valid_ids)),
                        )
                        emit_run_event(
                            "run_completed",
                            RunPhase.COMPLETED,
                            "답변을 완료했습니다.",
                            status=RunStatus.COMPLETED,
                        )
                        return self._result(answer, context=context, started=started)

                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_id = tool_call["id"]
                        tool = self._tools.get(tool_name)
                        phase = {
                            "sqlite_query": RunPhase.DATABASE,
                            "realtime_scraping": RunPhase.COLLECTION,
                            "review_recipe_candidates": RunPhase.REVIEW,
                        }.get(tool_name, RunPhase.PLANNING)
                        emit_run_event(
                            "tool_started",
                            phase,
                            {
                                "sqlite_query": "저장된 공고를 조회하고 있습니다.",
                                "realtime_scraping": "웹에서 채용공고를 수집하고 있습니다.",
                                "review_recipe_candidates": "반복 경로 후보를 검토하고 있습니다.",
                            }.get(tool_name, "다음 작업을 준비하고 있습니다."),
                            data={"tool": tool_name},
                        )
                        if tool is None:
                            tool_result = f"알 수 없는 도구: {tool_name}"
                        else:
                            logger.info("Chat orchestrator tool call", tool=tool_name)
                            with measure_step(f"tool:{tool_name}"):
                                tool_result = tool.invoke(tool_args)
                            if tool_name == "sqlite_query":
                                for document_id in re.findall(
                                    r'<document id="(\d+)">',
                                    tool_result,
                                ):
                                    valid_ids.append(int(document_id))
                        messages.append(
                            ToolMessage(content=tool_result, tool_call_id=tool_id)
                        )
            except Exception as exc:
                emit_run_event(
                    "run_failed",
                    RunPhase.FAILED,
                    "질문 처리 중 오류가 발생했습니다.",
                    status=RunStatus.FAILED,
                    data={"error": str(exc)[:300]},
                )
                raise

            logger.error("Chat orchestrator exceeded max_turns", max_turns=self.max_turns)
            emit_run_event(
                "run_failed",
                RunPhase.FAILED,
                "최대 추론 횟수를 초과했습니다.",
                status=RunStatus.FAILED,
            )
            return self._result(
                "답변 생성 실패: 최대 추론 횟수를 초과하였습니다.",
                context=context,
                started=started,
                status=RunStatus.FAILED,
            )

    def answer(
        self,
        query: str,
        *,
        run_id: str | None = None,
        event_sink: RunEventSink | None = None,
    ) -> str:
        return str(
            self.run(query, run_id=run_id, event_sink=event_sink).get(
                "last_action_result"
            )
            or ""
        )


_chat_service: ChatService | None = None


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


__all__ = ["ChatService", "get_chat_service", "validate_citations"]

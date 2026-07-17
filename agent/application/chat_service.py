"""사용자 질문을 조사 LangGraph로 실행하는 애플리케이션 서비스."""

from __future__ import annotations

import re
import time
from typing import Any

from agent.application.run_context import (
    RunCancelled,
    emit_run_event,
    raise_if_cancelled,
    run_context,
)
from agent.application.run_contracts import RunEventSink, RunPhase, RunStatus
from agent.utils.logger import logger


def validate_citations(answer: str, valid_ids: list[int]) -> str:
    """답변의 job_id 인용이 실제 근거 문서에 포함됐는지 검증한다."""

    valid = {str(job_id) for job_id in valid_ids}

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1) in valid else "[출처 확인 불가]"

    return re.sub(r"\[job_id:(\d+)\]", replace, answer)


class ChatService:
    """웹과 CLI가 공유하는 유일한 사용자 요청 진입점."""

    def __init__(self, investigation_workflow: Any = None):
        self._investigation_workflow = investigation_workflow

    def _get_investigation_workflow(self):
        if self._investigation_workflow is None:
            import shared.config as config
            from agent.graph.investigation_workflow import InvestigationWorkflow

            self._investigation_workflow = InvestigationWorkflow(db_path=config.DB_PATH)
        return self._investigation_workflow

    @staticmethod
    def _result(
        answer: str,
        *,
        context: Any,
        started: float,
        status: RunStatus = RunStatus.COMPLETED,
        clarification: dict[str, Any] | None = None,
        investigation_id: str = "",
    ) -> dict[str, Any]:
        duration = max(0.0, time.perf_counter() - started)
        context.set_outcome(status)
        metrics = context.snapshot()
        metrics["duration_sec"] = round(duration, 6)
        result = {
            "run_id": context.run_id,
            "run_status": status.value,
            "last_action_result": answer,
            "is_finished": status in {RunStatus.COMPLETED, RunStatus.FAILED},
            "duration_sec": duration,
            "step_durations": [{"node": "investigation_workflow", "duration": duration}],
            "metrics": metrics,
            "llm_usage": metrics.get("llm", {}),
        }
        if clarification:
            result["clarification"] = clarification
        if investigation_id:
            result["investigation_id"] = investigation_id
        return result

    def run(
        self,
        query: str,
        *,
        run_id: str | None = None,
        event_sink: RunEventSink | None = None,
        conversation_id: str = "",
        investigation_id: str = "",
        clarification_answer: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """요청을 계획·확인·근거 검사·실행 순서로 처리한다."""

        query = str(query or "").strip()
        with run_context(
            run_id=run_id,
            query=query,
            event_sink=event_sink,
            prefix="chat",
            metadata={
                "conversation_id": conversation_id,
                "investigation_id": investigation_id,
                "resume_mode": bool(investigation_id and clarification_answer),
            },
            tags=["chat-request"],
        ) as (context, _created):
            started = time.perf_counter()
            if not query and not (investigation_id and clarification_answer):
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

            logger.info("Executing investigation workflow")
            emit_run_event("planning_started", RunPhase.PLANNING, "질문을 분석하고 있습니다.")
            try:
                raise_if_cancelled()
                workflow_result = self._get_investigation_workflow().run(
                    query,
                    conversation_id=conversation_id,
                    investigation_id=investigation_id,
                    clarification_answer=clarification_answer,
                )
                workflow_investigation = dict(workflow_result.get("investigation") or {})
                try:
                    workflow_status = RunStatus(
                        workflow_result.get("run_status") or RunStatus.COMPLETED.value
                    )
                except ValueError:
                    workflow_status = RunStatus.FAILED
                answer = validate_citations(
                    str(workflow_result.get("final_answer") or ""),
                    [int(item) for item in workflow_result.get("valid_ids", [])],
                )
                return self._result(
                    answer,
                    context=context,
                    started=started,
                    status=workflow_status,
                    clarification=workflow_result.get("clarification"),
                    investigation_id=str(
                        workflow_investigation.get("investigation_id") or investigation_id
                    ),
                )
            except RunCancelled:
                emit_run_event(
                    "run_cancelled",
                    RunPhase.CANCELLED,
                    "사용자 요청으로 실행을 중단했습니다.",
                    status=RunStatus.CANCELLED,
                )
                return self._result(
                    "실행을 취소했습니다.",
                    context=context,
                    started=started,
                    status=RunStatus.CANCELLED,
                    investigation_id=investigation_id,
                )
            except Exception as exc:
                emit_run_event(
                    "run_failed",
                    RunPhase.FAILED,
                    "질문 처리 중 오류가 발생했습니다.",
                    status=RunStatus.FAILED,
                    data={"error": str(exc)[:300]},
                )
                logger.exception("Investigation workflow failed", error=str(exc))
                raise

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

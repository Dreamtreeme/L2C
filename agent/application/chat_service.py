"""사용자 질문을 조사 LangGraph로 실행하는 애플리케이션 서비스."""

from __future__ import annotations

import time
from typing import Any, Protocol

from agent.observability.run_context import (
    ModelRequestTimeout,
    RunCancelled,
    RunDeadlineExceeded,
    emit_run_event,
    raise_if_cancelled,
    run_context,
)
from agent.observability.run_contracts import RunEventSink, RunPhase, RunStatus
from agent.utils.logger import logger


class InvestigationRunner(Protocol):
    """대화 서비스가 요구하는 조사 실행기의 최소 계약."""

    def run(
        self,
        query: str,
        *,
        conversation_id: str = "",
        resume_run_id: str = "",
        investigation_id: str = "",
        clarification_answer: Any = None,
    ) -> dict[str, Any]: ...


class ChatService:
    """로컬 API가 사용하는 사용자 요청 애플리케이션 서비스."""

    def __init__(self, investigation_workflow: "InvestigationRunner"):
        self._investigation_workflow = investigation_workflow

    @staticmethod
    def _result(
        answer: str,
        *,
        context: Any,
        started: float,
        status: RunStatus = RunStatus.COMPLETED,
        clarification: dict[str, Any] | None = None,
        investigation_id: str = "",
        resume_mode: str = "",
    ) -> dict[str, Any]:
        duration = max(0.0, time.perf_counter() - started)
        context.set_outcome(status)
        metrics = context.snapshot()
        metrics["duration_sec"] = round(duration, 6)
        result = {
            "run_id": context.run_id,
            "run_status": status.value,
            "last_action_result": answer,
            "is_finished": status
            in {
                RunStatus.COMPLETED,
                RunStatus.PARTIAL,
                RunStatus.FAILED,
            },
            "duration_sec": duration,
            "metrics": metrics,
            "llm_usage": metrics.get("llm", {}),
        }
        if clarification:
            result["clarification"] = clarification
        if investigation_id:
            result["investigation_id"] = investigation_id
        if resume_mode:
            result["resume_mode"] = resume_mode
        return result

    def run(
        self,
        query: str,
        *,
        run_id: str | None = None,
        event_sink: RunEventSink | None = None,
        conversation_id: str = "",
        resume_run_id: str = "",
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
                "resume_requested": bool(resume_run_id or investigation_id),
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
            emit_run_event(
                "planning_started", RunPhase.PLANNING, "질문을 분석하고 있습니다."
            )
            try:
                raise_if_cancelled()
                if self._investigation_workflow is None:
                    raise RuntimeError("이미 종료된 채팅 서비스입니다.")
                workflow_result = self._investigation_workflow.run(
                    query,
                    conversation_id=conversation_id,
                    resume_run_id=resume_run_id,
                    investigation_id=investigation_id,
                    clarification_answer=clarification_answer,
                )
                workflow_investigation = dict(
                    workflow_result.get("investigation") or {}
                )
                try:
                    workflow_status = RunStatus(
                        workflow_result.get("run_status") or RunStatus.COMPLETED.value
                    )
                except ValueError:
                    workflow_status = RunStatus.FAILED
                answer = str(workflow_result.get("final_answer") or "")
                return self._result(
                    answer,
                    context=context,
                    started=started,
                    status=workflow_status,
                    clarification=workflow_result.get("clarification"),
                    investigation_id=str(
                        workflow_investigation.get("investigation_id")
                        or investigation_id
                    ),
                    resume_mode=str(workflow_result.get("resume_mode") or ""),
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
            except (RunDeadlineExceeded, ModelRequestTimeout) as exc:
                emit_run_event(
                    "run_partial",
                    RunPhase.PARTIAL,
                    "실행 시간 경계에 도달해 확보한 결과까지만 보존했습니다.",
                    status=RunStatus.PARTIAL,
                    data={"failure_code": type(exc).__name__},
                )
                return self._result(
                    (
                        "실행 제한시간에 도달했습니다. "
                        "이미 저장된 자료는 유지되며 이번 요청은 부분 완료로 종료했습니다."
                    ),
                    context=context,
                    started=started,
                    status=RunStatus.PARTIAL,
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


__all__ = [
    "ChatService",
    "InvestigationRunner",
]

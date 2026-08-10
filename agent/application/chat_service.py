"""사용자 질문을 조사 LangGraph로 실행하는 애플리케이션 서비스."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from agent.observability.run_context import (
    ModelRequestTimeout,
    RunCancelled,
    RunDeadlineExceeded,
    emit_run_event,
    raise_if_cancelled,
    run_context,
)
from agent.observability.run_contracts import (
    ChatErrorPayload,
    ChatRequest,
    ChatResult,
    ChatStartedPayload,
    ChatStreamFrame,
    RunEvent,
    RunEventSink,
    RunPhase,
    new_run_id,
)
from agent.observability.run_registry import RunRegistry, get_run_registry
from agent.utils.logger import logger
from shared.schema.investigation_schema import ClarificationQuestion, GroundedAnswer
from shared.schema.run_schema import RunStatus


class ChatService:
    """채팅 실행과 진행 상태의 전체 수명주기를 관리한다."""

    def __init__(
        self,
        investigation_workflow: Any,
        *,
        run_registry: RunRegistry | None = None,
    ) -> None:
        self._investigation_workflow = investigation_workflow
        self._run_registry = run_registry or get_run_registry()

    @staticmethod
    def _result(
        answer: str,
        *,
        context: Any,
        started: float,
        status: RunStatus,
        request: ChatRequest,
        grounded_answer: GroundedAnswer | None = None,
        clarification: ClarificationQuestion | None = None,
        investigation_id: str = "",
        resume_mode: str = "",
    ) -> ChatResult:
        duration = max(0.0, time.perf_counter() - started)
        context.set_outcome(status)
        metrics = context.snapshot()
        metrics["duration_sec"] = round(duration, 6)
        return ChatResult(
            run_id=context.run_id,
            status=status,
            text=answer,
            grounded_answer=grounded_answer,
            clarification=clarification,
            investigation_id=investigation_id or request.investigation_id,
            resume_mode=resume_mode,
            conversation_id=request.conversation_id,
            metrics=metrics,
        )

    def execute(
        self,
        request: ChatRequest,
        *,
        run_id: str,
        event_sink: RunEventSink | None = None,
    ) -> ChatResult:
        """구조화된 사용자 요청으로 조사 그래프를 한 번 실행한다."""

        query = request.query.strip()
        with run_context(
            run_id=run_id,
            query=query,
            event_sink=event_sink,
            cancel_requested=self._run_registry.is_cancel_requested,
            prefix="chat",
            metadata={
                "conversation_id": request.conversation_id,
                "investigation_id": request.investigation_id,
                "resume_requested": bool(request.investigation_id),
            },
            tags=["chat-request"],
        ) as (context, _created):
            started = time.perf_counter()
            if not query and not (
                request.investigation_id and request.clarification_answer
            ):
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
                    request=request,
                )

            logger.info("Executing investigation workflow")
            try:
                raise_if_cancelled()
                workflow_result = self._investigation_workflow.run(
                    query,
                    conversation_id=request.conversation_id,
                    investigation_id=request.investigation_id,
                    clarification_answer=request.clarification_answer,
                )
                return self._result(
                    workflow_result.final_answer,
                    context=context,
                    started=started,
                    status=workflow_result.run_status,
                    request=request,
                    grounded_answer=workflow_result.grounded_answer,
                    clarification=workflow_result.clarification,
                    investigation_id=workflow_result.investigation.investigation_id,
                    resume_mode=workflow_result.resume_mode,
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
                    request=request,
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
                    request=request,
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

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamFrame]:
        """실행을 등록하고 진행 이벤트와 최종 결과를 순서대로 내보낸다."""

        query = request.query.strip()
        run_id = new_run_id("chat")
        self._run_registry.start(
            run_id,
            query,
            conversation_id=request.conversation_id,
        )
        event_queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def event_sink(event: RunEvent) -> None:
            self._run_registry.apply_event(event)
            loop.call_soon_threadsafe(event_queue.put_nowait, event)

        logger.info("Received query for commander", query=query, run_id=run_id)
        yield ChatStreamFrame("processing", ChatStartedPayload(run_id=run_id))

        task = asyncio.create_task(
            asyncio.to_thread(
                self.execute,
                request,
                run_id=run_id,
                event_sink=event_sink,
            )
        )
        try:
            while not task.done() or not event_queue.empty():
                try:
                    event = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=0.25,
                    )
                except asyncio.TimeoutError:
                    continue
                yield ChatStreamFrame("event", event)
            result = await task
        except asyncio.CancelledError:
            self._run_registry.request_cancel(run_id)
            raise
        except Exception as exc:
            self._run_registry.fail(run_id, str(exc))
            logger.exception(
                "Commander execution failed",
                error=str(exc),
                run_id=run_id,
            )
            yield ChatStreamFrame(
                "error",
                ChatErrorPayload(
                    run_id=run_id,
                    message=f"지휘자 에이전트 실행 실패: {exc}",
                ),
            )
            return

        self._run_registry.complete(run_id, result)
        yield ChatStreamFrame("final", result)
        yield ChatStreamFrame("done")

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._run_registry.get(run_id)

    def cancel_run(self, run_id: str) -> dict[str, Any] | None:
        return self._run_registry.request_cancel(run_id)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._run_registry.list_recent(limit=limit)


__all__ = ["ChatService"]

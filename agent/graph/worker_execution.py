"""검증된 행동 요청 하나를 실행하는 작업자 그래프 노드."""

from __future__ import annotations

import time
from typing import Any

from agent.application.run_context import raise_if_cancelled
from agent.graph.action_request import ActionRequest, ActionResult
from agent.graph.state import GraphState
from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph.worker_execution_handlers import execute_action_request
from agent.utils.logger import logger


def _validated_action_request(state: GraphState) -> ActionRequest | None:
    raw_request = state.get("pending_action")
    if raw_request is None:
        return None
    try:
        request = (
            raw_request
            if isinstance(raw_request, ActionRequest)
            else ActionRequest.model_validate(raw_request)
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Action request state is invalid", error=str(exc))
        return None

    metadata = dict(request.metadata or {})
    capture_id = str(state.get("current_capture_id") or "")
    if capture_id:
        metadata.setdefault("decision_capture_id", capture_id)
    return request.model_copy(update={"metadata": metadata})


def _missing_action_update(
    request: ActionRequest | None,
) -> dict[str, Any]:
    logger.warning("No validated action request is available.")
    result = ActionResult(
        source=request.source if request else "unknown",
        summary=request.summary if request else "",
        status="error",
        tool_results=[
            {
                "action": "none",
                "status": "error",
                "error": "No validated action request",
                "args": {},
            }
        ],
    )
    return {
        "pending_action": None,
        "last_action_result": result,
        "execution_records": [],
        "action_history": result.tool_results,
    }


def execution_node(state: GraphState) -> dict[str, Any]:
    """현재 화면에서 선택된 행동 실행 단위를 검증하고 실행한다."""

    raise_if_cancelled()
    started = time.perf_counter()
    request = _validated_action_request(state)
    if request is None or not request.tool_calls:
        return _missing_action_update(request)

    logger.info(
        "Action request received",
        source=request.source,
        summary=request.summary,
    )
    context = WorkerExecutionContext.from_state(state, request)
    execute_action_request(context)
    logger.info(
        "Execution node completed",
        duration_sec=round(time.perf_counter() - started, 6),
    )
    return context.build_state_update()


__all__ = ["execution_node"]

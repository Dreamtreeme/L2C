"""검증된 행동 요청 하나를 실행하는 작업자 그래프 노드."""

from __future__ import annotations

import time
from typing import Any

from langgraph.runtime import Runtime

from agent.observability.run_context import raise_if_cancelled
from agent.runtime.worker_contracts import (
    ActionRequest,
    WorkerCompletionReason,
    WorkerState,
    WorkerStateUpdate,
    build_action_event,
)
from shared.schema.feedback_schema import ExecutionEvent
from agent.graph.worker_action_effects import (
    activate_clicked_job_card,
    execute_state_action,
    execute_ui_action,
    raise_for_action_failure,
)
from agent.graph.worker_action_guard import (
    guard_action_request,
    guard_ui_action,
)
from agent.graph.worker_action_recording import record_action_result
from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph.worker_execution_policy import compact_action_args
from agent.graph.worker_transition_recording import set_transition_request
from agent.runtime.worker_actions import (
    STATE_UPDATE_ACTIONS,
    TERMINAL_ACTIONS,
    UI_ACTIONS,
)
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.utils.logger import logger


def _record_failed_call(
    context: WorkerExecutionContext,
    *,
    action_name: str,
    args: dict[str, Any],
    error: Exception,
    action_sequence: int,
    tool_call_id: str,
    call_metadata: dict[str, Any],
) -> None:
    before_snapshot = context.before_snapshot()
    record_action_result(
        context,
        action_name=action_name,
        args=args,
        result={
            "action": action_name,
            "status": "error",
            "error": str(error),
        },
        before_snapshot=before_snapshot,
        action_sequence=action_sequence,
        tool_call_id=tool_call_id,
        tool_call_metadata=call_metadata,
    )
    transition = context.state["transition"]
    transition["error_count"] = int(transition.get("error_count", 0) or 0) + 1


def _execute_tool_call(
    context: WorkerExecutionContext,
    *,
    action_name: str,
    args: dict[str, Any],
    call_metadata: dict[str, Any],
    tool_call_id: str,
    action_sequence: int,
    before_snapshot: dict[str, Any],
    step_started: float,
) -> tuple[dict[str, Any], bool] | None:
    """가드가 허용한 도구 하나를 유형에 맞는 실행기로 전달한다."""

    state = context.state
    if action_name in UI_ACTIONS:
        if guard_ui_action(
            context,
            action_name,
            args,
            before_snapshot,
            step_started,
        ):
            return None
        result, screen_changed = execute_ui_action(
            context,
            action_name,
            args,
            call_metadata,
            tool_call_id,
            action_sequence,
        )
        activate_clicked_job_card(
            context,
            result,
            action_name,
            {**args, **call_metadata},
        )
        return result, screen_changed
    if action_name in STATE_UPDATE_ACTIONS:
        result = execute_state_action(
            context,
            action_name,
            args,
        )
        return result, False
    if action_name in TERMINAL_ACTIONS:
        result = context.worker_runtime.get_action_tools().finish_task(args["result"])
        raise_for_action_failure(result)
        completion_reason: WorkerCompletionReason = "agent_finished"
        if context.action_request.source == "reasoning_policy":
            completion_reason = "reasoning_limit"
        elif context.action_request.source == "screen_policy":
            completion_reason = "screen_unavailable"
        state["lifecycle"]["is_finished"] = True
        state["lifecycle"]["completion_reason"] = completion_reason
        state["progress"]["stage"] = "finished"
        return result, False
    raise ValueError(f"Unknown tool: {action_name}")


def _execute_action_request(context: WorkerExecutionContext) -> None:
    """요청에 포함된 검증된 도구를 선언된 순서대로 실행한다."""

    state = context.state
    request = context.action_request
    if guard_action_request(context):
        return
    for tool_call in request.tool_calls:
        action_name = tool_call.name
        args = dict(tool_call.args)
        call_metadata = dict(tool_call.metadata)
        if action_name == "review_job_detail":
            args.setdefault("page_role", "job_detail")

        logger.info(
            "Executing requested tool",
            source=request.source,
            action=action_name,
            args=compact_action_args(action_name, args),
        )
        step_started = time.perf_counter()
        before_snapshot = context.before_snapshot()
        action_sequence = context.next_action_sequence()
        try:
            outcome = _execute_tool_call(
                context,
                action_name=action_name,
                args=args,
                call_metadata=call_metadata,
                tool_call_id=tool_call.id,
                action_sequence=action_sequence,
                before_snapshot=before_snapshot,
                step_started=step_started,
            )
            if outcome is None:
                break
            result, screen_changed = outcome

            record_action_result(
                context,
                action_name=action_name,
                args=args,
                result=result,
                before_snapshot=before_snapshot,
                action_sequence=action_sequence,
                screen_changed=screen_changed,
                record_ui=action_name in UI_ACTIONS,
                tool_call_id=tool_call.id,
                tool_call_metadata=call_metadata,
            )
            state["transition"]["error_count"] = 0
            logger.info(
                "Action execution completed",
                action=action_name,
                status=result.get("status", ""),
                reason=result.get("reason", ""),
                duration_sec=round(time.perf_counter() - step_started, 6),
            )
            if state["lifecycle"].get("is_finished", False):
                break
        except Exception as exc:
            logger.error(
                "Failed to execute action",
                action=action_name,
                error=str(exc),
            )
            _record_failed_call(
                context,
                action_name=action_name,
                args=args,
                error=exc,
                action_sequence=action_sequence,
                tool_call_id=tool_call.id,
                call_metadata=call_metadata,
            )
            if request.source == "reflex":
                set_transition_request(
                    context,
                    action_sequence,
                    action_name,
                    args,
                    "reflex",
                )
                transition_request = state["transition"].get("transition_request")
                if transition_request is not None:
                    transition_request["execution_failed"] = True
            break


def _action_request_for_execution(state: WorkerState) -> ActionRequest | None:
    request = state["decision"].get("pending_action")
    if request is None:
        return None

    observation_id = str(state["observation"].get("observation_id") or "")
    return request.model_copy(
        update={"observation_id": request.observation_id or observation_id}
    )


def _missing_action_update(
    state: WorkerState,
    request: ActionRequest | None,
) -> WorkerStateUpdate:
    logger.warning("No validated action request is available.")
    result: dict[str, Any] = {
        "action": "none",
        "status": "error",
        "error": "No validated action request",
        "args": {},
        "action_source": request.source if request else "unknown",
    }
    prior_events = [
        ExecutionEvent.model_validate(event)
        for event in (state["transition"].get("action_events", []) or [])
    ]
    return {
        "decision": {"pending_action": None},
        "transition": {
            "action_events": [
                *prior_events,
                build_action_event(
                    len(prior_events),
                    result,
                    observation_id=str(
                        state["observation"].get("observation_id") or ""
                    ),
                ),
            ],
        },
    }


def execution_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> WorkerState | WorkerStateUpdate:
    """현재 화면에서 선택된 행동 실행 단위를 검증하고 실행한다."""

    raise_if_cancelled()
    started = time.perf_counter()
    request = _action_request_for_execution(state)
    if request is None or not request.tool_calls:
        return _missing_action_update(state, request)

    logger.info(
        "Action request received",
        source=request.source,
        summary=request.summary,
    )
    context = WorkerExecutionContext.from_state(
        state,
        request,
        runtime.context.vision,
        runtime.context.data,
    )
    _execute_action_request(context)
    logger.info(
        "Execution node completed",
        duration_sec=round(time.perf_counter() - started, 6),
    )
    return context.finish_state()


__all__ = ["execution_node"]

"""검증된 행동 요청 하나를 실행하는 작업자 그래프 노드."""

from __future__ import annotations

import time
from typing import Any

from langgraph.runtime import Runtime

from agent.observability.run_context import raise_if_cancelled
from agent.runtime.worker_contracts import (
    ActionRequest,
    WorkerState,
    build_action_event,
)
from agent.graph.worker_action_effects import (
    activate_clicked_job_card,
    execute_state_action,
    execute_ui_action,
    raise_for_action_failure,
)
from agent.graph.worker_action_guard import (
    guard_return_to_results,
    guard_ui_action,
)
from agent.graph.worker_action_recording import record_action_result
from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph.worker_execution_policy import compact_action_args
from agent.runtime.detail_runtime import is_job_detail_context
from agent.runtime.job_field_contract import merge_job_detail_coverage
from agent.runtime.worker_actions import (
    STATE_UPDATE_ACTIONS,
    TERMINAL_ACTIONS,
    UI_ACTIONS,
)
from agent.runtime.worker_state import job_detail_key_from_state
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.utils.logger import logger


def _record_successful_call(
    context: WorkerExecutionContext,
    *,
    action_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    before_snapshot: dict[str, Any],
    action_sequence: int,
    screen_changed: bool,
    tool_call_id: str,
    call_metadata: dict[str, Any],
) -> None:
    record_action_result(
        context,
        action_name=action_name,
        args=args,
        result=result,
        before_snapshot=before_snapshot,
        action_sequence=action_sequence,
        screen_changed=screen_changed,
        record_ui=action_name in UI_ACTIONS,
        tool_call_id=tool_call_id,
        tool_call_metadata=call_metadata,
    )


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
    transition = context.result.state["transition"]
    transition["error_count"] = int(
        transition.get("error_count", 0) or 0
    ) + 1


def _observe_job_detail_fields(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
) -> None:
    """기존 추론 호출이 읽은 상세 필드 근거를 현재 공고에 누적한다."""

    if action_name not in {
        "click_marker",
        "scroll",
        "finish_detail_reading",
    }:
        return
    state = context.result.state
    observation = state["observation"]
    current_url = str(observation.get("current_url") or "")
    page_role = str(
        args.get("page_role")
        or observation.get("current_page_role")
        or ""
    )


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
) -> tuple[dict[str, Any], bool, ActionRequest | None] | None:
    """가드가 허용한 도구 하나를 유형에 맞는 실행기로 전달한다."""

    state = context.result.state
    if guard_return_to_results(
        context,
        action_name,
        args,
        before_snapshot,
        step_started,
    ):
        return None
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
        return result, screen_changed, None
    if action_name in STATE_UPDATE_ACTIONS:
        result, follow_up = execute_state_action(
            context,
            action_name,
            args,
            action_sequence,
        )
        return result, False, follow_up
    if action_name in TERMINAL_ACTIONS:
        result = context.input.worker_runtime.get_action_tools().finish_task(
            args["result"]
        )
        raise_for_action_failure(result)
        state["lifecycle"]["is_finished"] = True
        return result, False, None
    raise ValueError(f"Unknown tool: {action_name}")
    if not is_job_detail_context(current_url, page_role=page_role):
        return
    collection = state["collection"]
    collection["job_detail_coverage"] = merge_job_detail_coverage(
        dict(collection.get("job_detail_coverage", {}) or {}),
        args,
        state=state,
        current_url=current_url,
        detail_key=job_detail_key_from_state(state),
    )


def _execute_action_request(context: WorkerExecutionContext) -> None:
    """요청에 포함된 검증된 도구를 선언된 순서대로 실행한다."""

    state = context.result.state
    request = context.input.action_request
    for tool_call in request.tool_calls:
        action_name = tool_call.name
        args = dict(tool_call.args)
        call_metadata = dict(tool_call.metadata)
        if action_name == "finish_detail_reading":
            args.setdefault("page_role", "job_detail")
        _observe_job_detail_fields(context, action_name, args)

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
            result, screen_changed, follow_up = outcome

            _record_successful_call(
                context,
                action_name=action_name,
                args=args,
                result=result,
                before_snapshot=before_snapshot,
                action_sequence=action_sequence,
                screen_changed=screen_changed,
                tool_call_id=tool_call.id,
                call_metadata=call_metadata,
            )
            logger.info(
                "Action execution completed",
                action=action_name,
                status=result.get("status", ""),
                reason=result.get("reason", ""),
                duration_sec=round(time.perf_counter() - step_started, 6),
            )
            is_finished = bool(
                state["lifecycle"].get("is_finished", False)
            )
            if follow_up is not None and not is_finished:
                context.result.next_pending_action = follow_up
                break
            if is_finished:
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
                transition = state["transition"]
                transition["transition_request"] = {
                    **dict(transition.get("transition_request", {}) or {}),
                    "source": "reflex",
                    "execution_failed": True,
                    "failed_action": action_name,
                }
            break


def _validated_action_request(state: WorkerState) -> ActionRequest | None:
    raw_request = state["decision"].get("pending_action")
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
    capture_id = str(
        state["observation"].get("current_capture_id") or ""
    )
    if capture_id:
        metadata.setdefault("decision_capture_id", capture_id)
    return request.model_copy(update={"metadata": metadata})


def _missing_action_update(
    state: WorkerState,
    request: ActionRequest | None,
) -> dict[str, Any]:
    logger.warning("No validated action request is available.")
    result = {
        "action": "none",
        "status": "error",
        "error": "No validated action request",
        "args": {},
        "action_source": request.source if request else "unknown",
    }
    prior_events = [
        dict(event)
        for event in (state["transition"].get("action_events", []) or [])
        if isinstance(event, dict)
    ]
    return {
        "decision": {"pending_action": None},
        "transition": {
            "action_events": [
                *prior_events,
                build_action_event(len(prior_events), result),
            ],
        },
    }


def execution_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> dict[str, Any]:
    """현재 화면에서 선택된 행동 실행 단위를 검증하고 실행한다."""

    raise_if_cancelled()
    started = time.perf_counter()
    request = _validated_action_request(state)
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
    return context.build_state_update()


__all__ = ["execution_node"]

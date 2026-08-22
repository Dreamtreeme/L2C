"""작업자 행동을 물리 실행하기 전에 적용하는 안전 검증."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.worker_action_recording import record_action_result
from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph.worker_execution_policy import (
    blocked_action_reason,
    repeats_no_effect_request,
    repeats_submitted_input_request,
)
from agent.runtime.transition_runtime import latest_no_effect_transition
from agent.utils.logger import logger


def _record_guard_result(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    before_snapshot: dict[str, Any],
    *,
    status: str,
    reason: str,
    message: str,
    step_started: float,
    increments_error: bool = False,
) -> None:
    state = context.state
    result: dict[str, Any] = {
        "status": status,
        "action": action_name,
        "result": message if status != "error" else None,
        "error": message if status == "error" else None,
        "reason": reason,
    }

    action_sequence = context.next_action_sequence()
    record_action_result(
        context,
        action_name=action_name,
        args=args,
        result=result,
        before_snapshot=before_snapshot,
        action_sequence=action_sequence,
    )
    if increments_error:
        transition = state["transition"]
        transition["error_count"] = int(transition.get("error_count", 0) or 0) + 1
    logger.warning(message, action=action_name, reason=reason)
    logger.debug(
        "Action guard completed",
        duration_sec=round(time.perf_counter() - step_started, 6),
    )


def _record_policy_block(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    reason: str,
    before_snapshot: dict[str, Any],
    step_started: float,
) -> None:
    _record_guard_result(
        context,
        action_name,
        args,
        before_snapshot,
        status="error",
        reason=reason,
        message="Blocked action outside the public job collection policy.",
        step_started=step_started,
        increments_error=True,
    )


def guard_action_request(context: WorkerExecutionContext) -> bool:
    """이미 확인된 동일 요청을 도구 묶음 실행 전에 막는다."""

    if repeats_submitted_input_request(context.state, context.action_request):
        first_call = context.action_request.tool_calls[0]
        _record_guard_result(
            context,
            first_call.name,
            dict(first_call.args),
            context.before_snapshot(),
            status="skipped",
            reason="search_query_already_submitted",
            message=(
                "Skipped the search query because the same value was already "
                "entered and submitted successfully."
            ),
            step_started=time.perf_counter(),
        )
        return True

    no_effect_transition = latest_no_effect_transition(context.state)
    if not repeats_no_effect_request(no_effect_transition, context.action_request):
        return False

    first_call = context.action_request.tool_calls[0]
    _record_guard_result(
        context,
        first_call.name,
        dict(first_call.args),
        context.before_snapshot(),
        status="skipped",
        reason="same_screen_no_effect_action_blocked",
        message=(
            "Blocked an action request that already had no effect on this screen. "
            "Choose another navigation method."
        ),
        step_started=time.perf_counter(),
        increments_error=True,
    )
    return True


def guard_ui_action(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    before_snapshot: dict[str, Any],
    step_started: float,
) -> bool:
    """작업 권한과 탭 닫기 선행조건을 검사한다."""

    state = context.state
    policy_reason = blocked_action_reason(
        state,
        action_name,
        args,
        source=context.action_request.source,
    )
    if policy_reason:
        _record_policy_block(
            context,
            action_name,
            args,
            policy_reason,
            before_snapshot,
            step_started,
        )
        return True

    no_effect_transition = latest_no_effect_transition(state)
    if (
        action_name == "close_current_tab"
        and no_effect_transition.get("action") != "go_back"
    ):
        _record_guard_result(
            context,
            action_name,
            args,
            before_snapshot,
            status="skipped",
            reason="close_tab_requires_failed_go_back",
            message=(
                "Blocked close_current_tab before a go_back attempt was confirmed "
                "to have no effect."
            ),
            step_started=step_started,
            increments_error=True,
        )
        return True

    return False


__all__ = ["guard_action_request", "guard_ui_action"]

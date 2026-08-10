"""작업자 행동을 물리 실행하기 전에 적용하는 안전 검증."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.worker_action_recording import record_action_result
from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph.worker_execution_policy import (
    blocked_action_reason,
    repeats_no_effect_target,
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


def guard_ui_action(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    before_snapshot: dict[str, Any],
    step_started: float,
) -> bool:
    """작업 권한을 벗어나거나 효과 없는 행동의 반복이면 실행을 막는다."""

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
    if repeats_no_effect_target(
        no_effect_transition,
        action_name,
        args,
    ):
        _record_guard_result(
            context,
            action_name,
            args,
            before_snapshot,
            status="skipped",
            reason="same_screen_no_effect_action_blocked",
            message=(
                "Blocked an atomic UI action that already had no effect on this "
                "screen. Choose another navigation method."
            ),
            step_started=step_started,
            increments_error=True,
        )
        return True

    return False


__all__ = ["guard_ui_action"]

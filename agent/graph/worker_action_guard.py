"""작업자 행동을 물리 실행하기 전에 적용하는 안전 검증."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.worker_action_recording import record_action_result
from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph.worker_execution_policy import (
    compact_action_args,
    repeats_no_effect_target,
    sensitive_action_reason,
)
from agent.graph.worker_transition_recording import set_transition_request
from agent.runtime.worker_state import return_to_job_results_for_url
from agent.runtime.action_validation import text_input_target_rejection
from agent.runtime.transition_runtime import latest_no_effect_transition
from agent.runtime.worker_actions import (
    DIRECT_SCREEN_ACTION_SOURCES,
    RETURN_ACTIONS,
    STATE_UPDATE_ACTIONS,
    UI_ACTIONS,
)
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
    observation_required: bool = False,
    details: dict[str, Any] | None = None,
) -> None:
    state = context.result.state
    if observation_required:
        state["observation"]["current_url_stale"] = True
        context.result.screen_changed = True
    result: dict[str, Any] = {
        "status": status,
        "action": action_name,
        "result": message if status != "error" else None,
        "error": message if status == "error" else None,
        "reason": reason,
    }
    if observation_required:
        result["observation_required"] = True
    if details:
        result["guard"] = dict(details)

    action_sequence = context.next_action_sequence()
    if observation_required:
        set_transition_request(
            context,
            action_sequence,
            action_name,
            args,
            "guard",
        )
    record_action_result(
        context,
        action_name=action_name,
        args=args,
        result=result,
        before_snapshot=before_snapshot,
        action_sequence=action_sequence,
        screen_changed=observation_required,
    )
    if increments_error:
        transition = state["transition"]
        transition["error_count"] = int(
            transition.get("error_count", 0) or 0
        ) + 1
    logger.warning(message, action=action_name, reason=reason)
    logger.debug(
        "Action guard completed",
        duration_sec=round(time.perf_counter() - step_started, 6),
    )


def _require_human_approval(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    reason: str,
    before_snapshot: dict[str, Any],
    step_started: float,
) -> None:
    state = context.result.state
    observation = state["observation"]
    state["safety"]["pending_human_approval"] = True
    state["safety"]["human_approval_request"] = {
        "status": "needs_human_approval",
        "reason": reason,
        "action": action_name,
        "args": compact_action_args(action_name, args),
        "current_url": str(observation.get("current_url") or ""),
        "message": (
            "Autonomous execution stopped before a sensitive or "
            "irreversible step."
        ),
    }
    _record_guard_result(
        context,
        action_name,
        args,
        before_snapshot,
        status="skipped",
        reason=reason,
        message="Skipped sensitive action; human confirmation is required.",
        step_started=step_started,
    )


def guard_return_to_results(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    before_snapshot: dict[str, Any],
    step_started: float,
) -> bool:
    """상세 수집이 끝난 뒤 목록 복귀 외의 추가 탐색을 차단한다."""

    state = context.result.state
    current_url = str(state["observation"].get("current_url") or "")
    return_pending = return_to_job_results_for_url(
        state,
        current_url,
    )
    if not return_pending:
        return False
    if action_name not in STATE_UPDATE_ACTIONS and (
        action_name not in UI_ACTIONS or action_name in RETURN_ACTIONS
    ):
        return False

    _record_guard_result(
        context,
        action_name,
        args,
        before_snapshot,
        status="skipped",
        reason="return_to_job_results",
        message=(
            "상세 수집이 이미 완료되었습니다. 같은 공고를 더 읽지 말고 "
            "검색 결과 화면으로 복귀해야 합니다."
        ),
        step_started=step_started,
    )
    return True


def guard_ui_action(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    before_snapshot: dict[str, Any],
    step_started: float,
) -> bool:
    """현재 캡처와 목표가 유효하며 안전할 때만 UI 행동을 허용한다."""

    state = context.result.state
    request = context.input.action_request
    sensitive_reason = sensitive_action_reason(
        state,
        action_name,
        args,
        source=request.source,
    )
    if sensitive_reason:
        _require_human_approval(
            context,
            action_name,
            args,
            sensitive_reason,
            before_snapshot,
            step_started,
        )
        return True

    if (
        action_name in {"click_marker", "type_in_marker"}
        and request.source not in DIRECT_SCREEN_ACTION_SOURCES
    ):
        guard_result = context.input.worker_runtime.check_reasoning_screen(
            state,
            marker_id=args.get("marker_id"),
        )
        if guard_result.get("must_refresh"):
            screen_changed = bool(guard_result.get("stale"))
            _record_guard_result(
                context,
                action_name,
                args,
                before_snapshot,
                status="skipped",
                reason=(
                    "screen_changed_during_reasoning"
                    if screen_changed
                    else "screen_validation_unavailable"
                ),
                message=(
                    "Skipped UI action because the screen changed while "
                    "reasoning; a fresh perception is required."
                    if screen_changed
                    else "Skipped UI action because its source screen could "
                    "not be validated; a fresh perception is required."
                ),
                step_started=step_started,
                observation_required=True,
                details=guard_result,
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
        )
        return True

    if action_name == "type_in_marker":
        target_rejection = text_input_target_rejection(
            list(state["observation"].get("current_markers", []) or []),
            args.get("marker_id"),
        )
        if target_rejection:
            _record_guard_result(
                context,
                action_name,
                args,
                before_snapshot,
                status="error",
                reason=target_rejection["reason"],
                message=(
                    "Blocked type_in_marker because the selected marker does not "
                    "look like a text input target. Choose the visible input "
                    "container or placeholder marker."
                ),
                step_started=step_started,
                increments_error=True,
            )
            return True

    return False


__all__ = ["guard_return_to_results", "guard_ui_action"]

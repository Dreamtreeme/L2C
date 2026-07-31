"""작업자 행동을 물리 실행하기 전에 적용하는 안전 검증."""

from __future__ import annotations

from typing import Any

from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph.worker_execution_policy import (
    repeats_no_effect_target,
    sensitive_action_reason,
)
from agent.graph.worker_resources import check_current_reasoning_screen
from agent.graph.worker_state import return_to_job_results_for_url
from agent.runtime.action_validation import text_input_target_rejection
from agent.runtime.transition_runtime import latest_no_effect_transition
from agent.runtime.worker_actions import (
    DIRECT_SCREEN_ACTION_SOURCES,
    RETURN_ACTIONS,
    STATE_UPDATE_ACTIONS,
    UI_ACTIONS,
)


def guard_return_to_results(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    before_snapshot: dict[str, Any],
    step_started: float,
) -> bool:
    """상세 수집이 끝난 뒤 목록 복귀 외의 추가 탐색을 차단한다."""

    return_pending = return_to_job_results_for_url(
        {
            **context.state,
            "return_to_job_results": context.return_to_job_results,
            "current_url": context.current_url,
        },
        context.current_url,
    )
    if not return_pending:
        return False
    if action_name not in STATE_UPDATE_ACTIONS and (
        action_name not in UI_ACTIONS or action_name in RETURN_ACTIONS
    ):
        return False

    context.append_guard_result(
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

    if (
        action_name in {"click_marker", "type_in_marker"}
        and context.action_request.source not in DIRECT_SCREEN_ACTION_SOURCES
    ):
        guard_result = check_current_reasoning_screen(
            context.state,
            marker_id=args.get("marker_id"),
        )
        if guard_result.get("stale"):
            context.append_guard_result(
                action_name,
                args,
                before_snapshot,
                status="skipped",
                reason="screen_changed_during_reasoning",
                message=(
                    "Skipped UI action because the screen changed while reasoning; "
                    "a fresh perception is required."
                ),
                step_started=step_started,
                observation_required=True,
                details=guard_result,
            )
            return True

    no_effect_transition = latest_no_effect_transition(context.state)
    if repeats_no_effect_target(
        no_effect_transition,
        action_name,
        args,
    ):
        context.append_guard_result(
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

    sensitive_reason = sensitive_action_reason(
        {
            **context.state,
            "current_markers": context.current_markers,
        },
        action_name,
        args,
        source=context.action_request.source,
    )
    if sensitive_reason:
        context.require_human_approval(
            action_name,
            args,
            sensitive_reason,
            before_snapshot,
            step_started,
        )
        return True

    if action_name == "type_in_marker":
        target_rejection = text_input_target_rejection(
            context.current_markers,
            args.get("marker_id"),
        )
        if target_rejection:
            context.append_guard_result(
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

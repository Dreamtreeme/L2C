"""작업자 그래프의 화면 전환 판정 노드."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langgraph.runtime import Runtime

from agent.recipe.replay_runtime import (
    blocked_recipe_keys_after,
    record_replay_outcome,
    replay_session_after_transition,
    verify_replay_after_state,
)
from agent.runtime.site_context import is_job_detail_context
from agent.runtime.worker_contracts import (
    ObservationPatch,
    RecipeReplayPatch,
    TransitionPatch,
    TransitionRequest,
    WorkerState,
    WorkerStateUpdate,
    apply_worker_state_update,
    attach_action_transition,
)
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.runtime.transition_runtime import (
    input_text_confirmed_by_ocr,
    reused_ocr_observation,
    transition_has_visual_change,
    transition_record,
    transition_result,
    transition_result_without_request,
)
from agent.utils.logger import logger
from shared.schema.execution_record_schema import TransitionStatus


@dataclass(frozen=True, slots=True)
class TransitionDecision:
    """현재 캡처에 대한 전이 판정과 필요한 상태 효과."""

    request: TransitionRequest
    status: TransitionStatus
    reason: str
    needs_ocr: bool = False
    records_outcome: bool = True
    block_recipe: bool = False
    reuse_previous_ocr: bool = False


def _decide_before_ocr(
    state: WorkerState,
    request: TransitionRequest,
    *,
    visual_changed: bool,
) -> TransitionDecision:
    source = str(request.get("source") or "")
    action = str(request.get("action") or "")
    transition_actions = list(request.get("transition_actions") or [])
    if source == "reflex":
        matched, _reason, _trace = verify_replay_after_state(request, state)
        if matched:
            return TransitionDecision(
                request=request,
                status="needs_ocr",
                reason="rule_effect_ocr_required",
                needs_ocr=True,
                records_outcome=False,
            )
    input_occurred = (
        action == "type_in_marker" or "type_in_marker" in transition_actions
    )
    if visual_changed or input_occurred:
        return TransitionDecision(
            request=request,
            status="needs_ocr",
            reason="ocr_required" if visual_changed else "input_ocr_required",
            needs_ocr=True,
            records_outcome=False,
        )
    return TransitionDecision(
        request=request,
        status="unknown",
        reason=(
            "reflex_no_screen_change" if source == "reflex" else "no_screen_change"
        ),
        block_recipe=source == "reflex",
        reuse_previous_ocr=True,
    )


def _decide_after_ocr(
    state: WorkerState,
    request: TransitionRequest,
    *,
    visual_changed: bool,
) -> TransitionDecision:
    source = str(request.get("source") or "")
    current_url = str(state["observation"].get("current_url") or "")
    before_url = str(request.get("before_url") or "")
    url_changed = bool(before_url and current_url and before_url != current_url)
    markers = list(state["observation"].get("current_markers") or [])
    input_confirmed = input_text_confirmed_by_ocr(state, request)
    queue_target_reached = is_job_detail_context(
        current_url,
        page_role=str(state["observation"].get("current_page_role") or ""),
        marker_texts=[
            marker.get("text") for marker in markers if isinstance(marker, dict)
        ],
    )

    evaluated_request = request
    block_recipe = False
    if source == "reflex":
        matched, reason, after_state_match = verify_replay_after_state(
            request,
            state,
        )
        evaluated_request = request.copy()
        evaluated_request["after_state_match"] = after_state_match
        status = "ready" if matched else "unknown"
        block_recipe = not matched
    elif source == "job_card_queue" and not queue_target_reached:
        status = "unknown"
        reason = "job_card_detail_not_reached"
    elif markers and (url_changed or visual_changed or input_confirmed):
        status = "ready"
        reason = (
            "screen_change_pixels_matched"
            if visual_changed
            else (
                "screen_change_url_matched" if url_changed else "input_text_ocr_matched"
            )
        )
    elif not url_changed and not visual_changed:
        status = "unknown"
        reason = "no_screen_change"
    else:
        status = "unknown"
        reason = "transition_change_unverified"
    return TransitionDecision(
        request=evaluated_request,
        status=status,
        reason=reason,
        block_recipe=block_recipe,
    )


def _pending_transition_update(
    decision: TransitionDecision,
    *,
    visual_changed: bool,
    visual_ratio: float | None,
) -> WorkerStateUpdate:
    transition: TransitionPatch = {
        "transition_request": decision.request,
        "transition_result": transition_result(
            decision.request,
            status=decision.status,
            reason=decision.reason,
            visual_change_detected=visual_changed,
            visual_change_ratio=visual_ratio,
            needs_ocr=decision.needs_ocr,
        ),
    }
    return WorkerStateUpdate(transition=transition)


def _completed_transition_update(
    state: WorkerState,
    decision: TransitionDecision,
    *,
    visual_changed: bool,
    visual_ratio: float | None,
    ocr_skipped: bool,
    record_replay_result: Callable[[str, bool], object],
) -> WorkerStateUpdate:
    request = decision.request
    source = str(request.get("source") or "")
    observation: ObservationPatch = (
        reused_ocr_observation(state, request) if decision.reuse_previous_ocr else {}
    )
    record_state = (
        apply_worker_state_update(
            state,
            WorkerStateUpdate(observation=observation),
        )
        if observation
        else state
    )
    record = transition_record(
        request,
        status=decision.status,
        source=source,
        reason=decision.reason,
        attempt=1,
        state=record_state,
        visual_change_ratio=visual_ratio,
        ocr_skipped=ocr_skipped,
    )
    logger.info(
        "Transition evaluated",
        source=source,
        status=decision.status,
        reason=decision.reason,
        ocr_skipped=ocr_skipped,
    )
    record_replay_outcome(
        state,
        request,
        status=decision.status,
        persist_result=record_replay_result,
    )
    transition: TransitionPatch = {
        "transition_request": None,
        "transition_result": transition_result(
            request,
            status=decision.status,
            reason=decision.reason,
            visual_change_detected=visual_changed,
            visual_change_ratio=visual_ratio,
            needs_ocr=decision.needs_ocr,
        ),
        "action_events": attach_action_transition(
            state["transition"].get("action_events", []) or [],
            record,
        ),
    }
    replay: RecipeReplayPatch = {
        "reflex_blocked_recipe_keys": blocked_recipe_keys_after(
            state,
            request,
            should_block=decision.block_recipe,
        ),
        "replay_session": replay_session_after_transition(
            state,
            source=source,
            status=decision.status,
        ),
    }
    update: WorkerStateUpdate = {
        "transition": transition,
        "replay": replay,
    }
    if observation:
        update["observation"] = observation
    return update


def inspect_action_effect(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> WorkerStateUpdate:
    """행동 직후 캡처를 CV로 비교해 OCR 필요 여부를 결정한다."""

    request = state["transition"].get("transition_request")
    initial_result = transition_result_without_request(state, request)
    if initial_result is not None:
        return initial_result
    assert request is not None

    visual_changed, visual_ratio = transition_has_visual_change(
        request,
        str(state["observation"].get("current_screenshot") or ""),
    )
    decision = _decide_before_ocr(
        state,
        request,
        visual_changed=visual_changed,
    )
    if not decision.records_outcome:
        return _pending_transition_update(
            decision,
            visual_changed=visual_changed,
            visual_ratio=visual_ratio,
        )
    return _completed_transition_update(
        state,
        decision,
        visual_changed=visual_changed,
        visual_ratio=visual_ratio,
        ocr_skipped=not bool(state["observation"].get("ocr_complete")),
        record_replay_result=runtime.context.data.record_recipe_replay,
    )


def complete_action_effect(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> WorkerStateUpdate:
    """OCR가 준비된 현재 화면으로 직전 행동의 결과를 확정한다."""

    request = state["transition"].get("transition_request")
    if request is None:
        return {}
    if not state["observation"].get("ocr_complete"):
        raise RuntimeError("행동 결과를 확정하려면 현재 화면 OCR가 필요합니다.")

    pending_result = state["transition"].get("transition_result", {}) or {}
    if pending_result.get("status") != "needs_ocr":
        raise RuntimeError("OCR 확정 대상인 전환 요청이 아닙니다.")
    visual_changed = bool(pending_result.get("visual_change_detected"))
    visual_ratio = pending_result.get("visual_change_ratio")
    decision = _decide_after_ocr(
        state,
        request,
        visual_changed=visual_changed,
    )
    return _completed_transition_update(
        state,
        decision,
        visual_changed=visual_changed,
        visual_ratio=visual_ratio,
        ocr_skipped=False,
        record_replay_result=runtime.context.data.record_recipe_replay,
    )


__all__ = ["complete_action_effect", "inspect_action_effect"]

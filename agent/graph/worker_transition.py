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
from agent.runtime.site_context import (
    is_job_detail_context,
)
from agent.runtime.job_card_queue import (
    release_active_job_card,
)
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
from shared.schema.recipe_schema import TransitionStatus


@dataclass(frozen=True, slots=True)
class TransitionDecision:
    """현재 캡처에 대한 전이 판정과 필요한 상태 효과."""

    request: TransitionRequest
    status: TransitionStatus
    reason: str
    needs_ocr: bool = False
    records_outcome: bool = True
    block_recipe: bool = False
    release_queue: bool = False
    reuse_previous_ocr: bool = False
    reset_no_effect: bool = False


def _reflex_before_ocr_decision(
    state: WorkerState,
    request: TransitionRequest,
) -> TransitionDecision | None:
    """연속 경로의 다음 ROI 또는 마지막 URL을 원본 캡처로 확인한다."""

    if str(request.get("source") or "") != "reflex":
        return None
    transition_index = request.get("recipe_transition_index")
    transition_count = request.get("recipe_transition_count")
    if not isinstance(transition_index, int) or not isinstance(transition_count, int):
        return None
    has_following = transition_index + 1 < transition_count
    expected = request.get("expected_after_state")
    if expected is None or (has_following and not expected.has_anchor()):
        return None

    matched, reason, after_state_match = verify_replay_after_state(request, state)
    accepted_reason = (
        "recipe_after_anchor_matched"
        if has_following
        else "recipe_after_url_matched"
    )
    if not matched or reason != accepted_reason:
        return None

    evaluated_request = request.copy()
    evaluated_request["after_state_match"] = after_state_match
    return TransitionDecision(
        request=evaluated_request,
        status="ready",
        reason=reason,
        needs_ocr=not has_following,
        reset_no_effect=True,
    )


def _decide_before_ocr(
    state: WorkerState,
    request: TransitionRequest,
    *,
    visual_changed: bool,
) -> TransitionDecision:
    reflex_decision = _reflex_before_ocr_decision(state, request)
    if reflex_decision is not None:
        return reflex_decision
    source = str(request.get("source") or "")
    action = str(request.get("action") or "")
    if visual_changed or action == "type_in_marker":
        return TransitionDecision(
            request=request,
            status="needs_ocr",
            reason="ocr_required" if visual_changed else "input_ocr_required",
            needs_ocr=True,
            records_outcome=False,
            reset_no_effect=True,
        )
    return TransitionDecision(
        request=request,
        status="unknown",
        reason=(
            "reflex_no_screen_change" if source == "reflex" else "no_screen_change"
        ),
        block_recipe=source == "reflex",
        release_queue=source == "job_card_queue",
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
        release_queue=source == "job_card_queue" and status != "ready",
    )


def _pending_transition_update(
    state: WorkerState,
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
    if decision.reset_no_effect:
        transition["no_effect_count"] = 0
    update: WorkerStateUpdate = {"transition": transition}
    if decision.release_queue:
        update["collection"] = {
            "job_card_queue": release_active_job_card(
                list(state["collection"].get("job_card_queue", []) or [])
            )
        }
    return update


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
        "no_effect_count": (
            0
            if decision.status == "ready"
            else int(state["transition"].get("no_effect_count") or 0) + 1
        ),
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
    if decision.release_queue:
        update["collection"] = {
            "job_card_queue": release_active_job_card(
                list(state["collection"].get("job_card_queue", []) or [])
            )
        }
    return update


def transition_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> WorkerStateUpdate:
    """직전 원자 행동과 현재 캡처를 비교하고 OCR 필요 여부를 결정한다."""

    request = state["transition"].get("transition_request")
    initial_result = transition_result_without_request(state, request)
    if initial_result is not None:
        return initial_result
    assert request is not None

    visual_changed, visual_ratio = transition_has_visual_change(
        request,
        str(state["observation"].get("current_screenshot") or ""),
    )
    if not state["observation"].get("ocr_complete"):
        decision = _decide_before_ocr(
            state,
            request,
            visual_changed=visual_changed,
        )
    else:
        decision = _decide_after_ocr(
            state,
            request,
            visual_changed=visual_changed,
        )
    if not decision.records_outcome:
        return _pending_transition_update(
            state,
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


__all__ = ["transition_node"]

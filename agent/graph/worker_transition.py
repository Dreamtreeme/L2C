"""작업자 그래프의 화면 전환 판정 노드."""

from __future__ import annotations

from typing import Any

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
    queue_click_used_cached_marker,
    release_active_job_card,
)
from agent.runtime.worker_contracts import (
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


def _evaluate_before_ocr(
    state: WorkerState,
    request: TransitionRequest,
    *,
    visual_changed: bool,
    visual_ratio: float | None,
    record_replay_result,
) -> dict[str, Any]:
    source = str(request.get("source") or "")
    action = str(request.get("action") or "")
    input_requires_ocr = action == "type_in_marker"
    cached_queue_marker_failed = not visual_changed and queue_click_used_cached_marker(
        state, request
    )
    if cached_queue_marker_failed:
        refresh_request = request.copy()
        refresh_request["queue_marker_refresh"] = True
        return {
            "transition": {
                "transition_request": refresh_request,
                "transition_result": transition_result(
                    refresh_request,
                    status="needs_ocr",
                    reason="queue_cached_marker_refresh_required",
                    visual_change_ratio=visual_ratio,
                    needs_ocr=True,
                ),
            },
            "collection": {
                "job_card_queue": release_active_job_card(
                    list(state["collection"].get("job_card_queue", []) or [])
                )
            },
        }
    if visual_changed or input_requires_ocr:
        return {
            "transition": {
                "no_effect_count": 0,
                "transition_result": transition_result(
                    request,
                    status="needs_ocr",
                    reason=("ocr_required" if visual_changed else "input_ocr_required"),
                    visual_change_detected=visual_changed,
                    visual_change_ratio=visual_ratio,
                    needs_ocr=True,
                ),
            }
        }

    reason = "reflex_no_screen_change" if source == "reflex" else "no_screen_change"
    observation_update = reused_ocr_observation(state, request)
    record_state = apply_worker_state_update(
        state,
        WorkerStateUpdate(observation=observation_update),
    )
    record = transition_record(
        request,
        status="unknown",
        source=source,
        reason=reason,
        attempt=1,
        state=record_state,
        visual_change_ratio=visual_ratio,
        ocr_skipped=True,
    )
    logger.info(
        "Transition no-effect detected before OCR",
        source=source,
        action=request.get("action", ""),
        visual_change_ratio=visual_ratio,
    )
    record_replay_outcome(
        state,
        request,
        status="unknown",
        persist_result=record_replay_result,
    )
    update = {
        "transition": {
            "no_effect_count": (
                int(state["transition"].get("no_effect_count") or 0) + 1
            ),
            "transition_request": None,
            "transition_result": transition_result(
                request,
                status="unknown",
                reason=reason,
                visual_change_ratio=visual_ratio,
            ),
            "action_events": attach_action_transition(
                state["transition"].get("action_events", []) or [],
                record,
            ),
        },
        "replay": {
            "reflex_blocked_recipe_keys": blocked_recipe_keys_after(
                state,
                request,
                should_block=source == "reflex",
            ),
            "replay_session": replay_session_after_transition(
                state,
                source=source,
                status="unknown",
            ),
        },
        "observation": observation_update,
    }
    if source == "job_card_queue":
        update["collection"] = {
            "job_card_queue": release_active_job_card(
                list(state["collection"].get("job_card_queue", []) or [])
            )
        }
    return update


def _evaluate_after_ocr(
    state: WorkerState,
    request: TransitionRequest,
    *,
    visual_changed: bool,
    visual_ratio: float | None,
    record_replay_result,
) -> dict[str, Any]:
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

    evaluated_request: TransitionRequest
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
        evaluated_request = request
        status = "unknown"
        reason = "job_card_detail_not_reached"
        block_recipe = False
    elif markers and (url_changed or visual_changed or input_confirmed):
        evaluated_request = request
        status = "ready"
        reason = (
            "screen_change_pixels_matched"
            if visual_changed
            else (
                "screen_change_url_matched" if url_changed else "input_text_ocr_matched"
            )
        )
        block_recipe = False
    elif not url_changed and not visual_changed:
        evaluated_request = request
        status = "unknown"
        reason = "no_screen_change"
        block_recipe = False
    else:
        evaluated_request = request
        status = "unknown"
        reason = "transition_change_unverified"
        block_recipe = False

    attempt = 1
    record = transition_record(
        evaluated_request,
        status=status,
        source=source,
        reason=reason,
        attempt=attempt,
        state=state,
        visual_change_ratio=visual_ratio,
        ocr_skipped=False,
    )
    logger.info(
        "Transition evaluated",
        source=source,
        status=status,
        reason=reason,
    )
    record_replay_outcome(
        state,
        evaluated_request,
        status=status,
        persist_result=record_replay_result,
    )
    update = {
        "transition": {
            "no_effect_count": (
                0
                if status == "ready"
                else int(state["transition"].get("no_effect_count") or 0) + 1
            ),
            "transition_request": None,
            "transition_result": transition_result(
                evaluated_request,
                status=status,
                reason=reason,
                visual_change_detected=visual_changed,
                visual_change_ratio=visual_ratio,
            ),
            "action_events": attach_action_transition(
                state["transition"].get("action_events", []) or [],
                record,
            ),
        },
        "replay": {
            "reflex_blocked_recipe_keys": blocked_recipe_keys_after(
                state,
                request,
                should_block=block_recipe,
            ),
            "replay_session": replay_session_after_transition(
                state,
                source=source,
                status=status,
            ),
        },
    }
    if source == "job_card_queue" and status != "ready":
        update["collection"] = {
            "job_card_queue": release_active_job_card(
                list(state["collection"].get("job_card_queue", []) or [])
            )
        }
    return update


def transition_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> dict[str, Any]:
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
        return _evaluate_before_ocr(
            state,
            request,
            visual_changed=visual_changed,
            visual_ratio=visual_ratio,
            record_replay_result=runtime.context.data.record_recipe_replay,
        )
    return _evaluate_after_ocr(
        state,
        request,
        visual_changed=visual_changed,
        visual_ratio=visual_ratio,
        record_replay_result=runtime.context.data.record_recipe_replay,
    )


__all__ = ["transition_node"]

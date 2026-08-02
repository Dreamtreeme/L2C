"""작업자 그래프의 화면 전환 판정 노드."""

from __future__ import annotations

from typing import Any

from agent.graph.state import GraphState
from agent.graph.worker_transition_policy import (
    decide_after_ocr,
    decide_before_ocr,
    verify_reflex_after_state,
)
from agent.graph.worker_transition_state import (
    active_reflex_recipe_after_transition,
    blocked_recipe_keys,
    reused_observation,
    transition_record,
    transition_result,
)
from agent.runtime.transition_runtime import transition_has_visual_change
from agent.utils.logger import logger


def _result_without_transition(
    state: GraphState,
    request: dict[str, Any],
) -> dict[str, Any] | None:
    if state.get("low_information_screen"):
        return {
            "transition_result": transition_result(
                request,
                status="pending" if request else "idle",
                reason="low_information_screen",
            ),
        }
    if not request:
        return {
            "transition_result": transition_result(
                {},
                status="idle",
                reason="no_transition_request",
                needs_ocr=not bool(state.get("ocr_complete")),
            ),
        }
    return None


def _blocked_keys_after_decision(
    state: GraphState,
    request: dict[str, Any],
    *,
    should_block: bool,
) -> list[str]:
    keys = blocked_recipe_keys(state)
    recipe_key = str(request.get("recipe_key") or "")
    if should_block and recipe_key and recipe_key not in keys:
        keys.append(recipe_key)
    return keys


def _evaluate_before_ocr(
    state: GraphState,
    request: dict[str, Any],
    *,
    visual_changed: bool,
    visual_ratio: float | None,
) -> dict[str, Any]:
    source = str(request.get("source") or "")
    decision = decide_before_ocr(
        source=source,
        visual_changed=visual_changed,
    )
    if decision.status != "unknown":
        return {
            "transition_result": transition_result(
                request,
                status=decision.status,
                reason=decision.reason,
                visual_change_detected=visual_changed,
                visual_change_ratio=visual_ratio,
                needs_ocr=decision.needs_ocr,
            ),
        }

    attempt = int(request.get("attempts") or 0) + 1
    observation_update = reused_observation(state, request)
    record_state = {**state, **observation_update}
    record = transition_record(
        request,
        status="unknown",
        source=source,
        reason=decision.reason,
        attempt=attempt,
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
    return {
        "transition_request": {},
        "transition_result": transition_result(
            request,
            status="unknown",
            reason=decision.reason,
            visual_change_ratio=visual_ratio,
        ),
        "transition_records": [record],
        "reflex_blocked_recipe_keys": _blocked_keys_after_decision(
            state,
            request,
            should_block=decision.block_reflex_recipe,
        ),
        "active_reflex_recipe": active_reflex_recipe_after_transition(
            state,
            source=source,
            status="unknown",
        ),
        **observation_update,
    }


def _evaluate_after_ocr(
    state: GraphState,
    request: dict[str, Any],
    *,
    visual_changed: bool,
    visual_ratio: float | None,
) -> dict[str, Any]:
    source = str(request.get("source") or "")
    current_url = str(state.get("current_url") or "")
    before_url = str(request.get("before_url") or "")
    url_changed = bool(
        before_url
        and current_url
        and before_url != current_url
    )
    markers = list(state.get("current_markers") or [])

    if source == "reflex":
        matched, reason, after_state_match = verify_reflex_after_state(
            request,
            state,
        )
        request["after_state_match"] = after_state_match
        decision = decide_after_ocr(
            source=source,
            markers_present=bool(markers),
            url_changed=url_changed,
            visual_changed=visual_changed,
            reflex_matched=matched,
            reflex_reason=reason,
        )
    else:
        decision = decide_after_ocr(
            source=source,
            markers_present=bool(markers),
            url_changed=url_changed,
            visual_changed=visual_changed,
        )

    attempt = int(request.get("attempts") or 0) + 1
    record = transition_record(
        request,
        status=decision.status,
        source=source,
        reason=decision.reason,
        attempt=attempt,
        state=state,
        visual_change_ratio=visual_ratio,
        ocr_skipped=False,
    )
    logger.info(
        "Transition evaluated",
        source=source,
        status=decision.status,
        reason=decision.reason,
    )
    return {
        "transition_request": {},
        "transition_result": transition_result(
            request,
            status=decision.status,
            reason=decision.reason,
            visual_change_detected=visual_changed,
            visual_change_ratio=visual_ratio,
        ),
        "transition_records": [record],
        "reflex_blocked_recipe_keys": _blocked_keys_after_decision(
            state,
            request,
            should_block=decision.block_reflex_recipe,
        ),
        "active_reflex_recipe": active_reflex_recipe_after_transition(
            state,
            source=source,
            status=decision.status,
        ),
    }


def transition_node(state: GraphState) -> dict[str, Any]:
    """직전 원자 행동과 현재 캡처를 비교하고 OCR 필요 여부를 결정한다."""

    request = dict(state.get("transition_request", {}) or {})
    initial_result = _result_without_transition(state, request)
    if initial_result is not None:
        return initial_result

    visual_changed, visual_ratio = transition_has_visual_change(
        request,
        str(state.get("current_screenshot") or ""),
    )
    if not state.get("ocr_complete"):
        return _evaluate_before_ocr(
            state,
            request,
            visual_changed=visual_changed,
            visual_ratio=visual_ratio,
        )
    return _evaluate_after_ocr(
        state,
        request,
        visual_changed=visual_changed,
        visual_ratio=visual_ratio,
    )


__all__ = ["transition_node"]

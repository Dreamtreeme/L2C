"""작업자 그래프의 화면 전환 판정 노드."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.action_request import attach_action_transition
from agent.graph.state import GraphState
from agent.recipe.page_context import normalize_page_role
from agent.runtime.transition_runtime import (
    build_transition_observation,
    transition_has_visual_change,
)
from agent.utils.logger import logger


def _verify_reflex_after_state(
    request: dict[str, Any],
    state: GraphState,
) -> tuple[bool, str, dict[str, Any]]:
    """저장된 레시피 도착 화면과 현재 관찰이 같은지 확인한다."""

    if request.get("execution_failed"):
        return False, "recipe_action_group_failed", {}
    expected = dict(request.get("expected_after_state") or {})
    if not expected:
        return False, "recipe_after_state_missing", {}

    from agent.recipe.phash_replay import (
        match_step_by_screen_signature,
        screen_context_signature_match,
    )
    from agent.recipe.text_utils import recipe_url_scope_matches, url_template

    expected_url = str(expected.get("url_template") or "")
    current_url = str(state.get("current_url") or "")
    if (
        expected_url
        and current_url
        and not recipe_url_scope_matches(expected_url, current_url)
    ):
        return False, "recipe_after_url_mismatch", {
            "expected_url_template": expected_url,
            "current_url": current_url,
        }

    before_role = normalize_page_role(request.get("before_page_role"))
    expected_role = normalize_page_role(expected.get("page_role"))
    current_role = normalize_page_role(state.get("current_page_role"))
    if (
        before_role
        and expected_role
        and expected_role != before_role
        and current_role
    ):
        matched = current_role == expected_role
        return matched, (
            "recipe_after_page_role_matched"
            if matched
            else "recipe_after_page_role_mismatch"
        ), {
            "before_page_role": before_role,
            "expected_page_role": expected_role,
            "current_page_role": current_role,
        }

    anchor_target = expected.get("anchor_target")
    anchor_signature = dict(expected.get("anchor_roi_signature") or {})
    if isinstance(anchor_target, dict) and anchor_signature:
        marker_id, match = match_step_by_screen_signature(
            {
                "target": anchor_target,
                "roi_signature": anchor_signature,
            },
            dict(state.get("screen_signature") or {}),
            list(state.get("current_markers") or []),
            current_image_path=str(state.get("current_screenshot") or ""),
        )
        if marker_id is None:
            return False, str(
                match.get("reason") or "recipe_after_anchor_mismatch"
            ), match
        return True, "recipe_after_anchor_matched", {
            **match,
            "marker_id": marker_id,
        }

    context_signature = dict(expected.get("screen_context_signature") or {})
    if context_signature:
        match = screen_context_signature_match(
            context_signature,
            dict(state.get("screen_signature") or {}),
        )
        matched = bool(match.get("matched"))
        return matched, (
            "recipe_after_context_matched"
            if matched
            else str(match.get("reason") or "recipe_after_context_mismatch")
        ), match

    before_url = url_template(str(request.get("before_url") or ""))
    if (
        expected_url
        and current_url
        and expected_url != before_url
        and recipe_url_scope_matches(expected_url, current_url)
    ):
        return True, "recipe_after_url_matched", {
            "expected_url_template": expected_url,
            "current_url": current_url,
        }
    return False, "recipe_after_state_unverifiable", {}


def _blocked_recipe_keys(state: GraphState) -> list[str]:
    return [
        str(key)
        for key in (state.get("reflex_blocked_recipe_keys") or [])
        if str(key)
    ]


def _active_recipe_after_transition(
    state: GraphState,
    *,
    source: str,
    status: str,
) -> dict[str, Any]:
    active_recipe = dict(state.get("active_reflex_recipe", {}) or {})
    if not active_recipe or source != "reflex":
        return active_recipe
    if status != "ready":
        return {}
    current_index = int(active_recipe.get("current_transition_index") or 0)
    pending_index = active_recipe.get("pending_transition_index")
    if pending_index is None or int(pending_index) != current_index:
        return {}
    next_index = current_index + 1
    if next_index >= int(active_recipe.get("transition_count") or 0):
        return {}
    active_recipe["current_transition_index"] = next_index
    active_recipe.pop("pending_transition_index", None)
    return active_recipe


def _reused_observation(
    state: GraphState,
    request: dict[str, Any],
) -> dict[str, Any]:
    """변화가 없을 때 직전 캡처의 OCR만 동일 화면에 다시 연결한다."""

    previous = dict(state.get("previous_screen_observation") or {})
    if not previous:
        return {}
    if str(request.get("from_capture_id") or "") != str(
        previous.get("capture_id") or ""
    ):
        return {}
    if str(request.get("before_screenshot") or "") != str(
        previous.get("screenshot") or ""
    ):
        return {}
    before_url = str(request.get("before_url") or "")
    previous_url = str(previous.get("current_url") or "")
    if before_url and previous_url and before_url != previous_url:
        return {}

    markers = [
        dict(marker)
        for marker in previous.get("markers", []) or []
        if isinstance(marker, dict)
    ]
    if not markers:
        return {}
    signature = dict(previous.get("screen_signature") or {})
    raw_signature = dict(state.get("raw_screen_signature") or {})
    for key in ("phash", "size"):
        if raw_signature.get(key):
            signature[key] = raw_signature[key]
    current_observation = {
        **previous,
        "capture_id": str(state.get("current_capture_id") or ""),
        "screenshot": str(state.get("current_screenshot") or ""),
        "current_url": str(state.get("current_url") or previous_url),
        "markers": markers,
        "screen_signature": signature,
    }
    return {
        "current_markers": markers,
        "ui_context": str(previous.get("ui_context") or ""),
        "marked_image": str(previous.get("marked_image") or ""),
        "screen_signature": signature,
        "current_page_role": str(previous.get("page_role") or ""),
        "analysis_mode": str(previous.get("analysis_mode") or "full"),
        "ocr_complete": True,
        "ocr_capture_id": str(state.get("current_capture_id") or ""),
        "previous_screen_observation": current_observation,
    }


def _transition_record(
    request: dict[str, Any],
    *,
    status: str,
    source: str,
    reason: str,
    attempt: int,
    state: GraphState,
    visual_change_ratio: float | None,
    ocr_skipped: bool,
) -> dict[str, Any]:
    started_at = float(request.get("started_at") or time.time())
    return build_transition_observation(
        request,
        status=status,
        outcome="",
        source=source,
        reason=reason,
        elapsed_sec=max(0.0, time.time() - started_at),
        attempt=attempt,
        markers=list(state.get("current_markers", []) or []),
        screenshot=str(state.get("current_screenshot") or ""),
        marked_image=str(state.get("marked_image") or ""),
        to_capture_id=str(state.get("current_capture_id") or ""),
        current_url=str(state.get("current_url") or ""),
        page_role=str(state.get("current_page_role") or ""),
        screen_signature=dict(state.get("screen_signature") or {}),
        visual_change_ratio=visual_change_ratio,
        ocr_skipped=ocr_skipped,
    )


def _transition_result(
    request: dict[str, Any],
    *,
    status: str,
    reason: str = "",
    visual_change_detected: bool = False,
    visual_change_ratio: float | None = None,
    needs_ocr: bool = False,
) -> dict[str, Any]:
    return {
        **request,
        "status": status,
        "outcome": "",
        "reason": reason,
        "visual_change_detected": visual_change_detected,
        "visual_change_ratio": visual_change_ratio,
        "needs_ocr": needs_ocr,
    }


def _result_without_transition(
    state: GraphState,
    request: dict[str, Any],
) -> dict[str, Any] | None:
    if state.get("low_information_screen"):
        return {
            "transition_result": _transition_result(
                request,
                status="pending" if request else "idle",
                reason="low_information_screen",
            ),
        }
    if not request:
        return {
            "transition_result": _transition_result(
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
    keys = _blocked_recipe_keys(state)
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
    if visual_changed:
        return {
            "transition_result": _transition_result(
                request,
                status="needs_ocr",
                reason="ocr_required",
                visual_change_detected=True,
                visual_change_ratio=visual_ratio,
                needs_ocr=True,
            ),
        }

    reason = (
        "reflex_no_screen_change"
        if source == "reflex"
        else "no_screen_change"
    )
    attempt = 1
    observation_update = _reused_observation(state, request)
    record_state = {**state, **observation_update}
    record = _transition_record(
        request,
        status="unknown",
        source=source,
        reason=reason,
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
        "transition_result": _transition_result(
            request,
            status="unknown",
            reason=reason,
            visual_change_ratio=visual_ratio,
        ),
        "action_events": attach_action_transition(
            state.get("action_events", []) or [],
            record,
        ),
        "reflex_blocked_recipe_keys": _blocked_keys_after_decision(
            state,
            request,
            should_block=source == "reflex",
        ),
        "active_reflex_recipe": _active_recipe_after_transition(
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
        matched, reason, after_state_match = _verify_reflex_after_state(
            request,
            state,
        )
        request["after_state_match"] = after_state_match
        status = "ready" if matched else "unknown"
        block_recipe = not matched
    elif markers and (url_changed or visual_changed):
        status = "ready"
        reason = (
            "screen_change_pixels_matched"
            if visual_changed
            else "screen_change_url_matched"
        )
        block_recipe = False
    elif not url_changed and not visual_changed:
        status = "unknown"
        reason = "no_screen_change"
        block_recipe = False
    else:
        status = "unknown"
        reason = "transition_change_unverified"
        block_recipe = False

    attempt = 1
    record = _transition_record(
        request,
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
    return {
        "transition_request": {},
        "transition_result": _transition_result(
            request,
            status=status,
            reason=reason,
            visual_change_detected=visual_changed,
            visual_change_ratio=visual_ratio,
        ),
        "action_events": attach_action_transition(
            state.get("action_events", []) or [],
            record,
        ),
        "reflex_blocked_recipe_keys": _blocked_keys_after_decision(
            state,
            request,
            should_block=block_recipe,
        ),
        "active_reflex_recipe": _active_recipe_after_transition(
            state,
            source=source,
            status=status,
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

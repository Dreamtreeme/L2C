"""화면 전환 판정 결과를 GraphState와 trace 형식으로 조립한다."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.state import GraphState
from agent.runtime.transition_runtime import build_transition_observation


def blocked_recipe_keys(state: GraphState) -> list[str]:
    return [
        str(key)
        for key in (state.get("reflex_blocked_recipe_keys") or [])
        if str(key)
    ]


def active_reflex_recipe_after_transition(
    state: GraphState,
    *,
    source: str,
    status: str,
) -> dict[str, Any]:
    """도착 상태가 검증된 뒤에만 다음 전이로 이동한다."""

    active_recipe = dict(state.get("active_reflex_recipe", {}) or {})
    if not active_recipe or source != "reflex":
        return active_recipe
    if status != "ready":
        return {}
    current_index = int(
        active_recipe.get("current_transition_index") or 0
    )
    pending_index = active_recipe.get("pending_transition_index")
    if pending_index is None or int(pending_index) != current_index:
        return {}
    transition_count = int(
        active_recipe.get("transition_count") or 0
    )
    next_index = current_index + 1
    if transition_count <= 0 or next_index >= transition_count:
        return {}
    active_recipe["current_transition_index"] = next_index
    active_recipe.pop("pending_transition_index", None)
    return active_recipe


def reused_observation(
    state: GraphState,
    pending: dict[str, Any],
) -> dict[str, Any]:
    """pHash 무변화가 확인된 경우에만 행동 전 OCR 관찰을 현재 캡처에 연결한다."""

    previous = dict(state.get("previous_screen_observation") or {})
    if not previous:
        return {}

    expected_capture_id = str(pending.get("from_capture_id") or "")
    previous_capture_id = str(previous.get("capture_id") or "")
    if (
        not expected_capture_id
        or not previous_capture_id
        or expected_capture_id != previous_capture_id
    ):
        return {}

    before_screenshot = str(pending.get("before_screenshot") or "")
    previous_screenshot = str(previous.get("screenshot") or "")
    if (
        not before_screenshot
        or not previous_screenshot
        or before_screenshot != previous_screenshot
    ):
        return {}

    before_url = str(pending.get("before_url") or "")
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

    screen_signature = dict(previous.get("screen_signature") or {})
    raw_signature = dict(state.get("raw_screen_signature") or {})
    if raw_signature.get("phash"):
        screen_signature["phash"] = raw_signature["phash"]
    if raw_signature.get("size"):
        screen_signature["size"] = raw_signature["size"]

    current_observation = {
        **previous,
        "capture_id": str(state.get("current_capture_id") or ""),
        "screenshot": str(state.get("current_screenshot") or ""),
        "current_url": str(state.get("current_url") or previous_url),
        "markers": markers,
        "screen_signature": screen_signature,
    }
    return {
        "current_markers": markers,
        "ui_context": str(previous.get("ui_context") or ""),
        "marked_image": str(previous.get("marked_image") or ""),
        "screen_signature": screen_signature,
        "current_page_role": str(previous.get("page_role") or ""),
        "analysis_mode": str(previous.get("analysis_mode") or "full"),
        "ocr_complete": True,
        "ocr_capture_id": str(state.get("current_capture_id") or ""),
        "previous_screen_observation": current_observation,
    }


def transition_record(
    pending: dict[str, Any],
    *,
    status: str,
    outcome: str,
    source: str,
    reason: str,
    attempt: int,
    state: GraphState,
    phash_distance: int | None,
    visual_change_ratio: float | None,
    ocr_skipped: bool,
) -> dict[str, Any]:
    started_at = float(pending.get("started_at") or time.time())
    return build_transition_observation(
        pending,
        status=status,
        outcome=outcome,
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
        phash_distance=phash_distance,
        visual_change_ratio=visual_change_ratio,
        ocr_skipped=ocr_skipped,
    )


def transition_result(
    request: dict[str, Any],
    *,
    status: str,
    outcome: str = "",
    reason: str = "",
    visual_change_detected: bool = False,
    visual_change_ratio: float | None = None,
    needs_ocr: bool = False,
) -> dict[str, Any]:
    """전환 요청의 식별 정보와 판정 값을 하나의 결과로 합친다."""

    return {
        **request,
        "status": status,
        "outcome": outcome,
        "reason": reason,
        "visual_change_detected": visual_change_detected,
        "visual_change_ratio": visual_change_ratio,
        "needs_ocr": needs_ocr,
    }


__all__ = [
    "active_reflex_recipe_after_transition",
    "blocked_recipe_keys",
    "reused_observation",
    "transition_record",
    "transition_result",
]

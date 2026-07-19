"""물리 UI 도구가 실행 전에 확인할 대상 형태 계약."""

from __future__ import annotations

from typing import Any


IMPLAUSIBLE_TEXT_INPUT_TARGET = "implausible_text_input_target"


def _marker_by_id(markers: list[dict[str, Any]], marker_id: Any) -> dict[str, Any] | None:
    try:
        target_id = int(marker_id)
    except (TypeError, ValueError):
        return None
    return next(
        (
            marker
            for marker in markers
            if isinstance(marker, dict) and marker.get("id") == target_id
        ),
        None,
    )


def _valid_bbox(marker: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = marker.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _contains_text_marker(
    selected_bbox: tuple[float, float, float, float],
    markers: list[dict[str, Any]],
) -> bool:
    left, top, right, bottom = selected_bbox
    for marker in markers:
        if not isinstance(marker, dict) or str(marker.get("type") or "").lower() != "text":
            continue
        bbox = _valid_bbox(marker)
        if bbox is None:
            continue
        text_left, text_top, text_right, text_bottom = bbox
        center_x = (text_left + text_right) / 2
        center_y = (text_top + text_bottom) / 2
        if left <= center_x <= right and top <= center_y <= bottom:
            return True
    return False


def text_input_target_rejection(
    markers: list[dict[str, Any]],
    marker_id: Any,
) -> dict[str, Any] | None:
    """선택한 마커가 텍스트 입력 대상으로 보기 어려울 때만 거절 정보를 반환한다."""

    marker = _marker_by_id(markers, marker_id)
    if marker is None:
        return None
    marker_type = str(marker.get("type") or "").lower()
    if marker_type != "icon":
        return None

    bbox = _valid_bbox(marker)
    if bbox is None:
        return {
            "reason": IMPLAUSIBLE_TEXT_INPUT_TARGET,
            "marker_id": marker_id,
            "marker_type": marker_type,
            "aspect_ratio": 0.0,
        }

    left, top, right, bottom = bbox
    aspect_ratio = (right - left) / (bottom - top)
    if aspect_ratio >= 2.0 or _contains_text_marker(bbox, markers):
        return None

    return {
        "reason": IMPLAUSIBLE_TEXT_INPUT_TARGET,
        "marker_id": marker_id,
        "marker_type": marker_type,
        "aspect_ratio": round(aspect_ratio, 3),
    }


__all__ = ["IMPLAUSIBLE_TEXT_INPUT_TARGET", "text_input_target_rejection"]

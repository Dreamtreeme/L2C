"""저장한 화면 단서와 현재 OCR 마커를 결정론적으로 대응시킨다."""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.utils.text import normalize_text
from agent.vision.marker_geometry import (
    bbox_to_ratio,
    marker_bbox,
    marker_center_ratio,
)


def _normalized_key(value: Any) -> str:
    return normalize_text(value).casefold().replace(" ", "")


def anchor_overlap(saved: list[Any], current: list[Any]) -> float:
    """저장한 OCR 앵커 중 현재 화면에도 보이는 비율을 반환한다."""

    saved_set = {
        _normalized_key(item) for item in saved or [] if _normalized_key(item)
    }
    current_set = {
        _normalized_key(item) for item in current or [] if _normalized_key(item)
    }
    if not saved_set or not current_set:
        return 0.0
    return len(saved_set & current_set) / max(1, len(saved_set))


def _target_center_ratio(target: dict[str, Any]) -> list[float]:
    center = target.get("center_ratio") or []
    if isinstance(center, list) and len(center) == 2:
        return [float(center[0]), float(center[1])]
    bbox = target.get("bbox_ratio") or []
    if isinstance(bbox, list) and len(bbox) == 4:
        return [
            round((float(bbox[0]) + float(bbox[2])) / 2, 4),
            round((float(bbox[1]) + float(bbox[3])) / 2, 4),
        ]
    return []


def _distance(left: list[float], right: list[float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def _bbox_ratio(target: dict[str, Any]) -> list[float]:
    bbox = target.get("bbox_ratio") or []
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []
    try:
        return [float(value) for value in bbox]
    except (TypeError, ValueError):
        return []


def _bbox_iou(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_target_by_ratio(
    target: dict[str, Any] | None,
    markers: list[dict[str, Any]],
    screen_size: list[int],
) -> int | None:
    """저장 좌표와 종류·형상이 가장 잘 맞는 현재 마커를 선택한다."""

    target = target or {}
    target_center = _target_center_ratio(target)
    if len(target_center) != 2 or not screen_size or len(screen_size) != 2:
        return None
    max_distance = get_settings().reflex.target_center_max_distance
    target_bbox = _bbox_ratio(target)
    target_type = normalize_text(target.get("marker_type")).casefold()
    scored: list[tuple[float, float, int, str]] = []
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        current_center = marker_center_ratio(marker, screen_size)
        if len(current_center) != 2:
            continue
        distance = _distance(target_center, current_center)
        if distance > max_distance:
            continue
        try:
            marker_id = int(marker.get("id"))
        except (TypeError, ValueError):
            continue
        current_bbox = bbox_to_ratio(marker_bbox(marker), screen_size)
        marker_type = normalize_text(marker.get("type")).casefold()
        scored.append(
            (_bbox_iou(target_bbox, current_bbox), distance, marker_id, marker_type)
        )
    if not scored:
        return None
    same_type = [item for item in scored if target_type and item[3] == target_type]
    candidates = same_type or scored
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if len(candidates) > 1:
        first, second = candidates[:2]
        if abs(first[0] - second[0]) < 1e-9 and abs(first[1] - second[1]) < 1e-9:
            return None
    return candidates[0][2]


__all__ = ["anchor_overlap", "match_target_by_ratio"]

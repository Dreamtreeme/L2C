"""현재 화면 마커에서 행동 대상 스냅샷을 생성한다."""

from __future__ import annotations

from typing import Any, Mapping

from agent.vision.marker_geometry import (
    bbox_center,
    bbox_to_ratio,
    center_ratio_from_bbox,
    screen_size_from_signature,
)


TARGET_MARKER_ACTIONS = frozenset(
    {
        "click_marker",
        "type_in_marker",
        "scroll",
    }
)


def marker_by_id(
    markers: list[dict[str, Any]],
    marker_id: Any,
) -> dict[str, Any] | None:
    """현재 마커 목록에서 정수 ID가 일치하는 마커를 찾는다."""

    try:
        target_id = int(marker_id)
    except (TypeError, ValueError):
        return None
    return next(
        (
            marker
            for marker in markers or []
            if isinstance(marker, dict) and marker.get("id") == target_id
        ),
        None,
    )


def is_icon_marker(marker: Mapping[str, Any]) -> bool:
    """마커가 텍스트 입력 대상이 아닌 아이콘 검출 결과인지 판정한다."""

    text = str(marker.get("text") or "")
    marker_type = str(marker.get("type") or "").strip().lower()
    return (
        marker_type == "icon"
        or text == "icon"
        or text.startswith("상호작용 가능한 요소 (")
        or text == "상호작용 가능한 요소"
    )


def build_marker_target_snapshot(
    markers: list[dict[str, Any]],
    marker_id: Any,
    *,
    screen_signature: Mapping[str, Any] | None = None,
    target_label: Any = "",
) -> dict[str, Any] | None:
    """마커의 텍스트와 픽셀·비율 좌표를 동일한 계약으로 반환한다."""

    if marker_id is None:
        return None
    marker = marker_by_id(markers, marker_id)
    if marker is None:
        return {"marker_id": marker_id, "missing": True}

    raw_bbox = marker.get("bbox")
    bbox = list(raw_bbox) if isinstance(raw_bbox, (list, tuple)) else []
    center = list(bbox_center(bbox)) if len(bbox) == 4 else None
    snapshot = {
        "marker_id": marker.get("id"),
        "text": marker.get("text", ""),
        "marker_type": marker.get("type", ""),
        "bbox": bbox,
        "center": center,
    }

    size = screen_size_from_signature(dict(screen_signature or {}))
    if size and len(bbox) == 4:
        snapshot["bbox_ratio"] = bbox_to_ratio(bbox, size)
        snapshot["center_ratio"] = center_ratio_from_bbox(bbox, size)

    normalized_label = str(target_label or "").strip()
    if normalized_label:
        snapshot["target_label"] = normalized_label
    return snapshot


def build_action_target_snapshot(
    state: Mapping[str, Any],
    action_name: str,
    args: Mapping[str, Any],
) -> dict[str, Any] | None:
    """현재 행동이 마커 대상 행동이면 상태에서 타깃 스냅샷을 만든다."""

    if action_name not in TARGET_MARKER_ACTIONS or args.get("marker_id") is None:
        return None
    raw_observation = state.get("observation")
    observation = raw_observation if isinstance(raw_observation, Mapping) else {}
    markers = [
        marker
        for marker in observation.get("current_markers", []) or []
        if isinstance(marker, dict)
    ]
    return build_marker_target_snapshot(
        markers,
        args.get("marker_id"),
        screen_signature=(
            observation.get("screen_signature")
            if isinstance(observation.get("screen_signature"), Mapping)
            else {}
        ),
        target_label=args.get("target_label"),
    )


__all__ = [
    "TARGET_MARKER_ACTIONS",
    "build_action_target_snapshot",
    "build_marker_target_snapshot",
    "is_icon_marker",
    "marker_by_id",
]

"""OCR 마커의 bbox, 중심점, 화면 비율 좌표 계산 유틸리티."""

from __future__ import annotations

from typing import Any


def marker_bbox(marker: dict[str, Any]) -> list[int]:
    """마커의 bbox를 정수 리스트로 정규화한다."""
    raw = marker.get("bbox") if isinstance(marker, dict) else None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return [0, 0, 0, 0]
    try:
        return [int(value or 0) for value in raw]
    except (TypeError, ValueError):
        return [0, 0, 0, 0]


def bbox_center(bbox: list[int] | tuple[int, int, int, int]) -> tuple[int, int]:
    """bbox 중심 좌표를 픽셀 단위로 반환한다."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return (0, 0)
    try:
        x1, y1, x2, y2 = [int(value or 0) for value in bbox]
    except (TypeError, ValueError):
        return (0, 0)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def marker_center(marker: dict[str, Any]) -> tuple[int, int]:
    """마커 bbox 중심 좌표를 픽셀 단위로 반환한다."""
    return bbox_center(marker_bbox(marker))


def screen_size_from_signature(signature: dict[str, Any]) -> list[int]:
    """screen_signature에서 화면 크기 [width, height]를 안전하게 꺼낸다."""
    size = signature.get("size") if isinstance(signature, dict) else []
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return []
    try:
        return [int(size[0]), int(size[1])]
    except (TypeError, ValueError):
        return []


def _round_ratio(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def bbox_to_ratio(
    bbox: list[int] | tuple[int, int, int, int],
    size: list[int] | tuple[int, int],
) -> list[float]:
    """픽셀 bbox를 화면 크기 대비 비율 bbox로 변환한다."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return []
    try:
        width = max(1, int(size[0] or 0))
        height = max(1, int(size[1] or 0))
        x1, y1, x2, y2 = [float(value or 0) for value in bbox]
    except (TypeError, ValueError):
        return []
    return [
        _round_ratio(x1 / width),
        _round_ratio(y1 / height),
        _round_ratio(x2 / width),
        _round_ratio(y2 / height),
    ]


def center_ratio_from_bbox(
    bbox: list[int] | tuple[int, int, int, int],
    size: list[int] | tuple[int, int],
) -> list[float]:
    """픽셀 bbox 중심점을 화면 크기 대비 비율 좌표로 변환한다."""
    ratios = bbox_to_ratio(bbox, size)
    if len(ratios) != 4:
        return []
    return [round((ratios[0] + ratios[2]) / 2, 4), round((ratios[1] + ratios[3]) / 2, 4)]


def marker_center_ratio(marker: dict[str, Any], size: list[int] | tuple[int, int]) -> list[float]:
    """마커 중심점을 화면 크기 대비 비율 좌표로 반환한다."""
    return center_ratio_from_bbox(marker_bbox(marker), size)


def bbox_from_ratio(bbox_ratio: list[float], size: list[int] | tuple[int, int]) -> list[int]:
    """화면 비율 bbox를 픽셀 bbox로 변환한다."""
    if not isinstance(bbox_ratio, (list, tuple)) or len(bbox_ratio) != 4:
        return [0, 0, 0, 0]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return [0, 0, 0, 0]
    try:
        width = max(1, int(size[0] or 0))
        height = max(1, int(size[1] or 0))
        x1 = int(float(bbox_ratio[0]) * width)
        y1 = int(float(bbox_ratio[1]) * height)
        x2 = int(float(bbox_ratio[2]) * width)
        y2 = int(float(bbox_ratio[3]) * height)
    except (TypeError, ValueError):
        return [0, 0, 0, 0]
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 0, 0]
    return [x1, y1, x2, y2]


def rect_to_ratio(rect: list[float], size: list[int] | tuple[int, int]) -> list[float]:
    """픽셀 사각형을 화면 비율 사각형으로 변환한다."""
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return []
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return []
    try:
        width = max(1, int(size[0] or 0))
        height = max(1, int(size[1] or 0))
        x1, y1, x2, y2 = [float(value or 0) for value in rect]
    except (TypeError, ValueError):
        return []
    return [
        _round_ratio(x1 / width),
        _round_ratio(y1 / height),
        _round_ratio(x2 / width),
        _round_ratio(y2 / height),
    ]


def ratio_rect_to_pixels(rect_ratio: list[float], size: list[int] | tuple[int, int]) -> list[int]:
    """화면 비율 사각형을 픽셀 사각형으로 변환한다."""
    if not isinstance(rect_ratio, (list, tuple)) or len(rect_ratio) != 4:
        return [0, 0, 0, 0]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return [0, 0, 0, 0]
    try:
        width = max(1, int(size[0] or 0))
        height = max(1, int(size[1] or 0))
        x1 = int(round(max(0.0, min(1.0, float(rect_ratio[0]))) * width))
        y1 = int(round(max(0.0, min(1.0, float(rect_ratio[1]))) * height))
        x2 = int(round(max(0.0, min(1.0, float(rect_ratio[2]))) * width))
        y2 = int(round(max(0.0, min(1.0, float(rect_ratio[3]))) * height))
    except (TypeError, ValueError):
        return [0, 0, 0, 0]
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 0, 0]
    return [x1, y1, x2, y2]


def roi_rect_around_bbox(
    bbox: list[int] | tuple[int, int, int, int],
    size: list[int] | tuple[int, int],
    *,
    margin_scale: float = 1.0,
    min_width_ratio: float = 0.08,
    min_height_ratio: float = 0.04,
    max_width_ratio: float = 0.28,
    max_height_ratio: float = 0.12,
) -> list[float]:
    """타깃 bbox 주변의 안정적인 ROI를 화면 비율 좌표로 만든다."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return []
    try:
        width = max(1, int(size[0] or 0))
        height = max(1, int(size[1] or 0))
        x1, y1, x2, y2 = [float(value or 0) for value in bbox]
    except (TypeError, ValueError):
        return []
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    crop_w = min(width * max_width_ratio, max(box_w * (1 + margin_scale * 2), width * min_width_ratio))
    crop_h = min(height * max_height_ratio, max(box_h * (1 + margin_scale * 2), height * min_height_ratio))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    left = cx - crop_w / 2
    top = cy - crop_h / 2
    right = left + crop_w
    bottom = top + crop_h
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(width), right)
    bottom = min(float(height), bottom)
    if right <= left or bottom <= top:
        return []
    return rect_to_ratio([left, top, right, bottom], [width, height])

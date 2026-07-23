"""Reflex Recipe 마커 매칭."""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.recipe.page_context import normalize_page_role
from agent.recipe.text_utils import normalize_text
from agent.vision.marker_geometry import marker_bbox, marker_center


def marker_region(marker: dict, markers: list[dict]) -> str:
    """현재 마커 집합 안에서 coarse 3x3 영역을 계산한다."""
    if not marker:
        return ""
    xs = []
    ys = []
    for item in markers or []:
        if isinstance(item, dict):
            x, y = marker_center(item)
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return ""

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x, y = marker_center(marker)

    def band(value: int, low: int, high: int, names: tuple[str, str, str]) -> str:
        span = max(1, high - low)
        ratio = (value - low) / span
        if ratio < 1 / 3:
            return names[0]
        if ratio < 2 / 3:
            return names[1]
        return names[2]

    return f"{band(y, min_y, max_y, ('top', 'middle', 'bottom'))}-{band(x, min_x, max_x, ('left', 'center', 'right'))}"


def marker_ordinal(target_marker: dict, markers: list[dict]) -> int | None:
    """브라우저 상단을 제외한 동일 텍스트·영역 마커 중 순서(0-base)를 반환한다."""
    target_text = normalize_text(target_marker.get("text"))
    if not target_text:
        return None
    target_region = marker_region(target_marker, markers)
    matches = [
        marker
        for marker in markers or []
        if (
            isinstance(marker, dict)
            and normalize_text(marker.get("text")) == target_text
            and marker_region(marker, markers) == target_region
        )
    ]
    content_top = get_settings().reflex.interactive_content_top_px
    if marker_bbox(target_marker)[1] >= content_top:
        content_matches = [marker for marker in matches if marker_bbox(marker)[1] >= content_top]
        if content_matches:
            matches = content_matches
    matches = sorted(matches, key=lambda marker: (marker_bbox(marker)[1], marker_bbox(marker)[0], marker.get("id", 0)))
    for idx, marker in enumerate(matches):
        if marker.get("id") == target_marker.get("id"):
            return idx
    return None


def _step_get(step: Any, key: str, default: Any = None) -> Any:
    if isinstance(step, dict):
        return step.get(key, default)
    return getattr(step, key, default)


def _target_get(target: Any, key: str, default: Any = None) -> Any:
    if target is None:
        return default
    if isinstance(target, dict):
        return target.get(key, default)
    return getattr(target, key, default)


def _target_semantic_label(target: Any) -> str:
    return normalize_text(
        _target_get(target, "semantic_label", "")
        or _target_get(target, "target_label", "")
    )


def is_replayable_step(step: Any, params: dict | None = None) -> bool:
    """Return whether a cached step has enough generic data to replay."""
    if _step_get(step, "replay_mode", "reasoning") == "reasoning":
        return False
    action = _step_get(step, "action")
    if action not in {"click_marker", "type_in_marker"}:
        return True
    if not normalize_page_role(_step_get(step, "page_role", "")):
        return False
    target = _step_get(step, "target")
    return bool(normalize_text(_target_get(target, "text", "")) or _target_semantic_label(target))

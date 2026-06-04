"""Reflex Recipe 마커 매칭."""

from __future__ import annotations

from typing import Any

from agent.recipe.state_key import normalize_text


def _bbox(marker: dict) -> list[int]:
    raw = marker.get("bbox") or [0, 0, 0, 0]
    if len(raw) != 4:
        return [0, 0, 0, 0]
    return [int(v or 0) for v in raw]


def _center(marker: dict) -> tuple[int, int]:
    x1, y1, x2, y2 = _bbox(marker)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def marker_region(marker: dict, markers: list[dict]) -> str:
    """현재 마커 집합 안에서 coarse 3x3 영역을 계산한다."""
    if not marker:
        return ""
    xs = []
    ys = []
    for item in markers or []:
        if isinstance(item, dict):
            x, y = _center(item)
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return ""

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x, y = _center(marker)

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
    """동일 정규화 텍스트 마커 중 화면상 순서(0-base)를 반환한다."""
    target_text = normalize_text(target_marker.get("text"))
    if not target_text:
        return None
    matches = [
        marker
        for marker in markers or []
        if isinstance(marker, dict) and normalize_text(marker.get("text")) == target_text
    ]
    matches = sorted(matches, key=lambda marker: (_bbox(marker)[1], _bbox(marker)[0], marker.get("id", 0)))
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


def match_marker(step: Any, markers: list[dict], params: dict | None = None) -> int | None:
    """RecipeStep의 target을 현재 OCR 마커에 매칭해 marker_id를 반환한다."""
    target = _step_get(step, "target")
    target_text = normalize_text(_target_get(target, "text", ""))
    if not target_text:
        return None

    candidates = [
        marker for marker in markers or []
        if isinstance(marker, dict) and normalize_text(marker.get("text")) == target_text
    ]
    if not candidates:
        candidates = [
            marker for marker in markers or []
            if isinstance(marker, dict) and target_text in normalize_text(marker.get("text"))
        ]
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda marker: (_bbox(marker)[1], _bbox(marker)[0], marker.get("id", 0)))

    region = _target_get(target, "region")
    if region:
        region_matches = [marker for marker in candidates if marker_region(marker, markers) == region]
        if region_matches:
            candidates = region_matches

    ordinal = _target_get(target, "ordinal")
    if isinstance(ordinal, int) and 0 <= ordinal < len(candidates):
        return candidates[ordinal].get("id")

    return candidates[0].get("id")

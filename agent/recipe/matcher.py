"""Reflex Recipe 마커 매칭."""

from __future__ import annotations

import os
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
    try:
        content_top = int(os.getenv("VISION_INTERACTIVE_CONTENT_TOP_PX", "180"))
    except ValueError:
        content_top = 180
    if _bbox(target_marker)[1] >= content_top:
        content_matches = [marker for marker in matches if _bbox(marker)[1] >= content_top]
        if content_matches:
            matches = content_matches
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


def _target_evidence_texts(target: Any) -> list[str]:
    raw = _target_get(target, "evidence_texts", []) or []
    out = []
    for item in raw:
        text = normalize_text(item)
        if text and text not in out:
            out.append(text)
    return out


def _target_semantic_label(target: Any) -> str:
    return normalize_text(
        _target_get(target, "semantic_label", "")
        or _target_get(target, "target_label", "")
    )


def _text_matches(needle: str, haystack: str) -> bool:
    needle_norm = normalize_text(needle)
    haystack_norm = normalize_text(haystack)
    if not needle_norm or not haystack_norm:
        return False
    return needle_norm == haystack_norm or needle_norm in haystack_norm or haystack_norm in needle_norm


def _is_generic_marker_text(text: str) -> bool:
    return normalize_text(text).startswith("상호작용 가능한 요소")


def _compound_text_candidates(target_text: str, markers: list[dict]) -> list[dict]:
    """OCR이 한 문구를 인접한 여러 마커로 나눈 경우 왼쪽 마커를 후보로 찾는다."""
    needle = normalize_text(target_text).casefold().replace(" ", "")
    if len(needle) < 3:
        return []

    ordered = sorted(
        [marker for marker in markers or [] if isinstance(marker, dict)],
        key=lambda marker: (_bbox(marker)[1], _bbox(marker)[0], marker.get("id", 0)),
    )
    candidates = []
    seen_ids = set()
    for index, marker in enumerate(ordered):
        x1, y1, x2, y2 = _bbox(marker)
        parts = [normalize_text(marker.get("text"))]
        if not parts[0]:
            continue
        right_edge = x2
        for other in ordered[index + 1:index + 4]:
            ox1, oy1, ox2, oy2 = _bbox(other)
            if abs(((y1 + y2) // 2) - ((oy1 + oy2) // 2)) > 60:
                continue
            if ox1 < x1 or ox1 - right_edge > 240:
                continue
            text = normalize_text(other.get("text"))
            if not text:
                continue
            parts.append(text)
            right_edge = max(right_edge, ox2)
            combined = "".join(parts).casefold().replace(" ", "")
            if needle in combined or combined in needle:
                marker_id = marker.get("id")
                if marker_id not in seen_ids:
                    seen_ids.add(marker_id)
                    candidates.append(marker)
                break
    return candidates


def _nearby(anchor: dict, candidate: dict, max_dx: int = 650, max_dy: int = 180) -> bool:
    ax, ay = _center(anchor)
    cx, cy = _center(candidate)
    return abs(ax - cx) <= max_dx and abs(ay - cy) <= max_dy


def _evidence_score(candidate: dict, evidence_texts: list[str], markers: list[dict]) -> int:
    score = 0
    for evidence in evidence_texts:
        for marker in markers or []:
            if not isinstance(marker, dict) or marker.get("id") == candidate.get("id"):
                continue
            if _text_matches(evidence, marker.get("text", "")) and _nearby(marker, candidate):
                score += 1
                break
    return score


def _narrow_by_evidence(candidates: list[dict], target: Any, markers: list[dict]) -> list[dict] | None:
    evidence_texts = _target_evidence_texts(target)
    if not evidence_texts or len(candidates) <= 1:
        return candidates
    scored = [(_evidence_score(candidate, evidence_texts, markers), candidate) for candidate in candidates]
    best_score = max(score for score, _candidate in scored)
    if best_score <= 0:
        return None
    best = [candidate for score, candidate in scored if score == best_score]
    return best if len(best) == 1 else None

def is_replayable_step(step: Any, params: dict | None = None) -> bool:
    """Return whether a cached step has enough generic data to replay."""
    if _step_get(step, "replay_mode", "reasoning") == "reasoning":
        return False
    action = _step_get(step, "action")
    if action not in {"click_marker", "type_in_marker"}:
        return True
    target = _step_get(step, "target")
    return bool(normalize_text(_target_get(target, "text", "")) or _target_semantic_label(target))

def match_marker(step: Any, markers: list[dict], params: dict | None = None) -> int | None:
    """RecipeStep의 target을 현재 OCR 마커에 매칭해 marker_id를 반환한다."""
    target = _step_get(step, "target")
    semantic_label = _target_semantic_label(target)
    if semantic_label:
        label_candidates = [
            marker for marker in markers or []
            if isinstance(marker, dict) and normalize_text(marker.get("text")) == semantic_label
        ]
        if not label_candidates:
            label_candidates = [
                marker for marker in markers or []
                if isinstance(marker, dict) and semantic_label in normalize_text(marker.get("text"))
            ]
        if label_candidates:
            label_candidates = sorted(label_candidates, key=lambda marker: (_bbox(marker)[1], _bbox(marker)[0], marker.get("id", 0)))
            narrowed = _narrow_by_evidence(label_candidates, target, markers)
            if narrowed is not None:
                return narrowed[0].get("id")
            return None

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
        candidates = _compound_text_candidates(target_text, markers)
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda marker: (_bbox(marker)[1], _bbox(marker)[0], marker.get("id", 0)))

    region = _target_get(target, "region")
    if region and _is_generic_marker_text(target_text):
        region_matches = [marker for marker in candidates if marker_region(marker, markers) == region]
        if region_matches:
            candidates = region_matches

    ordinal = _target_get(target, "ordinal")
    if _is_generic_marker_text(target_text) and isinstance(ordinal, int) and 0 <= ordinal < len(candidates):
        try:
            content_top = int(os.getenv("VISION_INTERACTIVE_CONTENT_TOP_PX", "180"))
        except ValueError:
            content_top = 180
        content_candidates = [marker for marker in candidates if _bbox(marker)[1] >= content_top]
        if content_candidates:
            candidates = content_candidates
        if 0 <= ordinal < len(candidates):
            ordinal_candidate = candidates[ordinal]
            evidence_texts = _target_evidence_texts(target)
            if not evidence_texts or _evidence_score(ordinal_candidate, evidence_texts, markers) > 0:
                return ordinal_candidate.get("id")

    narrowed = _narrow_by_evidence(candidates, target, markers)
    if narrowed is None:
        return None
    candidates = narrowed

    if region and not _is_generic_marker_text(target_text):
        region_matches = [marker for marker in candidates if marker_region(marker, markers) == region]
        if region_matches:
            candidates = region_matches

    if isinstance(ordinal, int) and 0 <= ordinal < len(candidates):
        return candidates[ordinal].get("id")

    return candidates[0].get("id")

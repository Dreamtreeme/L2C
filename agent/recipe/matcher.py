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
        return None

    candidates = sorted(candidates, key=lambda marker: (_bbox(marker)[1], _bbox(marker)[0], marker.get("id", 0)))
    narrowed = _narrow_by_evidence(candidates, target, markers)
    if narrowed is None:
        return None
    candidates = narrowed

    region = _target_get(target, "region")
    if region:
        region_matches = [marker for marker in candidates if marker_region(marker, markers) == region]
        if region_matches:
            candidates = region_matches

    ordinal = _target_get(target, "ordinal")
    if isinstance(ordinal, int) and 0 <= ordinal < len(candidates):
        return candidates[ordinal].get("id")

    return candidates[0].get("id")

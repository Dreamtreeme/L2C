"""pHash + OCR 좌표비율로 Reflex replay 대상을 검증한다."""

from __future__ import annotations

import os
from typing import Any

from agent.recipe.state_key import normalize_text
from agent.vision.screen_signature import hamming_distance, marker_center_ratio


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


def _target_for_step(step: Any) -> Any:
    return _step_get(step, "target")


def _screen_signature_for_step(step: Any) -> dict[str, Any]:
    raw = _step_get(step, "screen_signature") or _step_get(step, "before_screen_signature") or {}
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    return dict(raw or {}) if isinstance(raw, dict) else {}


def _norm_key(value: Any) -> str:
    return normalize_text(value).casefold().replace(" ", "")


def anchor_overlap(saved: list[Any], current: list[Any]) -> float:
    saved_set = {_norm_key(item) for item in saved or [] if _norm_key(item)}
    current_set = {_norm_key(item) for item in current or [] if _norm_key(item)}
    if not saved_set or not current_set:
        return 0.0
    return len(saved_set & current_set) / max(1, len(saved_set))


def screen_signature_match(saved: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    max_distance = int(os.getenv("REFLEX_PHASH_MAX_DISTANCE", "10"))
    min_anchor_overlap = float(os.getenv("REFLEX_PHASH_MIN_ANCHOR_OVERLAP", "0.20"))
    distance = hamming_distance(str(saved.get("phash") or ""), str(current.get("phash") or ""))
    overlap = anchor_overlap(saved.get("anchors") or [], current.get("anchors") or [])
    if distance is None:
        return {"matched": False, "reason": "phash_missing", "distance": None, "anchor_overlap": overlap}
    if distance > max_distance:
        return {"matched": False, "reason": "phash_distance", "distance": distance, "anchor_overlap": overlap}
    if (saved.get("anchors") or []) and (current.get("anchors") or []) and overlap < min_anchor_overlap:
        return {"matched": False, "reason": "anchor_overlap", "distance": distance, "anchor_overlap": overlap}
    return {"matched": True, "reason": "matched", "distance": distance, "anchor_overlap": overlap}


def _target_center_ratio(target: Any) -> list[float]:
    center = _target_get(target, "center_ratio") or []
    if isinstance(center, list) and len(center) == 2:
        return [float(center[0]), float(center[1])]
    bbox = _target_get(target, "bbox_ratio") or []
    if isinstance(bbox, list) and len(bbox) == 4:
        return [round((float(bbox[0]) + float(bbox[2])) / 2, 4), round((float(bbox[1]) + float(bbox[3])) / 2, 4)]
    return []


def target_bbox_ratio(target: Any) -> list[float]:
    bbox = _target_get(target, "bbox_ratio") or []
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            return [float(item) for item in bbox]
        except Exception:
            return []
    return []


def bbox_from_ratio(
    bbox_ratio: list[float],
    screen_size: list[int] | tuple[int, int],
    *,
    padding_ratio: float = 0.0,
    min_padding_px: int = 0,
) -> list[int]:
    if len(bbox_ratio or []) != 4 or len(screen_size or []) != 2:
        return []
    width = max(1, int(screen_size[0] or 0))
    height = max(1, int(screen_size[1] or 0))
    x1 = int(max(0.0, min(1.0, float(bbox_ratio[0]))) * width)
    y1 = int(max(0.0, min(1.0, float(bbox_ratio[1]))) * height)
    x2 = int(max(0.0, min(1.0, float(bbox_ratio[2]))) * width)
    y2 = int(max(0.0, min(1.0, float(bbox_ratio[3]))) * height)
    pad_x = max(int(width * max(0.0, padding_ratio)), int(min_padding_px or 0))
    pad_y = max(int(height * max(0.0, padding_ratio)), int(min_padding_px or 0))
    return [
        max(0, min(width - 1, x1 - pad_x)),
        max(0, min(height - 1, y1 - pad_y)),
        max(1, min(width, x2 + pad_x)),
        max(1, min(height, y2 + pad_y)),
    ]


def synthetic_marker_from_target(target: Any, screen_size: list[int], marker_id: int = 0) -> dict[str, Any] | None:
    bbox = bbox_from_ratio(target_bbox_ratio(target), screen_size)
    if len(bbox) != 4:
        return None
    text = (
        _target_get(target, "semantic_label", "")
        or _target_get(target, "target_label", "")
        or _target_get(target, "text", "")
        or "reflex target"
    )
    return {
        "id": marker_id,
        "bbox": bbox,
        "text": str(text),
        "type": "reflex_target",
        "conf": 1.0,
    }


def _distance(left: list[float], right: list[float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def _text_match(left: Any, right: Any) -> bool:
    left_key = _norm_key(left)
    right_key = _norm_key(right)
    return bool(left_key and right_key and (left_key == right_key or left_key in right_key or right_key in left_key))


def _generic_target_text(value: Any) -> bool:
    key = _norm_key(value)
    return not key or key.startswith("상호작용가능한요소") or key in {"icon", "element", "미식별"}


def _target_validation_texts(target: Any) -> list[str]:
    out: list[str] = []
    for raw in (
        _target_get(target, "text", ""),
        _target_get(target, "semantic_label", ""),
        _target_get(target, "target_label", ""),
    ):
        text = normalize_text(raw)
        if text and not _generic_target_text(text) and text not in out:
            out.append(text)
    return out


def _target_evidence_texts(target: Any) -> list[str]:
    out: list[str] = []
    raw_items = _target_get(target, "evidence_texts", []) or []
    for raw in raw_items:
        text = normalize_text(raw)
        if len(_norm_key(text)) >= 2 and not _generic_target_text(text) and text not in out:
            out.append(text)
    return out


def roi_target_match(target: Any, roi_markers: list[dict[str, Any]]) -> dict[str, Any]:
    """ROI OCR 결과가 저장된 target 텍스트나 주변 anchor를 지지하는지 검사한다."""
    roi_texts = [
        normalize_text(marker.get("text"))
        for marker in roi_markers or []
        if isinstance(marker, dict) and normalize_text(marker.get("text"))
    ]
    if not roi_texts:
        return {"matched": False, "reason": "roi_ocr_empty", "roi_texts": []}

    direct_matches = [
        text
        for text in _target_validation_texts(target)
        if any(_text_match(text, roi_text) for roi_text in roi_texts)
    ]
    if direct_matches:
        return {
            "matched": True,
            "reason": "target_text",
            "matches": direct_matches,
            "roi_texts": roi_texts,
        }

    evidence_matches = [
        text
        for text in _target_evidence_texts(target)
        if any(_text_match(text, roi_text) for roi_text in roi_texts)
    ]
    min_evidence = int(os.getenv("REFLEX_FAST_ROI_MIN_ANCHOR_MATCHES", "1"))
    if len(evidence_matches) >= max(1, min_evidence):
        return {
            "matched": True,
            "reason": "evidence_anchor",
            "matches": evidence_matches,
            "roi_texts": roi_texts,
        }

    reason = "target_text_miss" if _target_validation_texts(target) else "target_anchor_miss"
    return {
        "matched": False,
        "reason": reason,
        "matches": evidence_matches,
        "roi_texts": roi_texts,
    }


def _text_match_score(target: Any, marker: dict[str, Any]) -> int:
    marker_text = _norm_key(marker.get("text"))
    candidates = [
        _target_get(target, "text", ""),
        _target_get(target, "semantic_label", ""),
        _target_get(target, "target_label", ""),
    ]
    for raw in candidates:
        key = _norm_key(raw)
        if key and marker_text and (key == marker_text or key in marker_text or marker_text in key):
            return 1
    if not any(_norm_key(raw) for raw in candidates):
        return 0
    return -1


def match_target_by_ratio(target: Any, markers: list[dict[str, Any]], screen_size: list[int]) -> int | None:
    target_center = _target_center_ratio(target)
    if len(target_center) != 2 or not screen_size or len(screen_size) != 2:
        return None
    max_distance = float(os.getenv("REFLEX_TARGET_CENTER_MAX_DISTANCE", "0.065"))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        current_center = marker_center_ratio(marker, screen_size)
        if len(current_center) != 2:
            continue
        distance = _distance(target_center, current_center)
        if distance > max_distance:
            continue
        text_score = _text_match_score(target, marker)
        if text_score < 0 and distance > max_distance / 2:
            continue
        scored.append((distance - (0.02 * text_score), int(marker.get("id") or 0), marker))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2].get("id")


def match_step_by_screen_signature(
    step: Any,
    current_signature: dict[str, Any],
    markers: list[dict[str, Any]],
) -> tuple[int | None, dict[str, Any]]:
    saved_signature = _screen_signature_for_step(step)
    signature_result = screen_signature_match(saved_signature, current_signature or {})
    if not signature_result.get("matched"):
        return None, signature_result
    marker_id = match_target_by_ratio(_target_for_step(step), markers, list((current_signature or {}).get("size") or []))
    if marker_id is None:
        signature_result = dict(signature_result)
        signature_result["matched"] = False
        signature_result["reason"] = "target_ratio_miss"
        return None, signature_result
    return marker_id, signature_result

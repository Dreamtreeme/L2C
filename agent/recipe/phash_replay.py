"""pHash + OCR 좌표비율로 Reflex replay 대상을 검증한다."""

from __future__ import annotations

import os
from typing import Any

from agent.recipe.state_key import normalize_text
from agent.utils.model_dump import dump_model
from agent.vision.marker_geometry import marker_center_ratio
from agent.vision.screen_signature import compute_roi_signature, hamming_distance


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


def _roi_signature_for_step(step: Any) -> dict[str, Any]:
    raw = _step_get(step, "roi_signature") or {}
    return dump_model(raw)


def _norm_key(value: Any) -> str:
    return normalize_text(value).casefold().replace(" ", "")


def anchor_overlap(saved: list[Any], current: list[Any]) -> float:
    saved_set = {_norm_key(item) for item in saved or [] if _norm_key(item)}
    current_set = {_norm_key(item) for item in current or [] if _norm_key(item)}
    if not saved_set or not current_set:
        return 0.0
    return len(saved_set & current_set) / max(1, len(saved_set))


# 전체 화면 pHash replay fallback은 ROI pHash 전환 후 비활성화했다.
# target 주변 ROI가 없는 오래된 레시피는 reasoning으로 폴백한다.


def roi_signature_match(saved: dict[str, Any], current_image_path: str) -> dict[str, Any]:
    max_distance = int(os.getenv("REFLEX_ROI_PHASH_MAX_DISTANCE", "22"))
    crop_rect_ratio = saved.get("crop_rect_ratio") or []
    if not current_image_path:
        return {"matched": False, "reason": "roi_current_image_missing", "distance": None, "mode": "roi_phash"}
    if not saved.get("phash") or not crop_rect_ratio:
        return {"matched": False, "reason": "roi_signature_missing", "distance": None, "mode": "roi_phash"}
    current = compute_roi_signature(current_image_path, crop_rect_ratio)
    distance = hamming_distance(str(saved.get("phash") or ""), str(current.get("phash") or ""))
    if distance is None:
        return {"matched": False, "reason": "roi_phash_missing", "distance": None, "mode": "roi_phash"}
    if distance > max_distance:
        return {
            "matched": False,
            "reason": "roi_phash_distance",
            "distance": distance,
            "max_distance": max_distance,
            "mode": "roi_phash",
            "crop_rect_ratio": crop_rect_ratio,
        }
    return {
        "matched": True,
        "reason": "roi_matched",
        "distance": distance,
        "max_distance": max_distance,
        "mode": "roi_phash",
        "crop_rect_ratio": crop_rect_ratio,
    }


def _target_center_ratio(target: Any) -> list[float]:
    center = _target_get(target, "center_ratio") or []
    if isinstance(center, list) and len(center) == 2:
        return [float(center[0]), float(center[1])]
    bbox = _target_get(target, "bbox_ratio") or []
    if isinstance(bbox, list) and len(bbox) == 4:
        return [round((float(bbox[0]) + float(bbox[2])) / 2, 4), round((float(bbox[1]) + float(bbox[3])) / 2, 4)]
    return []


def _distance(left: list[float], right: list[float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


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
    current_image_path: str = "",
) -> tuple[int | None, dict[str, Any]]:
    saved_roi_signature = _roi_signature_for_step(step)
    if not saved_roi_signature:
        return None, {
            "matched": False,
            "reason": "roi_signature_missing",
            "distance": None,
            "mode": "roi_phash",
        }
    signature_result = roi_signature_match(saved_roi_signature, current_image_path)
    if not signature_result.get("matched"):
        return None, signature_result
    marker_id = match_target_by_ratio(_target_for_step(step), markers, list((current_signature or {}).get("size") or []))
    if marker_id is None:
        signature_result = dict(signature_result)
        signature_result["matched"] = False
        signature_result["reason"] = "target_ratio_miss"
        return None, signature_result
    return marker_id, signature_result

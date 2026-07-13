"""pHash + OCR 좌표비율로 Reflex replay 대상을 검증한다."""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

from agent.recipe.text_utils import normalize_text
from agent.utils.model_dump import dump_model
from agent.vision.marker_geometry import marker_center_ratio
from agent.vision.marker_geometry import marker_bbox
from agent.vision.screen_signature import (
    build_capture_context,
    compute_roi_signature,
    compute_target_roi_signature_from_image,
    hamming_distance,
    image_dimensions,
)


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


def _capture_size(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return []
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return []


def capture_context_match(
    saved: dict[str, Any],
    current_signature: dict[str, Any],
    current_image_path: str,
) -> dict[str, Any]:
    """반응형 레이아웃이 달라질 정도의 캡처 크기 차이를 먼저 차단한다."""

    saved_context = dict(saved.get("capture_context") or {})
    saved_size = _capture_size(saved_context.get("size") or saved.get("source_size"))
    current_context = dict((current_signature or {}).get("capture_context") or {})
    current_size = _capture_size(current_context.get("size") or (current_signature or {}).get("size"))
    if not current_size and current_image_path:
        try:
            current_size = list(image_dimensions(current_image_path))
        except Exception:
            current_size = []
    if not saved_size or not current_size:
        return {"matched": True, "reason": "capture_context_unknown"}

    width_tolerance = max(0, int(os.getenv("REFLEX_CAPTURE_WIDTH_TOLERANCE_PX", "32")))
    height_tolerance = max(0, int(os.getenv("REFLEX_CAPTURE_HEIGHT_TOLERANCE_PX", "48")))
    if (
        abs(saved_size[0] - current_size[0]) > width_tolerance
        or abs(saved_size[1] - current_size[1]) > height_tolerance
    ):
        return {
            "matched": False,
            "reason": "capture_size_mismatch",
            "saved_size": saved_size,
            "current_size": current_size,
        }
    return {
        "matched": True,
        "reason": "capture_context_matched",
        "saved_size": saved_size,
        "current_size": current_size,
    }


def roi_signature_match(
    saved: dict[str, Any],
    current_image_path: str,
    current_signature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    max_distance = int(os.getenv("REFLEX_ROI_PHASH_MAX_DISTANCE", "22"))
    crop_rect_ratio = saved.get("crop_rect_ratio") or []
    if not current_image_path:
        return {"matched": False, "reason": "roi_current_image_missing", "distance": None, "mode": "roi_phash"}
    if not saved.get("phash") or not crop_rect_ratio:
        return {"matched": False, "reason": "roi_signature_missing", "distance": None, "mode": "roi_phash"}
    context_result = capture_context_match(saved, dict(current_signature or {}), current_image_path)
    if not context_result.get("matched"):
        return {**context_result, "distance": None, "mode": "roi_phash"}
    current_context = dict((current_signature or {}).get("capture_context") or {})
    current = compute_roi_signature(
        current_image_path,
        crop_rect_ratio,
        algorithm=str(saved.get("algorithm") or "roi-phash-dct64-v1"),
        capture_context=current_context,
    )
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


def _marker_centered_roi_match(
    saved: dict[str, Any],
    target: Any,
    markers: list[dict[str, Any]],
    screen_size: list[int],
    current_image_path: str,
    current_signature: dict[str, Any],
) -> tuple[int | None, dict[str, Any]]:
    """현재 마커 중심으로 작은 ROI를 다시 잘라 반응형 위치 이동을 흡수한다."""

    if str(saved.get("algorithm") or "") != "roi-phash-dct64-v2":
        return None, {"matched": False, "reason": "roi_marker_scan_unsupported", "mode": "roi_phash"}
    target_center = _target_center_ratio(target)
    if len(target_center) != 2 or len(screen_size) != 2 or not current_image_path:
        return None, {"matched": False, "reason": "roi_marker_scan_missing_context", "mode": "roi_phash"}

    max_center_distance = float(os.getenv("REFLEX_TARGET_SCAN_MAX_DISTANCE", "0.18"))
    max_phash_distance = int(os.getenv("REFLEX_ROI_PHASH_MAX_DISTANCE", "22"))
    min_margin = max(0, int(os.getenv("REFLEX_ROI_SCAN_MIN_MARGIN", "3")))
    target_type = _norm_key(_target_get(target, "marker_type", ""))
    capture_context = dict((current_signature or {}).get("capture_context") or {})
    scored: list[tuple[int, float, int]] = []
    try:
        with Image.open(current_image_path) as image:
            if not capture_context:
                capture_context = build_capture_context(image.size)
            for marker in markers or []:
                if not isinstance(marker, dict):
                    continue
                marker_type = _norm_key(marker.get("type"))
                if target_type and marker_type and target_type != marker_type:
                    continue
                current_center = marker_center_ratio(marker, screen_size)
                if len(current_center) != 2:
                    continue
                center_distance = _distance(target_center, current_center)
                if center_distance > max_center_distance:
                    continue
                current_roi = compute_target_roi_signature_from_image(
                    image,
                    marker_bbox(marker),
                    screen_size,
                    capture_context=capture_context,
                )
                distance = hamming_distance(str(saved.get("phash") or ""), str(current_roi.get("phash") or ""))
                if distance is None:
                    continue
                scored.append((distance, center_distance, int(marker.get("id") or 0)))
    except Exception:
        scored = []

    if not scored:
        return None, {"matched": False, "reason": "roi_marker_scan_no_candidate", "mode": "roi_phash"}
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    best = scored[0]
    if best[0] > max_phash_distance:
        return None, {
            "matched": False,
            "reason": "roi_marker_scan_distance",
            "distance": best[0],
            "max_distance": max_phash_distance,
            "mode": "roi_phash",
        }
    if len(scored) > 1 and scored[1][0] - best[0] < min_margin:
        return None, {
            "matched": False,
            "reason": "roi_marker_scan_ambiguous",
            "distance": best[0],
            "second_distance": scored[1][0],
            "min_margin": min_margin,
            "mode": "roi_phash",
        }
    return best[2], {
        "matched": True,
        "reason": "roi_marker_scan_matched",
        "distance": best[0],
        "target_distance": round(best[1], 4),
        "max_distance": max_phash_distance,
        "mode": "roi_phash",
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


def _nearby_icon_candidates(
    target: Any,
    markers: list[dict[str, Any]],
    screen_size: list[int],
    max_distance: float,
) -> list[dict[str, Any]]:
    target_center = _target_center_ratio(target)
    if len(target_center) != 2 or len(screen_size) != 2:
        return []
    candidates: list[dict[str, Any]] = []
    for marker in markers or []:
        if not isinstance(marker, dict) or _norm_key(marker.get("type")) != "icon":
            continue
        current_center = marker_center_ratio(marker, screen_size)
        if len(current_center) == 2 and _distance(target_center, current_center) <= max_distance:
            candidates.append(marker)
    return candidates


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
    screen_size = list((current_signature or {}).get("size") or [])
    signature_result = roi_signature_match(
        saved_roi_signature,
        current_image_path,
        current_signature=current_signature,
    )

    target = _target_for_step(step)
    target_type = _norm_key(_target_get(target, "marker_type", ""))
    roi_caption_enabled = os.getenv("REFLEX_ROI_CAPTION_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if signature_result.get("matched") and target_type == "icon" and roi_caption_enabled:
        strict_distance = float(os.getenv("REFLEX_TARGET_CENTER_MAX_DISTANCE", "0.065"))
        strict_candidates = _nearby_icon_candidates(target, markers, screen_size, strict_distance)
        if len(strict_candidates) == 1:
            marker_id = int(strict_candidates[0].get("id") or 0)
            return marker_id, {
                **signature_result,
                "matched": True,
                "mode": "roi_geometry",
                "reason": "single_nearby_icon",
                "candidate_ids": [marker_id],
            }

        scan_distance = float(os.getenv("REFLEX_TARGET_SCAN_MAX_DISTANCE", "0.18"))
        icon_candidates = _nearby_icon_candidates(target, markers, screen_size, scan_distance)
        if icon_candidates:
            try:
                from agent.vision.roi_caption import select_marker_by_roi_caption

                target_context = {
                    "label": str(
                        _target_get(target, "semantic_label", "")
                        or _target_get(target, "target_label", "")
                        or _target_get(target, "text", "")
                    ),
                    "component": str(_step_get(step, "component", "") or ""),
                    "intent": str(_step_get(step, "intent", "") or ""),
                }
                caption_marker_id, caption_result = select_marker_by_roi_caption(
                    current_image_path,
                    icon_candidates,
                    target_context,
                )
                if caption_marker_id is not None:
                    return caption_marker_id, {"matched": True, "mode": "roi_caption", **caption_result}
                return None, {"matched": False, "mode": "roi_caption", **caption_result}
            except Exception as exc:
                return None, {
                    "matched": False,
                    "mode": "roi_caption",
                    "reason": "roi_caption_failed",
                    "error": str(exc)[:200],
                }

    marker_id = None
    if signature_result.get("matched"):
        marker_id = match_target_by_ratio(_target_for_step(step), markers, screen_size)
    if marker_id is None and signature_result.get("reason") != "capture_size_mismatch":
        scanned_marker_id, scan_result = _marker_centered_roi_match(
            saved_roi_signature,
            _target_for_step(step),
            markers,
            screen_size,
            current_image_path,
            current_signature,
        )
        if scanned_marker_id is not None:
            return scanned_marker_id, scan_result
        if not signature_result.get("matched"):
            return None, signature_result
        signature_result = scan_result
    elif not signature_result.get("matched"):
        return None, signature_result
    if marker_id is None:
        signature_result = dict(signature_result)
        signature_result["matched"] = False
        signature_result["reason"] = "target_ratio_miss"
        return None, signature_result
    return marker_id, signature_result

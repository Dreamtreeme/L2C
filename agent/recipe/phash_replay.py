"""pHash + OCR 좌표비율로 Reflex replay 대상을 검증한다."""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.recipe.text_utils import normalize_text
from agent.vision.marker_geometry import bbox_to_ratio, marker_bbox, marker_center_ratio
from agent.vision.screen_signature import (
    compute_roi_signature,
    hamming_distance,
    image_dimensions,
)


def _norm_key(value: Any) -> str:
    return normalize_text(value).casefold().replace(" ", "")


def anchor_overlap(saved: list[Any], current: list[Any]) -> float:
    saved_set = {_norm_key(item) for item in saved or [] if _norm_key(item)}
    current_set = {_norm_key(item) for item in current or [] if _norm_key(item)}
    if not saved_set or not current_set:
        return 0.0
    return len(saved_set & current_set) / max(1, len(saved_set))


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
    saved_size = _capture_size(
        saved_context.get("size")
        or saved.get("source_size")
        or saved.get("size")
    )
    current_context = dict((current_signature or {}).get("capture_context") or {})
    current_size = _capture_size(current_context.get("size") or (current_signature or {}).get("size"))
    if not current_size and current_image_path:
        try:
            current_size = list(image_dimensions(current_image_path))
        except Exception:
            current_size = []
    if not saved_size or not current_size:
        return {"matched": True, "reason": "capture_context_unknown"}

    settings = get_settings().reflex
    width_tolerance = settings.capture_width_tolerance_px
    height_tolerance = settings.capture_height_tolerance_px
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
    max_distance = get_settings().reflex.roi_phash_max_distance
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


def screen_context_signature_match(
    saved: dict[str, Any],
    current_signature: dict[str, Any],
) -> dict[str, Any]:
    """좌표 없는 행동 직전 화면이 자율탐색 기록과 같은지 확인한다."""

    saved_phash = str((saved or {}).get("phash") or "")
    current_phash = str((current_signature or {}).get("phash") or "")
    if not saved_phash or not current_phash:
        return {
            "matched": False,
            "reason": "screen_context_signature_missing",
            "distance": None,
            "mode": "screen_context_phash",
        }
    context_result = capture_context_match(
        dict(saved or {}),
        dict(current_signature or {}),
        "",
    )
    if not context_result.get("matched"):
        return {
            **context_result,
            "distance": None,
            "mode": "screen_context_phash",
        }
    distance = hamming_distance(saved_phash, current_phash)
    if distance is None:
        return {
            "matched": False,
            "reason": "screen_context_phash_missing",
            "distance": None,
            "mode": "screen_context_phash",
        }
    max_distance = get_settings().reflex.screen_context_phash_max_distance
    return {
        "matched": distance <= max_distance,
        "reason": (
            "screen_context_matched"
            if distance <= max_distance
            else "screen_context_phash_distance"
        ),
        "distance": distance,
        "max_distance": max_distance,
        "mode": "screen_context_phash",
    }


def _target_center_ratio(target: dict[str, Any]) -> list[float]:
    center = target.get("center_ratio") or []
    if isinstance(center, list) and len(center) == 2:
        return [float(center[0]), float(center[1])]
    bbox = target.get("bbox_ratio") or []
    if isinstance(bbox, list) and len(bbox) == 4:
        return [round((float(bbox[0]) + float(bbox[2])) / 2, 4), round((float(bbox[1]) + float(bbox[3])) / 2, 4)]
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
        scored.append((_bbox_iou(target_bbox, current_bbox), distance, marker_id, marker_type))
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


def match_step_by_screen_signature(
    step: dict[str, Any],
    current_signature: dict[str, Any],
    markers: list[dict[str, Any]],
    current_image_path: str = "",
) -> tuple[int | None, dict[str, Any]]:
    saved_roi_signature = dict(step.get("roi_signature") or {})
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
    if not signature_result.get("matched"):
        return None, signature_result

    marker_id = match_target_by_ratio(step.get("target"), markers, screen_size)
    if marker_id is None:
        signature_result = dict(signature_result)
        signature_result["matched"] = False
        signature_result["reason"] = "target_ratio_miss"
        return None, signature_result
    return marker_id, signature_result

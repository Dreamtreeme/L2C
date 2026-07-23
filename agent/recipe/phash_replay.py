"""pHash + OCR 좌표비율로 Reflex replay 대상을 검증한다."""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.recipe.text_utils import normalize_text
from agent.utils.model_dump import dump_model
from agent.vision.marker_geometry import marker_center_ratio
from agent.vision.screen_signature import (
    compute_roi_signature,
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


def match_target_by_ratio(target: Any, markers: list[dict[str, Any]], screen_size: list[int]) -> int | None:
    """저장된 중심과 허용 반경 안에서 가장 가까운 마커를 선택한다."""

    target_center = _target_center_ratio(target)
    if len(target_center) != 2 or not screen_size or len(screen_size) != 2:
        return None
    max_distance = get_settings().reflex.target_center_max_distance
    scored: list[tuple[float, int]] = []
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        current_center = marker_center_ratio(marker, screen_size)
        if len(current_center) != 2:
            continue
        distance = _distance(target_center, current_center)
        if distance > max_distance:
            continue
        scored.append((distance, int(marker.get("id") or 0)))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][1]


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
    if not signature_result.get("matched"):
        return None, signature_result

    marker_id = match_target_by_ratio(_target_for_step(step), markers, screen_size)
    if marker_id is None:
        signature_result = dict(signature_result)
        signature_result["matched"] = False
        signature_result["reason"] = "target_ratio_miss"
        return None, signature_result
    return marker_id, signature_result

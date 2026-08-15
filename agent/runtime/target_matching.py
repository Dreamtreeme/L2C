"""저장한 화면 단서와 현재 OCR 마커를 결정론적으로 대응시킨다."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.config import get_settings
from agent.vision.screen_signature import (
    compute_roi_signature,
    hamming_distance,
)


def _capture_size(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return []
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return []


def capture_context_match(
    saved: dict[str, Any],
    current_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """반응형 레이아웃이 달라질 정도의 캡처 크기 차이를 차단한다."""

    saved_context = dict(saved.get("capture_context") or {})
    saved_size = _capture_size(
        saved_context.get("size") or saved.get("source_size") or saved.get("size")
    )
    current_context = dict(current_signature.get("capture_context") or {})
    current_size = _capture_size(
        current_context.get("size") or current_signature.get("size")
    )
    if not saved_size or not current_size:
        return {"matched": False, "reason": "capture_context_missing"}

    settings = get_settings().reflex
    if (
        abs(saved_size[0] - current_size[0]) > settings.capture_width_tolerance_px
        or abs(saved_size[1] - current_size[1]) > settings.capture_height_tolerance_px
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
    current_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """저장된 ROI와 현재 화면의 같은 영역을 pHash로 비교한다."""

    crop_rect_ratio = saved.get("crop_rect_ratio") or []
    if not current_image_path:
        return {
            "matched": False,
            "reason": "roi_current_image_missing",
            "distance": None,
            "mode": "roi_phash",
        }
    if not saved.get("phash") or not crop_rect_ratio:
        return {
            "matched": False,
            "reason": "roi_signature_missing",
            "distance": None,
            "mode": "roi_phash",
        }
    context_result = capture_context_match(
        saved,
        dict(current_signature or {}),
    )
    if not context_result.get("matched"):
        return {**context_result, "distance": None, "mode": "roi_phash"}

    current_context = dict((current_signature or {}).get("capture_context") or {})
    current = compute_roi_signature(
        current_image_path,
        crop_rect_ratio,
        algorithm=str(saved.get("algorithm") or "roi-phash-dct64-v1"),
        capture_context=current_context,
    )
    distance = hamming_distance(
        str(saved.get("phash") or ""),
        str(current.get("phash") or ""),
    )
    if distance is None:
        return {
            "matched": False,
            "reason": "roi_phash_missing",
            "distance": None,
            "mode": "roi_phash",
        }
    max_distance = get_settings().reflex.roi_phash_max_distance
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
    current_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """좌표 없는 행동 직전 화면이 자율탐색 기록과 같은지 확인한다."""

    saved_phash = str(saved.get("phash") or "")
    current_phash = str(current_signature.get("phash") or "")
    if not saved_phash or not current_phash:
        return {
            "matched": False,
            "reason": "screen_context_signature_missing",
            "distance": None,
            "mode": "screen_context_phash",
        }
    context_result = capture_context_match(saved, current_signature)
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
    matched = distance <= max_distance
    return {
        "matched": matched,
        "reason": (
            "screen_context_matched" if matched else "screen_context_phash_distance"
        ),
        "distance": distance,
        "max_distance": max_distance,
        "mode": "screen_context_phash",
    }


__all__ = [
    "capture_context_match",
    "roi_signature_match",
    "screen_context_signature_match",
]

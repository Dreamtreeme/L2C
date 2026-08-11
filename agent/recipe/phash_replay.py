"""저장된 Reflex 단계와 현재 OCR 마커를 대응시킨다."""

from __future__ import annotations

from typing import Any

from agent.runtime.target_matching import (
    match_target_by_ratio,
    roi_signature_match,
)


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

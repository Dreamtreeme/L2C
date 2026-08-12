"""저장된 Reflex 단계와 현재 OCR 마커를 대응시킨다."""

from __future__ import annotations

from typing import Any

from agent.runtime.target_matching import (
    match_target_by_ratio,
    roi_signature_match,
)
from agent.runtime.worker_contracts import ScreenMarker, ScreenSignature
from shared.schema.recipe_schema import ActionTarget, PhysicalAction


def match_target_by_screen_signature(
    target: ActionTarget | None,
    saved_roi_signature: dict[str, Any],
    current_signature: ScreenSignature,
    markers: list[ScreenMarker],
    current_image_path: str = "",
) -> tuple[int | None, dict[str, Any]]:
    """저장된 ROI와 대상 좌표를 현재 화면의 마커에 대응시킨다."""

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

    target_payload = target.model_dump(mode="json") if target else None
    marker_id = match_target_by_ratio(target_payload, markers, screen_size)
    if marker_id is None:
        signature_result = dict(signature_result)
        signature_result["matched"] = False
        signature_result["reason"] = "target_ratio_miss"
        return None, signature_result
    return marker_id, signature_result


def match_step_by_screen_signature(
    step: PhysicalAction,
    current_signature: ScreenSignature,
    markers: list[ScreenMarker],
    current_image_path: str = "",
) -> tuple[int | None, dict[str, Any]]:
    return match_target_by_screen_signature(
        step.target,
        step.roi_signature,
        current_signature,
        markers,
        current_image_path,
    )


__all__ = [
    "match_step_by_screen_signature",
    "match_target_by_screen_signature",
]

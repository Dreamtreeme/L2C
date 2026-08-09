"""추론에 사용한 화면과 실제 행동 직전 화면이 같은지 빠르게 검증한다."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.config import get_settings
from agent.runtime.worker_contracts import WorkerState
from agent.utils.logger import logger
from agent.vision.marker_geometry import marker_bbox
from agent.vision.screen_signature import (
    compute_roi_signature,
    compute_target_roi_signature,
    hamming_distance,
    perceptual_hash,
)


def _marker_for_id(
    state: WorkerState,
    marker_id: int | None,
) -> dict[str, Any] | None:
    if marker_id is None:
        return None
    for marker in state["observation"].get("current_markers", []) or []:
        if isinstance(marker, dict) and marker.get("id") == marker_id:
            return marker
    return None


def _target_roi_signature(
    state: WorkerState,
    marker_id: int | None,
) -> dict[str, Any]:
    marker = _marker_for_id(state, marker_id)
    observation = state["observation"]
    image_path = str(observation.get("current_screenshot") or "")
    screen_size = list(
        (observation.get("screen_signature") or {}).get("size") or []
    )
    if marker is None or not image_path or len(screen_size) != 2:
        return {}
    return compute_target_roi_signature(
        image_path,
        marker_bbox(marker),
        screen_size,
    )


def check_reasoning_screen_stale(
    state: WorkerState,
    perception: Any,
    *,
    marker_id: int | None = None,
) -> dict[str, Any]:
    """OCR 없이 행동 대상 ROI를 다시 계산해 오래된 마커 클릭을 차단한다."""

    target_signature = _target_roi_signature(state, marker_id)
    previous_phash = str(target_signature.get("phash") or "")
    mode = "target_roi"
    if not previous_phash:
        previous_phash = str(
            (state["observation"].get("screen_signature") or {}).get("phash")
            or ""
        )
        mode = "full_screen"
    if not previous_phash:
        return {
            "checked": False,
            "stale": False,
            "must_refresh": True,
            "reason": "previous_phash_missing",
        }

    filename = f"pre_action_{int(time.time() * 1000)}.png"
    image_path: Path | None = None
    try:
        image_path = Path(
            perception.capture_screen(
                filename=filename,
                initial_wait_sec=0,
                wait_for_stable=False,
            )
        )
        crop_rect_ratio = target_signature.get("crop_rect_ratio") or []
        if crop_rect_ratio:
            current_signature = compute_roi_signature(
                image_path,
                crop_rect_ratio,
                algorithm=str(target_signature.get("algorithm") or "roi-phash-dct64-v2"),
            )
            current_phash = str(current_signature.get("phash") or "")
        else:
            current_phash = perceptual_hash(image_path)
        distance = hamming_distance(previous_phash, current_phash)
    except Exception as exc:
        logger.debug("Reasoning screen guard skipped", error=str(exc))
        return {
            "checked": False,
            "stale": False,
            "must_refresh": True,
            "reason": "capture_failed",
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    finally:
        if image_path is not None and image_path.name == filename:
            try:
                image_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Reasoning screen guard temporary capture cleanup failed", error=str(exc))

    if distance is None:
        return {
            "checked": False,
            "stale": False,
            "must_refresh": True,
            "reason": "hash_comparison_failed",
        }

    max_distance = get_settings().vision.reasoning_stale_phash_max_distance
    stale = distance > max_distance
    result = {
        "checked": True,
        "stale": stale,
        "must_refresh": stale,
        "reason": "screen_changed_during_reasoning" if stale else "screen_unchanged",
        "distance": distance,
        "max_distance": max_distance,
        "mode": mode,
    }
    logger.info("Reasoning screen guard completed", **result)
    return result


__all__ = ["check_reasoning_screen_stale"]

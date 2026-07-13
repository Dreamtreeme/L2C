"""추론에 사용한 화면과 실제 행동 직전 화면이 같은지 빠르게 검증한다."""

from __future__ import annotations

import os
import time
from typing import Any

from agent.utils.logger import logger
from agent.vision.screen_signature import hamming_distance, perceptual_hash


def reasoning_screen_guard_enabled() -> bool:
    raw = os.getenv("VISION_REASONING_SCREEN_GUARD", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def check_reasoning_screen_stale(state: dict[str, Any], perception: Any) -> dict[str, Any]:
    """OCR 없이 현재 pHash만 다시 계산해 오래된 마커 클릭을 차단한다."""

    if not reasoning_screen_guard_enabled():
        return {"checked": False, "stale": False, "reason": "disabled"}
    previous_phash = str((state.get("screen_signature") or {}).get("phash") or "")
    if not previous_phash:
        return {"checked": False, "stale": False, "reason": "previous_phash_missing"}

    filename = f"pre_action_{int(time.time() * 1000)}.png"
    try:
        image_path = perception.capture_screen(
            filename=filename,
            initial_wait_sec=0,
            wait_for_stable=False,
        )
        current_phash = perceptual_hash(image_path)
        distance = hamming_distance(previous_phash, current_phash)
    except Exception as exc:
        logger.debug("Reasoning screen guard skipped", error=str(exc))
        return {"checked": False, "stale": False, "reason": "capture_failed"}

    max_distance = max(0, int(os.getenv("VISION_REASONING_STALE_PHASH_MAX_DISTANCE", "10")))
    stale = distance is not None and distance > max_distance
    result = {
        "checked": True,
        "stale": stale,
        "reason": "screen_changed_during_reasoning" if stale else "screen_unchanged",
        "distance": distance,
        "max_distance": max_distance,
        "image_path": str(image_path),
    }
    logger.info("Reasoning screen guard completed", **result)
    return result


__all__ = ["check_reasoning_screen_stale", "reasoning_screen_guard_enabled"]

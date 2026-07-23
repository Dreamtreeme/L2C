"""추론에 사용한 화면과 실제 행동 직전 화면이 같은지 빠르게 검증한다."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.config import get_settings
from agent.utils.logger import logger
from agent.vision.screen_signature import hamming_distance, perceptual_hash


def reasoning_screen_guard_enabled() -> bool:
    return get_settings().vision.reasoning_screen_guard


def check_reasoning_screen_stale(state: dict[str, Any], perception: Any) -> dict[str, Any]:
    """OCR 없이 현재 pHash만 다시 계산해 오래된 마커 클릭을 차단한다."""

    if not reasoning_screen_guard_enabled():
        return {"checked": False, "stale": False, "reason": "disabled"}
    previous_phash = str((state.get("screen_signature") or {}).get("phash") or "")
    if not previous_phash:
        return {"checked": False, "stale": False, "reason": "previous_phash_missing"}

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
        current_phash = perceptual_hash(image_path)
        distance = hamming_distance(previous_phash, current_phash)
    except Exception as exc:
        logger.debug("Reasoning screen guard skipped", error=str(exc))
        return {"checked": False, "stale": False, "reason": "capture_failed"}
    finally:
        if image_path is not None and image_path.name == filename:
            try:
                image_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Reasoning screen guard temporary capture cleanup failed", error=str(exc))

    max_distance = get_settings().vision.reasoning_stale_phash_max_distance
    stale = distance is not None and distance > max_distance
    result = {
        "checked": True,
        "stale": stale,
        "reason": "screen_changed_during_reasoning" if stale else "screen_unchanged",
        "distance": distance,
        "max_distance": max_distance,
    }
    logger.info("Reasoning screen guard completed", **result)
    return result


__all__ = ["check_reasoning_screen_stale", "reasoning_screen_guard_enabled"]

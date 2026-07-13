"""성공한 Reflex 후보를 사용자 요청과 분리해 비평·승격한다."""

from __future__ import annotations

import os
import queue
import threading
from typing import Any

from agent.utils.logger import logger


_PROMOTION_QUEUE: queue.Queue[str] = queue.Queue()
_PROMOTION_LOCK = threading.Lock()
_PROMOTION_EVENTS: dict[str, threading.Event] = {}
_PROMOTION_RESULTS: dict[str, dict[str, Any]] = {}
_QUEUED_CANDIDATES: set[str] = set()
_PROMOTION_THREAD: threading.Thread | None = None


def auto_promotion_enabled() -> bool:
    raw = os.getenv("VISION_RECIPE_AUTO_PROMOTE", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _promotion_worker() -> None:
    while True:
        candidate_id = _PROMOTION_QUEUE.get()
        try:
            from agent.recipe.candidate_reviewer import review_and_apply_candidate

            result = review_and_apply_candidate(candidate_id, mode="promote")
            logger.info(
                "Recipe candidate background promotion completed",
                candidate_id=candidate_id,
                decision=result.get("decision", ""),
                promoted=bool((result.get("promotion") or {}).get("promoted")),
                saved_count=int((result.get("promotion") or {}).get("saved_count") or 0),
            )
        except Exception as exc:
            result = {"decision": "revise", "error": str(exc)}
            logger.exception(
                "Recipe candidate background promotion failed",
                candidate_id=candidate_id,
                error=str(exc),
            )
        finally:
            with _PROMOTION_LOCK:
                _PROMOTION_RESULTS[candidate_id] = result
                _QUEUED_CANDIDATES.discard(candidate_id)
                event = _PROMOTION_EVENTS.setdefault(candidate_id, threading.Event())
                event.set()
            _PROMOTION_QUEUE.task_done()


def _ensure_promotion_thread() -> None:
    global _PROMOTION_THREAD
    with _PROMOTION_LOCK:
        if _PROMOTION_THREAD is not None and _PROMOTION_THREAD.is_alive():
            return
        _PROMOTION_THREAD = threading.Thread(
            target=_promotion_worker,
            name="recipe-promotion",
            daemon=True,
        )
        _PROMOTION_THREAD.start()


def schedule_recipe_candidate_promotion(candidate_id: str) -> bool:
    """후보 하나를 중복 없이 비동기 승격 대기열에 넣는다."""

    resolved = str(candidate_id or "").strip()
    if not resolved or not auto_promotion_enabled():
        return False
    with _PROMOTION_LOCK:
        if resolved in _QUEUED_CANDIDATES:
            return False
        _QUEUED_CANDIDATES.add(resolved)
        _PROMOTION_EVENTS[resolved] = threading.Event()
        _PROMOTION_RESULTS.pop(resolved, None)
    _ensure_promotion_thread()
    _PROMOTION_QUEUE.put(resolved)
    logger.info("Recipe candidate background promotion scheduled", candidate_id=resolved)
    return True


def wait_for_recipe_candidate_promotion(candidate_id: str, timeout: float = 60.0) -> dict[str, Any]:
    """벤치마크·테스트에서 특정 후보의 후처리 완료를 기다린다."""

    resolved = str(candidate_id or "").strip()
    with _PROMOTION_LOCK:
        event = _PROMOTION_EVENTS.get(resolved)
        existing = _PROMOTION_RESULTS.get(resolved)
    if existing is not None:
        return dict(existing)
    if event is None:
        return {"decision": "revise", "error": "candidate_not_scheduled"}
    if not event.wait(max(0.0, float(timeout))):
        return {"decision": "revise", "error": "promotion_timeout"}
    with _PROMOTION_LOCK:
        return dict(_PROMOTION_RESULTS.get(resolved) or {})


__all__ = [
    "auto_promotion_enabled",
    "schedule_recipe_candidate_promotion",
    "wait_for_recipe_candidate_promotion",
]

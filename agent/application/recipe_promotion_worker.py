"""백엔드 수명주기에서 실행되는 Reflex 후보 승격 작업자."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from agent.recipe.candidate_store import RecipeCandidateStore
from agent.utils.logger import logger


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


class RecipePromotionWorker:
    """SQLite 대기열을 한 번에 하나씩 처리하는 저우선순위 작업자."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        poll_interval_sec: float | None = None,
        retry_delay_sec: float | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else None
        self.poll_interval_sec = (
            _env_float("VISION_RECIPE_PROMOTION_POLL_SEC", 1.0)
            if poll_interval_sec is None
            else max(0.01, float(poll_interval_sec))
        )
        self.retry_delay_sec = (
            _env_float("VISION_RECIPE_PROMOTION_RETRY_DELAY_SEC", 30.0)
            if retry_delay_sec is None
            else max(0.0, float(retry_delay_sec))
        )
        self.max_attempts = (
            _env_int("VISION_RECIPE_PROMOTION_MAX_ATTEMPTS", 3)
            if max_attempts is None
            else max(1, int(max_attempts))
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> bool:
        if self.is_running:
            return False
        self._stop_event.clear()
        recovered = RecipeCandidateStore(self.db_path).recover_interrupted_reviews()
        self._thread = threading.Thread(
            target=self._run,
            name="recipe-promotion-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("Recipe promotion worker started", recovered_candidates=recovered)
        return True

    def stop(self, timeout_sec: float = 1.0) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(max(0.0, float(timeout_sec)))
        stopped = not thread.is_alive()
        logger.info("Recipe promotion worker stop requested", stopped=stopped)
        return stopped

    def process_one(self) -> dict[str, Any] | None:
        store = RecipeCandidateStore(self.db_path)
        candidate = store.claim_next_review()
        if candidate is None:
            return None
        candidate_id = str(candidate.get("candidate_id") or "")
        attempts = int(candidate.get("review_attempts") or 0)
        try:
            from agent.recipe.candidate_reviewer import review_and_apply_candidate

            result = review_and_apply_candidate(
                candidate_id,
                db_path=self.db_path,
                mode="promote",
                raise_on_critic_error=True,
            )
            logger.info(
                "Recipe candidate promotion reviewed",
                candidate_id=candidate_id,
                decision=result.get("decision", ""),
                promoted=bool((result.get("promotion") or {}).get("promoted")),
                saved_count=int((result.get("promotion") or {}).get("saved_count") or 0),
            )
            return result
        except Exception as exc:
            terminal = attempts >= self.max_attempts
            store.defer_review(
                candidate_id,
                str(exc),
                retry_delay_sec=self.retry_delay_sec,
                terminal=terminal,
            )
            logger.exception(
                "Recipe candidate promotion deferred" if not terminal else "Recipe candidate promotion failed",
                candidate_id=candidate_id,
                attempt=attempts,
                terminal=terminal,
                error=str(exc),
            )
            return {
                "candidate_id": candidate_id,
                "status": "review_failed" if terminal else "pending_review",
                "error": str(exc),
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            result = self.process_one()
            if result is None:
                self._stop_event.wait(self.poll_interval_sec)


_DEFAULT_WORKER_LOCK = threading.Lock()
_DEFAULT_WORKER: RecipePromotionWorker | None = None


def start_recipe_promotion_worker(
    db_path: str | Path | None = None,
) -> RecipePromotionWorker | None:
    """자동승격이 켜진 백엔드에서 장기 실행 작업자를 시작한다."""

    from agent.application.recipe_promotion_service import auto_promotion_enabled

    if not auto_promotion_enabled():
        return None
    global _DEFAULT_WORKER
    with _DEFAULT_WORKER_LOCK:
        if _DEFAULT_WORKER is None:
            _DEFAULT_WORKER = RecipePromotionWorker(db_path)
        _DEFAULT_WORKER.start()
        return _DEFAULT_WORKER


def stop_recipe_promotion_worker(timeout_sec: float = 1.0) -> bool:
    global _DEFAULT_WORKER
    with _DEFAULT_WORKER_LOCK:
        worker = _DEFAULT_WORKER
    if worker is None:
        return True
    return worker.stop(timeout_sec=timeout_sec)


__all__ = [
    "RecipePromotionWorker",
    "start_recipe_promotion_worker",
    "stop_recipe_promotion_worker",
]

"""백엔드 수명주기에서 실행되는 Reflex 후보 승격 작업자."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from agent.config import get_settings
from agent.recipe.candidate_store import RecipeCandidateStore
from agent.utils.logger import logger


def _review_metric_summary(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    metrics = dict(snapshot or {})
    llm = dict(metrics.get("llm") or {})
    totals = dict(llm.get("totals") or {})
    cost = dict(llm.get("cost") or {})
    return {
        "run_id": str(metrics.get("run_id") or ""),
        "duration_sec": float(metrics.get("duration_sec") or 0.0),
        "llm_call_count": len(llm.get("calls") or []),
        "input_tokens": int(totals.get("input_tokens") or 0),
        "output_tokens": int(totals.get("output_tokens") or 0),
        "total_tokens": int(totals.get("total_tokens") or 0),
        "estimated_cost": cost.get("estimated_total"),
        "unpriced_models": list(cost.get("unpriced_models") or []),
    }


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
        settings = get_settings().recipe
        self.poll_interval_sec = (
            settings.promotion_poll_sec
            if poll_interval_sec is None
            else max(0.01, float(poll_interval_sec))
        )
        self.retry_delay_sec = (
            settings.promotion_retry_delay_sec
            if retry_delay_sec is None
            else max(0.0, float(retry_delay_sec))
        )
        self.max_attempts = (
            settings.promotion_max_attempts
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

    def process_one(self, run_id: str | None = None) -> dict[str, Any] | None:
        store = RecipeCandidateStore(self.db_path)
        candidate = store.claim_review(run_id)
        if candidate is None:
            return None
        run_id = candidate.run_id
        attempts = candidate.review_attempts
        review_context = None
        try:
            from agent.application.recipe_candidate_review_service import (
                review_and_apply_candidate,
            )
            from agent.observability.run_context import run_context
            from shared.schema.run_schema import RunStatus

            site = candidate.site
            with run_context(
                query=f"recipe candidate {run_id}",
                prefix="recipe-promotion",
                metadata={
                    "run_id": run_id,
                    "site": site,
                    "review_attempt": attempts,
                },
                tags=["recipe-promotion", f"site:{site}" if site else "site:unknown"],
            ) as (review_context, _created):
                result = review_and_apply_candidate(
                    run_id,
                    db_path=self.db_path,
                    mode="promote",
                    raise_on_critic_error=True,
                )
                review_context.set_outcome(RunStatus.COMPLETED)
            review_metrics = _review_metric_summary(review_context.snapshot())
            result = {
                **dict(result),
                "run_id": run_id,
                "review_attempts": attempts,
                "review_metrics": review_metrics,
            }
            logger.info(
                "Recipe candidate promotion reviewed",
                run_id=run_id,
                decision=result.get("decision", ""),
                promoted=bool((result.get("promotion") or {}).get("promoted")),
                saved_count=int(
                    (result.get("promotion") or {}).get("saved_count") or 0
                ),
                duration_sec=review_metrics["duration_sec"],
                total_tokens=review_metrics["total_tokens"],
                estimated_cost=review_metrics["estimated_cost"],
            )
            return result
        except Exception as exc:
            review_metrics = _review_metric_summary(
                review_context.snapshot() if review_context is not None else None
            )
            terminal = attempts >= self.max_attempts
            store.defer_review(
                run_id,
                str(exc),
                retry_delay_sec=self.retry_delay_sec,
                terminal=terminal,
            )
            logger.exception(
                "Recipe candidate promotion deferred"
                if not terminal
                else "Recipe candidate promotion failed",
                run_id=run_id,
                attempt=attempts,
                terminal=terminal,
                error=str(exc),
                duration_sec=review_metrics["duration_sec"],
                total_tokens=review_metrics["total_tokens"],
                estimated_cost=review_metrics["estimated_cost"],
            )
            return {
                "run_id": run_id,
                "status": "review_failed" if terminal else "pending_review",
                "review_attempts": attempts,
                "error": str(exc),
                "review_metrics": review_metrics,
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            result = self.process_one()
            if result is None:
                self._stop_event.wait(self.poll_interval_sec)


__all__ = [
    "RecipePromotionWorker",
]

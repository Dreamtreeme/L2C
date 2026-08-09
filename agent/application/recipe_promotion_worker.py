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


def _aggregate_review_metrics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [dict(item.get("review_metrics") or {}) for item in attempts]
    costs = [
        float(item["estimated_cost"])
        for item in metrics
        if item.get("estimated_cost") is not None
    ]
    return {
        "attempt_count": len(attempts),
        "duration_sec": round(sum(float(item.get("duration_sec") or 0.0) for item in metrics), 6),
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in metrics),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in metrics),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in metrics),
        "estimated_cost": round(sum(costs), 10) if costs else None,
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

    def process_one(self, candidate_id: str | None = None) -> dict[str, Any] | None:
        store = RecipeCandidateStore(self.db_path)
        candidate = store.claim_review(candidate_id)
        if candidate is None:
            return None
        candidate_id = str(candidate.get("candidate_id") or "")
        attempts = int(candidate.get("review_attempts") or 0)
        review_context = None
        try:
            from agent.application.recipe_candidate_review_service import (
                review_and_apply_candidate,
            )
            from agent.observability.run_context import run_context
            from agent.observability.run_contracts import RunStatus

            site = str(candidate.get("site") or "")
            with run_context(
                query=f"recipe candidate {candidate_id}",
                prefix="recipe-promotion",
                metadata={
                    "candidate_id": candidate_id,
                    "site": site,
                    "review_attempt": attempts,
                },
                tags=["recipe-promotion", f"site:{site}" if site else "site:unknown"],
            ) as (review_context, _created):
                result = review_and_apply_candidate(
                    candidate_id,
                    db_path=self.db_path,
                    mode="promote",
                    raise_on_critic_error=True,
                )
                review_context.set_outcome(RunStatus.COMPLETED)
            review_metrics = _review_metric_summary(review_context.snapshot())
            result = {
                **dict(result),
                "candidate_id": candidate_id,
                "review_attempts": attempts,
                "review_metrics": review_metrics,
            }
            logger.info(
                "Recipe candidate promotion reviewed",
                candidate_id=candidate_id,
                decision=result.get("decision", ""),
                promoted=bool((result.get("promotion") or {}).get("promoted")),
                saved_count=int((result.get("promotion") or {}).get("saved_count") or 0),
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
                duration_sec=review_metrics["duration_sec"],
                total_tokens=review_metrics["total_tokens"],
                estimated_cost=review_metrics["estimated_cost"],
            )
            return {
                "candidate_id": candidate_id,
                "status": "review_failed" if terminal else "pending_review",
                "review_attempts": attempts,
                "error": str(exc),
                "review_metrics": review_metrics,
            }

    def process_candidate_until_settled(
        self,
        candidate_id: str,
        *,
        enqueue: bool = False,
    ) -> dict[str, Any]:
        """특정 후보만 설정된 횟수까지 처리하고 최종 DB 상태를 반환한다."""

        store = RecipeCandidateStore(self.db_path)
        candidate = store.get_candidate(candidate_id)
        if candidate is None:
            return {
                "candidate_id": candidate_id,
                "review_status": "not_found",
                "review_attempts": 0,
                "review_error": "",
                "validation": {},
                "attempts": [],
                "review_metrics": _aggregate_review_metrics([]),
            }

        status = str(candidate.get("status") or "")
        if status == "pending_replay" and enqueue:
            store.enqueue_review(candidate_id)
            candidate = store.get_candidate(candidate_id) or candidate
            status = str(candidate.get("status") or "")

        attempt_results: list[dict[str, Any]] = []
        if status in {"pending_review", "reviewing"}:
            for _attempt in range(self.max_attempts):
                result = self.process_one(candidate_id)
                if result is None:
                    break
                attempt_results.append(
                    {
                        "candidate_id": str(result.get("candidate_id") or ""),
                        "status": str(result.get("status") or ""),
                        "decision": str(result.get("decision") or ""),
                        "error": str(result.get("error") or "")[:300],
                        "review_metrics": dict(result.get("review_metrics") or {}),
                    }
                )
                candidate = store.get_candidate(candidate_id) or {}
                status = str(candidate.get("status") or "")
                if status not in {"pending_review", "reviewing"}:
                    break

        candidate = store.get_candidate(candidate_id) or {}
        return {
            "candidate_id": candidate_id,
            "review_status": str(candidate.get("status") or ""),
            "review_attempts": int(candidate.get("review_attempts") or 0),
            "review_error": str(candidate.get("review_error") or ""),
            "validation": dict(candidate.get("validation") or {}),
            "attempts": attempt_results,
            "review_metrics": _aggregate_review_metrics(attempt_results),
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            result = self.process_one()
            if result is None:
                self._stop_event.wait(self.poll_interval_sec)


__all__ = [
    "RecipePromotionWorker",
]

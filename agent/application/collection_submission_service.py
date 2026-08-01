"""수집 공고 저장과 Reflex 후보 등록을 확정한다."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent.application.job_persistence_service import persist_collected_data_with_report
from agent.application.run_context import raise_if_cancelled
from shared.schema.feedback_schema import WorkerSubmission

if TYPE_CHECKING:
    from agent.application.collection_worker_runner import WorkerRunResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalizedSubmission:
    submission: WorkerSubmission
    submission_id: str
    persistence: dict[str, Any]
    recipe_learning: dict[str, Any]


def _learning_result(
    status: str,
    *,
    candidate_id: str = "",
    reason: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "candidate_id": candidate_id,
        "reason": reason,
        "error": error,
    }


def _run_is_reusable(submission: WorkerSubmission, persistence: dict[str, Any]) -> bool:
    return bool(
        submission.run_status == "finished"
        and submission.is_finished
        and not submission.hit_recursion_limit
        and submission.recorded_steps
        and int(persistence.get("persisted_count") or 0) > 0
        and int(persistence.get("rejected_count") or 0) == 0
    )


def _record_recipe_candidate(
    submission: WorkerSubmission,
    persistence: dict[str, Any],
    submission_id: str,
) -> dict[str, Any]:
    if not _run_is_reusable(submission, persistence):
        return _learning_result("not_eligible")

    try:
        from agent.recipe.candidate_store import RecipeCandidateStore

        candidate_id = RecipeCandidateStore().commit_candidate(
            submission.model_dump(mode="json"),
            submission_id=submission_id,
        )
        if not candidate_id:
            return _learning_result("failed", reason="candidate_not_saved")

        from agent.application.recipe_promotion_service import (
            schedule_recipe_candidate_promotion,
        )

        scheduled = schedule_recipe_candidate_promotion(candidate_id)
        return _learning_result(
            "queued" if scheduled else "recorded",
            candidate_id=candidate_id,
        )
    except Exception as exc:
        logger.warning("레시피 후보 등록 실패: %s", exc)
        return _learning_result(
            "failed",
            reason="candidate_registration_failed",
            error=f"{type(exc).__name__}: {exc}"[:1000],
        )


def finalize_worker_submission(
    worker_result: "WorkerRunResult",
    source: str = "realtime_scraping",
) -> FinalizedSubmission:
    """공고, 실행 제출물과 재사용 후보를 한 번씩 저장한다."""

    raise_if_cancelled()
    submission = worker_result.submission
    extracted_jd = worker_result.extracted_jd
    persistence = (
        persist_collected_data_with_report(
            extracted_jd,
            collection_intent=submission.collection_intent.model_dump(mode="json"),
        )
        if extracted_jd
        else {}
    )
    submission = submission.model_copy(
        update={"persisted_count": int(persistence.get("persisted_count") or 0)}
    )

    from agent.recipe.submission_store import SubmissionStore

    submission_id = SubmissionStore().commit_submission(
        submission.model_dump(mode="json"),
        source=source,
    )
    recipe_learning = _record_recipe_candidate(
        submission,
        persistence,
        submission_id,
    )
    logger.info(
        "작업자 제출물 확정: id=%s persisted=%s",
        submission_id,
        submission.persisted_count,
    )
    return FinalizedSubmission(
        submission=submission,
        submission_id=submission_id,
        persistence=persistence,
        recipe_learning=recipe_learning,
    )


__all__ = ["FinalizedSubmission", "finalize_worker_submission"]

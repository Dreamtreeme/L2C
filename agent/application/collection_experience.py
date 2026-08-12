"""작업자 실행 기록과 경험 기반 탐색 후보를 공고 저장과 분리해 보존한다."""

from __future__ import annotations

import logging
from pathlib import Path

from agent.config import get_settings
from agent.recipe.candidate_store import RecipeCandidateStore
from agent.recipe.submission_store import SubmissionStore
from shared.schema.collection_run import (
    CollectionBatch,
    CollectionExperienceResult,
    PersistenceReport,
    RecipeLearningResult,
)
from shared.schema.feedback_schema import WorkerSubmission

logger = logging.getLogger(__name__)


def _learning_result(
    status: str,
    *,
    reason: str = "",
    error: str = "",
) -> RecipeLearningResult:
    return RecipeLearningResult(
        status=status,
        reason=reason,
        error=error,
    )


def _run_is_reusable(
    submission: WorkerSubmission,
    persistence: PersistenceReport,
) -> bool:
    return bool(
        submission.run_status == "finished"
        and submission.recorded_steps
        and persistence.persisted_count > 0
        and persistence.rejected_count == 0
    )


def _record_recipe_candidate(
    submission: WorkerSubmission,
    persistence: PersistenceReport,
    run_id: str,
    db_path: str | Path,
) -> RecipeLearningResult:
    if not _run_is_reusable(submission, persistence):
        return _learning_result("not_eligible")
    try:
        candidate_store = RecipeCandidateStore(db_path)
        recorded_run_id = candidate_store.commit_candidate(
            submission,
            run_id=run_id,
        )
        if not recorded_run_id:
            return _learning_result("failed", reason="candidate_not_saved")
        scheduled = bool(
            get_settings().recipe.auto_promote
            and candidate_store.enqueue_review(recorded_run_id)
        )
        return _learning_result("queued" if scheduled else "recorded")
    except Exception as exc:
        logger.warning("레시피 후보 등록 실패: %s", exc)
        return _learning_result(
            "failed",
            reason="candidate_registration_failed",
            error=f"{type(exc).__name__}: {exc}"[:1000],
        )


def record_collection_experience(
    batch: CollectionBatch,
    persistence: PersistenceReport,
    *,
    source: str = "realtime_scraping",
    db_path: str | Path,
) -> CollectionExperienceResult:
    """정보 저장 결과와 무관하게 실행 이력과 레시피 후보를 기록한다."""

    submission = batch.submission.model_copy(
        update={"persisted_count": persistence.persisted_count}
    )
    try:
        run_id = SubmissionStore(db_path).commit_submission(submission, source=source)
    except Exception as exc:
        logger.warning("작업자 제출물 저장 실패: %s", exc)
        return CollectionExperienceResult(
            run_id="",
            recipe_learning=_learning_result(
                "failed",
                reason="submission_registration_failed",
                error=f"{type(exc).__name__}: {exc}"[:1000],
            ),
        )
    return CollectionExperienceResult(
        run_id=run_id,
        recipe_learning=_record_recipe_candidate(
            submission,
            persistence,
            run_id,
            db_path,
        ),
    )


__all__ = ["record_collection_experience"]

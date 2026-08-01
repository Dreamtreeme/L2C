"""비전 작업자 제출물의 검토, 저장과 레시피 후보 등록을 담당한다."""

from __future__ import annotations

import logging
from typing import Any

from agent.application.job_persistence_service import (
    persist_collected_data_with_report,
)
from agent.application.run_context import raise_if_cancelled
from agent.config import get_settings

logger = logging.getLogger(__name__)


def commit_worker_review(
    submission: dict,
    source: str = "realtime_scraping",
) -> tuple[dict, str]:
    """작업자 제출물을 검토하고 판정과 함께 저장한다."""

    from agent.recipe.reviewer import review_worker_submission
    from agent.recipe.submission_store import SubmissionStore

    review = review_worker_submission(submission)
    submission_id = SubmissionStore().commit_submission(
        submission,
        review=review,
        source=source,
    )
    logger.info(
        "작업자 제출물 검토 완료: id=%s decision=%s confidence=%s",
        submission_id,
        review.get("decision"),
        review.get("confidence"),
    )
    return review, submission_id


def _recipe_learning_mode() -> str:
    mode = get_settings().recipe.learning_mode.strip().lower()
    # 작업 중에는 후보만 저장하고 검토와 승격은 별도 후처리에서 수행한다.
    return mode if mode in {"off", "record"} else "record"


def _recipe_learning_result(
    mode: str,
    status: str,
    *,
    reason: str = "",
    candidate_id: str = "",
    promotion_scheduled: bool = False,
    error: str = "",
) -> dict[str, Any]:
    return {
        "mode": mode,
        "status": status,
        "reason": reason,
        "candidate_id": candidate_id,
        "promotion_scheduled": promotion_scheduled,
        "error": error,
    }


def _commit_recipe_candidate(
    submission: dict,
    review: dict,
    source: str,
    submission_id: str,
    mode: str,
) -> dict[str, Any]:
    """학습 모드에 따라 Reflex 레시피 후보를 저장한다."""

    if mode == "off":
        return _recipe_learning_result(
            mode,
            "disabled",
            reason="learning_mode_off",
        )
    if not review.get("recipe_candidate"):
        return _recipe_learning_result(
            mode,
            "not_eligible",
            reason="critic_did_not_select_recipe_candidate",
        )
    try:
        from agent.recipe.candidate_store import RecipeCandidateStore

        candidate_id = RecipeCandidateStore().commit_candidate(
            submission,
            review=review,
            source=source,
            submission_id=submission_id,
        )
        candidate_id = str(candidate_id or "")
        if candidate_id:
            logger.info(
                "레시피 후보 저장: id=%s mode=%s",
                candidate_id,
                mode,
            )
        return _recipe_learning_result(
            mode,
            "recorded" if candidate_id else "failed",
            reason="" if candidate_id else "candidate_id_missing",
            candidate_id=candidate_id,
        )
    except Exception as exc:
        logger.warning("레시피 후보 저장 실패: %s", exc)
        return _recipe_learning_result(
            mode,
            "failed",
            reason="candidate_persistence_failed",
            error=f"{type(exc).__name__}: {exc}"[:1000],
        )


def _schedule_recipe_candidate_promotion(candidate_id: str) -> bool:
    """후보 승격을 작업자와 분리된 후처리 서비스에 맡긴다."""

    from agent.application.recipe_promotion_service import (
        schedule_recipe_candidate_promotion,
    )

    return schedule_recipe_candidate_promotion(candidate_id)


def _recipe_candidate_run_is_complete(submission: dict) -> bool:
    """정상 종료한 전체 작업만 경험 기반 탐색 후보로 허용한다."""

    return bool(
        submission.get("run_status") == "finished"
        and submission.get("is_finished")
        and not submission.get("hit_recursion_limit")
    )


def _validated_review(
    review: dict,
    validation: dict[str, Any],
) -> dict:
    persisted_count = int(validation.get("persisted_count") or 0)
    if persisted_count <= 0:
        return {
            **review,
            "decision": "reject",
            "reasons": list(review.get("reasons") or [])
            + ["all collected jobs failed pre-persistence validation"],
            "recipe_candidate": False,
        }
    rejected_count = int(validation.get("rejected_count") or 0)
    if rejected_count > 0:
        return {
            **review,
            "reasons": list(review.get("reasons") or [])
            + [
                f"{rejected_count} collected jobs failed "
                "pre-persistence validation"
            ],
            "recipe_candidate": False,
        }
    return review


def persist_accepted_worker_result(
    worker_result: dict,
    review: dict,
    source: str = "realtime_scraping",
) -> tuple[int, dict, dict, str]:
    """검토가 허용한 수집 데이터를 저장하고 제출 상태를 갱신한다."""

    raise_if_cancelled()
    submission = dict(worker_result.get("submission") or {})
    learning_mode = _recipe_learning_mode()
    accepts_data = bool(
        review.get("decision") == "accept"
        or review.get("accept_collected_data")
    )
    if not accepts_data or not worker_result.get("extracted_jd"):
        learning = (
            _recipe_learning_result(
                learning_mode,
                "disabled",
                reason="learning_mode_off",
            )
            if learning_mode == "off"
            else _recipe_learning_result(
                learning_mode,
                "not_eligible",
                reason=(
                    "review_not_accepted"
                    if not accepts_data
                    else "no_extracted_data"
                ),
            )
        )
        submission["recipe_learning"] = learning
        worker_result["submission"] = submission
        worker_result["recipe_learning"] = learning
        from agent.recipe.submission_store import SubmissionStore

        submission_id = SubmissionStore().commit_submission(
            submission,
            review=review,
            source=source,
        )
        return 0, submission, review, submission_id

    validation = persist_collected_data_with_report(
        worker_result.get("extracted_jd") or {},
        worker_result.get("keyword", ""),
        collection_intent=worker_result.get("collection_intent") or {},
    )
    persisted_count = int(validation.get("persisted_count") or 0)
    submission["persisted_count"] = persisted_count
    submission["persistence_validation"] = validation
    worker_result["submission"] = submission
    worker_result["persistence_validation"] = validation
    review = _validated_review(review, validation)

    from agent.recipe.submission_store import SubmissionStore

    submission_store = SubmissionStore()
    submission_id = submission_store.commit_submission(
        submission,
        review=review,
        source=source,
    )
    candidate_run_complete = _recipe_candidate_run_is_complete(submission)
    if learning_mode == "off":
        learning = _recipe_learning_result(
            learning_mode,
            "disabled",
            reason="learning_mode_off",
        )
    elif review.get("decision") != "accept":
        learning = _recipe_learning_result(
            learning_mode,
            "not_eligible",
            reason="review_not_accepted_after_validation",
        )
    elif not candidate_run_complete:
        learning = _recipe_learning_result(
            learning_mode,
            "not_eligible",
            reason="worker_run_incomplete",
        )
        logger.info(
            "레시피 후보 제외: run_status=%s is_finished=%s "
            "hit_recursion_limit=%s",
            submission.get("run_status"),
            bool(submission.get("is_finished")),
            bool(submission.get("hit_recursion_limit")),
        )
    else:
        learning = _commit_recipe_candidate(
            submission,
            review,
            source,
            submission_id,
            learning_mode,
        )
        candidate_id = str(learning.get("candidate_id") or "")
        if candidate_id:
            submission["recipe_candidate_id"] = candidate_id
            submission["recipe_learning_mode"] = learning_mode
            try:
                scheduled = _schedule_recipe_candidate_promotion(candidate_id)
            except Exception as exc:
                learning = {
                    **learning,
                    "reason": "promotion_schedule_failed",
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                }
                logger.warning("레시피 후보 승격 예약 실패: %s", exc)
            else:
                if scheduled:
                    learning = {
                        **learning,
                        "status": "queued",
                        "promotion_scheduled": True,
                    }
    submission["recipe_learning"] = learning
    worker_result["submission"] = submission
    worker_result["recipe_learning"] = learning
    submission_id = submission_store.commit_submission(
        submission,
        review=review,
        source=source,
    )
    return persisted_count, submission, review, submission_id


__all__ = [
    "commit_worker_review",
    "persist_accepted_worker_result",
]

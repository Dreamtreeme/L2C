"""상세 OCR 초안을 검토하고 수집 순회 상태를 확정하는 그래프 노드."""

from __future__ import annotations

from langgraph.runtime import Runtime

from agent.runtime.job_capture import store_job_capture
from agent.runtime.job_card_queue import (
    complete_active_job_card,
    job_card_queue_scope_complete,
    reject_active_job_card,
    resolved_job_card_count,
)
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.runtime.worker_contracts import (
    WorkerCompletionReason,
    WorkerState,
    WorkerStateUpdate,
)
from agent.runtime.worker_state import count_mode_from_state, target_count_from_state
from agent.utils.logger import logger
from shared.schema.jd_schema import (
    CollectedJob,
    JobCapture,
    JobCollectionEvidence,
    JobReview,
    JobReviewStatus,
)


def _store_collected_job(
    jobs: list[CollectedJob],
    collected_job: CollectedJob,
) -> list[CollectedJob]:
    url = str(collected_job.posting.url or "").strip().rstrip("/")
    updated = list(jobs)
    for index, existing in enumerate(updated):
        if str(existing.posting.url or "").strip().rstrip("/") == url:
            updated[index] = collected_job
            return updated
    updated.append(collected_job)
    return updated


def _review_history(state: WorkerState, review: JobReview) -> list[JobReview]:
    return [*state["collection"].get("job_reviews", []), review]


def _queue_finished(
    state: WorkerState,
    queue: list[dict],
    *,
    collected_count: int,
) -> bool:
    target_count = target_count_from_state(state)
    if target_count > 0 and max(
        collected_count,
        resolved_job_card_count(queue),
    ) >= target_count:
        return True
    return job_card_queue_scope_complete(
        queue,
        count_mode=count_mode_from_state(state),
        target_count=target_count,
    )


def _finished_update(state: WorkerState) -> WorkerStateUpdate:
    completion_reason: WorkerCompletionReason = (
        "visible_scope_completed"
        if count_mode_from_state(state) == "visible_all"
        else "target_reached"
    )
    return {
        "progress": {
            "stage": "finished",
        },
        "lifecycle": {
            "is_finished": True,
            "completion_reason": completion_reason,
        },
    }


def _complete_review(
    state: WorkerState,
    review: JobReview,
) -> WorkerStateUpdate:
    draft = state["collection"]["pending_job_draft"]
    if draft is None:
        return {}
    evidence = JobCollectionEvidence(
        required_fields=draft.required_fields,
        field_evidence=review.field_evidence,
        screenshot_path=draft.screenshot_path,
        source_card_key=draft.source_card_key,
    )
    capture = JobCapture(
        url=draft.url,
        raw_ocr_text=draft.raw_ocr_text,
        evidence=evidence,
    )
    collected_job = CollectedJob(posting=review.posting, evidence=evidence)
    captures = store_job_capture(
        list(state["collection"].get("job_captures", [])),
        capture,
    )
    jobs = _store_collected_job(
        list(state["collection"].get("collected_jobs", [])),
        collected_job,
    )
    queue = complete_active_job_card(
        list(state["collection"].get("job_card_queue", []))
    )
    update: WorkerStateUpdate = {
        "collection": {
            "job_captures": captures,
            "collected_jobs": jobs,
            "job_card_queue": queue,
            "job_detail_buffer": {},
            "pending_job_draft": None,
            "last_job_review": review,
            "job_reviews": _review_history(state, review),
        }
    }
    if _queue_finished(state, queue, collected_count=len(captures)):
        update.update(_finished_update(state))
    return update


def _continue_review(
    state: WorkerState,
    review: JobReview,
) -> WorkerStateUpdate:
    return {
        "collection": {
            "pending_job_draft": None,
            "last_job_review": review,
            "job_reviews": _review_history(state, review),
        }
    }


def _reject_review(
    state: WorkerState,
    review: JobReview,
) -> WorkerStateUpdate:
    queue = reject_active_job_card(
        list(state["collection"].get("job_card_queue", [])),
        reason=review.reason or review.status.value,
        url=review.url,
    )
    update: WorkerStateUpdate = {
        "collection": {
            "job_card_queue": queue,
            "job_detail_buffer": {},
            "pending_job_draft": None,
            "last_job_review": review,
            "job_reviews": _review_history(state, review),
        }
    }
    if _queue_finished(
        state,
        queue,
        collected_count=len(state["collection"].get("job_captures", [])),
    ):
        update.update(_finished_update(state))
    return update


def review_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> WorkerStateUpdate:
    """누적 상세 근거를 한 번 검토하고 해당 결과만 상태에 반영한다."""

    draft = state["collection"].get("pending_job_draft")
    if draft is None:
        return {}
    review = runtime.context.data.review_job_draft(
        draft,
        state["request"]["collection_intent"],
    )
    logger.info(
        "Worker job review applied",
        status=review.status.value,
        detail_key=review.detail_key,
        missing_fields=[field.value for field in review.missing_fields],
    )
    if review.status == JobReviewStatus.COMPLETE:
        return _complete_review(state, review)
    if review.status == JobReviewStatus.NEEDS_MORE:
        return _continue_review(state, review)
    return _reject_review(state, review)


__all__ = ["review_node"]

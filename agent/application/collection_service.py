"""수집 작업자 실행, 공고 저장과 결과 반환을 조율한다."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from agent.application.collection_submission_service import (
    FinalizedSubmission,
    finalize_worker_submission,
)
from agent.observability.run_context import emit_run_event, measure_step
from agent.observability.run_contracts import RunPhase
from agent.recipe.task_category import normalize_task_category
from agent.utils.logger import logger
from shared.schema.collection_intent import CollectionIntent, CollectionResult
from shared.schema.feedback_schema import WorkerSubmission

if TYPE_CHECKING:
    from agent.application.collection_worker_runner import WorkerRunResult


def failed_collection_result(
    message: str,
    *,
    error_code: str,
    intent: CollectionIntent,
) -> CollectionResult:
    return CollectionResult(
        status="failed",
        message=message,
        error_code=error_code,
        site=intent.site,
        search_keyword=intent.search_keyword,
        task_category=intent.task_category,
        target_count=intent.target_count,
    )


class CollectionService:
    """확정된 수집 의도를 실행하고 지휘자용 결과만 반환한다."""

    def __init__(
        self,
        run_worker: Callable[[CollectionIntent], "WorkerRunResult"],
        finalize_submission: Callable[["WorkerRunResult"], FinalizedSubmission] = finalize_worker_submission,
    ) -> None:
        self.run_worker = run_worker
        self.finalize_submission = finalize_submission

    @staticmethod
    def _scope_exhausted(
        submission: WorkerSubmission,
        resolved_count: int,
    ) -> bool:
        availability = submission.extracted_summary.get(
            "job_results_availability"
        ) or {}
        available_count = availability.get("available_job_count")
        confidence = float(availability.get("count_confidence") or 0.0)
        evidence = str(availability.get("count_evidence") or "").strip()
        return bool(
            isinstance(available_count, int)
            and available_count >= 0
            and confidence >= 0.8
            and evidence
            and resolved_count >= available_count
        )

    @staticmethod
    def _status(
        *,
        resolved_count: int,
        target_count: int,
        rejected_count: int,
        worker_finished: bool,
        scope_exhausted: bool,
    ) -> str:
        if resolved_count <= 0:
            return "failed"
        if rejected_count > 0:
            return "partial"
        if target_count > 0 and resolved_count < target_count and not scope_exhausted:
            return "partial"
        if not worker_finished and not scope_exhausted:
            return "partial"
        return "completed"

    def collect(self, intent: CollectionIntent) -> CollectionResult:
        if not intent.search_keyword.strip():
            return failed_collection_result(
                "collection failed: missing search keyword",
                error_code="missing_search_keyword",
                intent=intent,
            )
        if not intent.site.strip():
            return failed_collection_result(
                "collection failed: missing site",
                error_code="missing_site",
                intent=intent,
            )

        intent = intent.model_copy(
            update={"task_category": normalize_task_category(intent.task_category)}
        )
        emit_run_event(
            "collection_started",
            RunPhase.COLLECTION,
            "비전 작업자가 채용공고 수집을 시작했습니다.",
            data={
                "site": intent.site,
                "target_count": intent.target_count,
                "task_category": intent.task_category,
            },
        )
        try:
            with measure_step("vision_worker", site=intent.site):
                worker_result = self.run_worker(intent)
            with measure_step("job_persistence"):
                finalized = self.finalize_submission(worker_result)

            persistence = finalized.persistence
            persisted_items = list(persistence.get("persisted_items") or [])
            persisted_ids = {
                int(item["job_id"])
                for item in persisted_items
                if str(item.get("job_id") or "").isdigit()
            }
            observed_ids = {
                int(job_id)
                for job_id in finalized.submission.observed_job_ids
                if int(job_id) > 0
            }
            persisted_count = int(persistence.get("persisted_count") or 0)
            resolved_count = len(persisted_ids | observed_ids)
            scope_exhausted = self._scope_exhausted(
                finalized.submission,
                resolved_count,
            )
            worker_finished = finalized.submission.is_finished
            rejected_count = int(persistence.get("rejected_count") or 0)
            status = self._status(
                resolved_count=resolved_count,
                target_count=intent.target_count,
                rejected_count=rejected_count,
                worker_finished=worker_finished,
                scope_exhausted=scope_exhausted,
            )
            document_ids = sorted(persisted_ids | observed_ids)
            result = CollectionResult(
                status=status,
                message=(
                    f"collection {status}: keyword={intent.search_keyword!r}, "
                    f"site={worker_result.site_name or intent.site}, "
                    f"resolved={resolved_count}, persisted={persisted_count}"
                ),
                site=worker_result.site_slug or intent.site,
                site_name=worker_result.site_name or intent.site,
                search_keyword=intent.search_keyword,
                task_category=intent.task_category,
                target_count=intent.target_count,
                collected_count=finalized.submission.collected_count,
                resolved_count=resolved_count,
                persisted_count=persisted_count,
                created_count=int(persistence.get("created_count") or 0),
                updated_count=int(persistence.get("updated_count") or 0),
                rejected_count=rejected_count,
                persisted_items=persisted_items,
                observed_job_ids=sorted(observed_ids),
                document_ids=document_ids,
                scope_exhausted=scope_exhausted,
                worker_finished=worker_finished,
                hit_recursion_limit=finalized.submission.hit_recursion_limit,
                submission_id=finalized.submission_id,
                worker_run_id=finalized.submission.run_id,
                candidate_id=str(finalized.recipe_learning.get("candidate_id") or ""),
            )
            emit_run_event(
                "collection_completed",
                RunPhase.COLLECTION,
                "채용공고 수집과 저장을 마쳤습니다.",
                data={
                    "collection_status": result.status,
                    "site": result.site,
                    "target_count": result.target_count,
                    "resolved_count": result.resolved_count,
                    "persisted_count": result.persisted_count,
                    "submission_id": result.submission_id,
                },
            )
            return result
        except Exception as exc:
            from agent.observability.run_context import (
                ModelRequestTimeout,
                RunCancelled,
                RunDeadlineExceeded,
            )

            if isinstance(exc, (RunCancelled, RunDeadlineExceeded, ModelRequestTimeout)):
                raise
            logger.exception("Vision worker execution failed", error=str(exc))
            return failed_collection_result(
                f"collection error: {exc}",
                error_code=f"collection_error:{type(exc).__name__}",
                intent=intent,
            )


def create_collection_service(worker_runtime: Any) -> CollectionService:
    from agent.application.collection_worker_runner import run_worker_once
    from agent.application.worker_execution_service import WorkerExecutionService

    return CollectionService(
        WorkerExecutionService(worker_runtime, run_worker_once).run
    )


__all__ = [
    "CollectionService",
    "create_collection_service",
    "failed_collection_result",
]

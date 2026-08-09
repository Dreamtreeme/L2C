"""조사 계획의 비전 수집과 DB 저장 단계를 실행한다."""

from __future__ import annotations

from typing import Any

from agent.graph.investigation_context import InvestigationState
from agent.graph.investigation_ports import (
    CollectionPersistencePort,
    CollectionRunnerPort,
)
from agent.observability.run_context import (
    ModelRequestTimeout,
    RunCancelled,
    RunDeadlineExceeded,
    emit_run_event,
    measure_step,
    raise_if_cancelled,
)
from agent.observability.run_contracts import RunPhase
from agent.utils.logger import logger
from shared.schema.collection_intent import CollectionIntent, CollectionResult
from shared.schema.collection_run import CollectionBatch, PersistedCollection
from shared.schema.feedback_schema import WorkerSubmission
from shared.schema.investigation_schema import (
    InvestigationPlanStep,
    InvestigationStatus,
)


def _scope_exhausted(submission: WorkerSubmission, resolved_count: int) -> bool:
    availability = submission.extracted_summary.get("job_results_availability") or {}
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


def _collection_status(
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


def build_collection_result(
    intent: CollectionIntent,
    batch: CollectionBatch,
    persisted: PersistedCollection,
) -> CollectionResult:
    """작업자 관찰과 DB 저장 결과를 지휘자용 결과로 합친다."""

    report = persisted.persistence
    persisted_items = list(report.get("persisted_items") or [])
    persisted_ids = {
        int(item["job_id"])
        for item in persisted_items
        if str(item.get("job_id") or "").isdigit()
    }
    observed_ids = {
        int(job_id)
        for job_id in persisted.submission.observed_job_ids
        if int(job_id) > 0
    }
    document_ids = sorted(persisted_ids | observed_ids)
    resolved_count = len(document_ids)
    scope_exhausted = _scope_exhausted(persisted.submission, resolved_count)
    rejected_count = int(report.get("rejected_count") or 0)
    persisted_count = int(report.get("persisted_count") or 0)
    status = _collection_status(
        resolved_count=resolved_count,
        target_count=intent.target_count,
        rejected_count=rejected_count,
        worker_finished=persisted.submission.is_finished,
        scope_exhausted=scope_exhausted,
    )
    return CollectionResult(
        status=status,
        message=(
            f"collection {status}: keyword={intent.search_keyword!r}, "
            f"site={batch.site_name or intent.site}, "
            f"resolved={resolved_count}, persisted={persisted_count}"
        ),
        site=batch.site_slug or intent.site,
        site_name=batch.site_name or intent.site,
        search_keyword=intent.search_keyword,
        task_category=intent.task_category,
        target_count=intent.target_count,
        collected_count=persisted.submission.collected_count,
        resolved_count=resolved_count,
        persisted_count=persisted_count,
        created_count=int(report.get("created_count") or 0),
        updated_count=int(report.get("updated_count") or 0),
        rejected_count=rejected_count,
        persisted_items=persisted_items,
        observed_job_ids=sorted(observed_ids),
        document_ids=document_ids,
        scope_exhausted=scope_exhausted,
        worker_finished=persisted.submission.is_finished,
        hit_recursion_limit=persisted.submission.hit_recursion_limit,
        submission_id=persisted.submission_id,
        worker_run_id=persisted.submission.run_id,
        candidate_id=str(persisted.recipe_learning.get("candidate_id") or ""),
    )


def _failed_result(
    intent: CollectionIntent,
    message: str,
    *,
    error_code: str,
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


class InvestigationCollectionNodes:
    """수집과 저장을 별도 체크포인트 단계로 실행한다."""

    def __init__(
        self,
        run_collection: CollectionRunnerPort,
        persist_collection: CollectionPersistencePort,
    ) -> None:
        self.run_collection = run_collection
        self.persist_collection = persist_collection

    @staticmethod
    def _active_step(state: InvestigationState) -> InvestigationPlanStep:
        investigation = state["request"]["investigation"]
        active_step_id = state["execution"].get("active_step_id", "")
        if active_step_id:
            return next(
                item for item in investigation.plan if item.step_id == active_step_id
            )
        return next(
            item
            for item in investigation.plan
            if item.step_id not in investigation.executed_step_ids
        )

    def collect(self, state: InvestigationState) -> dict[str, Any]:
        raise_if_cancelled()
        step = self._active_step(state)
        emit_run_event(
            "collection_started",
            RunPhase.COLLECTION,
            step.purpose or "계획한 채용공고 수집을 실행하고 있습니다.",
        )
        try:
            with measure_step("vision_worker", site=step.arguments.site):
                batch = self.run_collection(step.arguments)
            error = ""
        except (RunCancelled, RunDeadlineExceeded, ModelRequestTimeout):
            raise
        except Exception as exc:
            logger.exception("Vision worker execution failed", error=str(exc))
            batch = None
            error = f"{type(exc).__name__}: {exc}"
        return {
            "request": {
                "investigation": state["request"]["investigation"].model_copy(
                    update={"status": InvestigationStatus.PERSISTING}
                )
            },
            "execution": {
                "active_step_id": step.step_id,
                "pending_collection": batch,
                "collection_error": error,
            },
        }

    def persist(self, state: InvestigationState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = state["request"]["investigation"]
        step = self._active_step(state)
        batch = state["execution"].get("pending_collection")
        error = state["execution"].get("collection_error", "")
        if batch is None:
            result = _failed_result(
                step.arguments,
                f"collection error: {error or 'worker returned no collection batch'}",
                error_code="collection_worker_failed",
            )
        else:
            try:
                with measure_step("job_persistence"):
                    persisted = self.persist_collection(batch)
                result = build_collection_result(step.arguments, batch, persisted)
            except (RunCancelled, RunDeadlineExceeded, ModelRequestTimeout):
                raise
            except Exception as exc:
                logger.exception("Collection persistence failed", error=str(exc))
                result = _failed_result(
                    step.arguments,
                    f"collection persistence error: {exc}",
                    error_code=f"persistence_error:{type(exc).__name__}",
                )

        updated = investigation.model_copy(
            update={
                "executed_step_ids": [*investigation.executed_step_ids, step.step_id],
                "collection_document_ids": sorted(
                    set(investigation.collection_document_ids)
                    | set(result.document_ids)
                ),
                "status": InvestigationStatus.VALIDATING,
            }
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
                "document_ids": result.document_ids,
            },
        )
        return {
            "request": {"investigation": updated},
            "execution": {
                "active_step_id": "",
                "pending_collection": None,
                "collection_error": "",
                "collection_results": [
                    *state["execution"].get("collection_results", []),
                    result,
                ],
            },
        }


__all__ = ["InvestigationCollectionNodes", "build_collection_result"]

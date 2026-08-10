"""조사 계획의 원문 수집, 후처리와 저장 단계를 실행한다."""

from __future__ import annotations

from typing import Any, Callable

from agent.graph.investigation_context import InvestigationState
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
from shared.schema.collection_run import (
    CollectionBatch,
    CollectionExperienceResult,
    PersistenceReport,
    PostprocessedCollection,
    RecipeLearningResult,
)
from shared.schema.feedback_schema import WorkerSubmission
from shared.schema.investigation_schema import InvestigationPlanStep


def _scope_exhausted(submission: WorkerSubmission, resolved_count: int) -> bool:
    availability = submission.extracted_summary.get("job_results_availability") or {}
    available_count = availability.get("available_job_count")
    evidence = str(availability.get("count_evidence") or "").strip()
    return bool(
        isinstance(available_count, int)
        and available_count >= 0
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
    batch: CollectionBatch,
    persistence: PersistenceReport,
    experience: CollectionExperienceResult,
) -> CollectionResult:
    """작업자 관찰과 DB 저장 결과를 지휘자용 결과로 합친다."""

    submission = batch.submission
    intent = submission.collection_intent
    persisted_items = list(persistence.persisted_items)
    persisted_ids = {
        int(item["job_id"])
        for item in persisted_items
        if str(item.get("job_id") or "").isdigit()
    }
    observed_ids = {
        int(job_id) for job_id in submission.observed_job_ids if int(job_id) > 0
    }
    document_ids = sorted(persisted_ids | observed_ids)
    resolved_count = len(document_ids)
    scope_exhausted = _scope_exhausted(submission, resolved_count)
    worker_finished = submission.run_status == "finished"
    hit_recursion_limit = submission.run_status == "recursion_limit"
    status = _collection_status(
        resolved_count=resolved_count,
        target_count=intent.target_count,
        rejected_count=persistence.rejected_count,
        worker_finished=worker_finished,
        scope_exhausted=scope_exhausted,
    )
    return CollectionResult(
        status=status,
        message=(
            f"collection {status}: keyword={intent.search_keyword!r}, "
            f"site={batch.site_name}, resolved={resolved_count}, "
            f"persisted={persistence.persisted_count}"
        ),
        site=intent.site,
        site_name=batch.site_name,
        search_keyword=intent.search_keyword,
        task_category=intent.task_category,
        target_count=intent.target_count,
        collected_count=submission.collected_count,
        resolved_count=resolved_count,
        persisted_count=persistence.persisted_count,
        created_count=persistence.created_count,
        updated_count=persistence.updated_count,
        rejected_count=persistence.rejected_count,
        persisted_items=persisted_items,
        observed_job_ids=sorted(observed_ids),
        document_ids=document_ids,
        scope_exhausted=scope_exhausted,
        worker_finished=worker_finished,
        hit_recursion_limit=hit_recursion_limit,
        worker_run_id=experience.run_id,
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


def _complete_step_update(
    state: InvestigationState,
    step: InvestigationPlanStep,
    result: CollectionResult,
) -> dict[str, Any]:
    execution = state["execution"]
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
            "worker_run_id": result.worker_run_id,
            "document_ids": result.document_ids,
        },
    )
    return {
        "execution": {
            "executed_step_ids": [
                *execution.get("executed_step_ids", []),
                step.step_id,
            ],
            "collection_document_ids": sorted(
                set(execution.get("collection_document_ids", []))
                | set(result.document_ids)
            ),
            "collection_results": [
                *execution.get("collection_results", []),
                result,
            ],
            "pending_collection": None,
            "postprocessed_collection": None,
        }
    }


class InvestigationCollectionNodes:
    """원문 수집, 구조화와 저장의 책임 경계를 순서대로 실행한다."""

    def __init__(
        self,
        run_collection: Callable[[CollectionIntent], CollectionBatch],
        postprocess_collection: Callable[
            [CollectionBatch], PostprocessedCollection
        ],
        store_collection: Callable[[PostprocessedCollection], PersistenceReport],
        record_experience: Callable[
            [CollectionBatch, PersistenceReport], CollectionExperienceResult
        ],
    ) -> None:
        self.run_collection = run_collection
        self.postprocess_collection = postprocess_collection
        self.store_collection = store_collection
        self.record_experience = record_experience

    @staticmethod
    def _next_step(state: InvestigationState) -> InvestigationPlanStep:
        execution = state["execution"]
        return next(
            item
            for item in execution.get("plan", [])
            if item.step_id not in execution.get("executed_step_ids", [])
        )

    def collect(self, state: InvestigationState) -> dict[str, Any]:
        raise_if_cancelled()
        step = self._next_step(state)
        emit_run_event(
            "collection_started",
            RunPhase.COLLECTION,
            step.purpose or "계획한 채용공고 수집을 실행하고 있습니다.",
        )
        try:
            with measure_step("vision_worker", site=step.arguments.site):
                batch = self.run_collection(step.arguments)
        except (RunCancelled, RunDeadlineExceeded, ModelRequestTimeout):
            raise
        except Exception as exc:
            logger.exception("Vision worker execution failed", error=str(exc))
            return _complete_step_update(
                state,
                step,
                _failed_result(
                    step.arguments,
                    f"collection error: {type(exc).__name__}: {exc}",
                    error_code="collection_worker_failed",
                ),
            )
        return {
            "execution": {
                "pending_collection": batch,
                "postprocessed_collection": None,
            }
        }

    @staticmethod
    def route_after_collect(state: InvestigationState) -> str:
        return (
            "postprocess"
            if state["execution"].get("pending_collection") is not None
            else "inspect_evidence"
        )

    def postprocess(self, state: InvestigationState) -> dict[str, Any]:
        batch = state["execution"].get("pending_collection")
        if not isinstance(batch, CollectionBatch):
            raise TypeError("후처리할 CollectionBatch가 없습니다.")
        step = self._next_step(state)
        try:
            with measure_step("collection_postprocessing"):
                processed = self.postprocess_collection(batch)
        except (RunCancelled, RunDeadlineExceeded, ModelRequestTimeout):
            raise
        except Exception as exc:
            logger.exception("Collection postprocessing failed", error=str(exc))
            return _complete_step_update(
                state,
                step,
                _failed_result(
                    step.arguments,
                    f"collection postprocessing error: {exc}",
                    error_code=f"postprocessing_error:{type(exc).__name__}",
                ),
            )
        return {"execution": {"postprocessed_collection": processed}}

    @staticmethod
    def route_after_postprocess(state: InvestigationState) -> str:
        return (
            "persist"
            if state["execution"].get("postprocessed_collection") is not None
            else "inspect_evidence"
        )

    def persist(self, state: InvestigationState) -> dict[str, Any]:
        execution = state["execution"]
        batch = execution.get("pending_collection")
        processed = execution.get("postprocessed_collection")
        if not isinstance(batch, CollectionBatch) or not isinstance(
            processed, PostprocessedCollection
        ):
            raise TypeError("저장할 후처리 결과가 없습니다.")
        step = self._next_step(state)
        try:
            with measure_step("job_persistence"):
                persistence = self.store_collection(processed)
        except (RunCancelled, RunDeadlineExceeded, ModelRequestTimeout):
            raise
        except Exception as exc:
            logger.exception("Collection persistence failed", error=str(exc))
            return _complete_step_update(
                state,
                step,
                _failed_result(
                    step.arguments,
                    f"collection persistence error: {exc}",
                    error_code=f"persistence_error:{type(exc).__name__}",
                ),
            )

        try:
            experience = self.record_experience(batch, persistence)
        except Exception as exc:
            logger.warning("Collection experience recording failed: %s", exc)
            experience = CollectionExperienceResult(
                run_id="",
                recipe_learning=RecipeLearningResult(
                    status="failed",
                    reason="experience_recording_failed",
                    error=f"{type(exc).__name__}: {exc}"[:1000],
                ),
            )
        return _complete_step_update(
            state,
            step,
            build_collection_result(batch, persistence, experience),
        )


__all__ = ["InvestigationCollectionNodes", "build_collection_result"]

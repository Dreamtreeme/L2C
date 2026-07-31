"""비전 작업자 실행, 검토, 저장을 조율하는 채용공고 수집 서비스."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from agent.application.collection_outcome import build_collection_outcome
from agent.application.run_context import emit_run_event, measure_step
from agent.application.run_contracts import RunPhase
from agent.utils.logger import logger
from agent.utils.model_dump import dump_model
from shared.schema.collection_intent import normalize_collection_intent


@dataclass(frozen=True)
class CollectionRequest:
    search_keyword: str
    site: str | None = None
    target_count: int = 0
    task_category: str = ""
    search_intent_resolved: bool = False
    review_feedback: str = ""
    review_attempt: int = 0
    collection_intent: dict[str, Any] | None = None


@dataclass(frozen=True)
class CollectionOperations:
    normalize_target_count: Callable[[Any], int]
    normalize_task_category: Callable[[str], str]
    review_retries: Callable[[], int]
    run_worker: Callable[..., dict]
    review_worker: Callable[[dict], tuple[dict, str]]
    persist_result: Callable[[dict, dict], tuple[int, dict, dict, str]]
    render_review_feedback: Callable[[dict], str]
    needs_approval: Callable[..., bool]
    build_intermediate_report: Callable[..., dict]
    report_requires_more_collection: Callable[[dict], bool]
    close_browser: Callable[[], None]


class CollectionService:
    """작업자의 의미 판단에는 개입하지 않고 실행 수명과 저장만 조율한다."""

    def __init__(self, operations: CollectionOperations):
        self.operations = operations

    @staticmethod
    def _job_results_availability(submission: dict, worker_result: dict) -> dict[str, Any]:
        summary = submission.get("extracted_summary") if isinstance(submission, dict) else {}
        summary = summary if isinstance(summary, dict) else {}
        final_state = worker_result.get("final_state") if isinstance(worker_result, dict) else {}
        final_state = final_state if isinstance(final_state, dict) else {}
        availability = summary.get("job_results_availability") or final_state.get("job_results_availability") or {}
        return dict(availability) if isinstance(availability, dict) else {}

    @classmethod
    def _search_scope_exhausted(
        cls,
        submission: dict,
        worker_result: dict,
        resolved_count: int,
    ) -> tuple[bool, dict[str, Any]]:
        """화면에서 확인한 전체 결과를 모두 처리했는지 판단한다."""

        availability = cls._job_results_availability(submission, worker_result)
        try:
            available_count = int(availability.get("available_job_count"))
            confidence = float(availability.get("count_confidence") or 0.0)
        except (TypeError, ValueError):
            return False, {}
        evidence = str(availability.get("count_evidence") or "").strip()
        trusted = available_count >= 0 and confidence >= 0.8 and bool(evidence)
        return bool(trusted and resolved_count >= available_count), availability

    @staticmethod
    def _merge_persistence_validation(
        aggregate: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        """여러 작업자 시도의 저장 결과를 문서 기준으로 합친다."""

        aggregate_items = [
            item
            for item in (aggregate.get("persisted_items") or [])
            if isinstance(item, dict)
        ]
        current_items = [
            item
            for item in (current.get("persisted_items") or [])
            if isinstance(item, dict)
        ]
        persisted: dict[str, dict[str, Any]] = {}
        for item in [*aggregate_items, *current_items]:
            if not isinstance(item, dict):
                continue
            key = str(item.get("job_id") or item.get("url") or len(persisted))
            persisted[key] = dict(item)
        unidentified_count = max(
            0,
            int(aggregate.get("persisted_count") or 0) - len(aggregate_items),
        ) + max(
            0,
            int(current.get("persisted_count") or 0) - len(current_items),
        )
        rejected = [
            dict(item)
            for item in [
                *(aggregate.get("rejected_items") or []),
                *(current.get("rejected_items") or []),
            ]
            if isinstance(item, dict)
        ]
        return {
            "submitted_count": int(aggregate.get("submitted_count") or 0)
            + int(current.get("submitted_count") or 0),
            "persisted_count": len(persisted) + unidentified_count,
            "created_count": int(aggregate.get("created_count") or 0)
            + int(current.get("created_count") or 0),
            "updated_count": int(aggregate.get("updated_count") or 0)
            + int(current.get("updated_count") or 0),
            "persisted_items": list(persisted.values()),
            "rejected_count": len(rejected),
            "rejected_items": rejected,
        }

    def collect(self, request: CollectionRequest) -> dict[str, Any]:
        keyword = str(request.search_keyword or "").strip()
        if not keyword:
            return {
                "message": "collection failed: missing search keyword",
                "review": {"decision": "reject"},
            }

        intent = normalize_collection_intent(
            request.collection_intent,
            original_query=keyword,
            site=request.site or "",
            search_keyword=keyword,
            target_count=request.target_count,
        )
        intent_payload = dump_model(intent)
        keyword = intent.search_keyword
        target_count = self.operations.normalize_target_count(intent.target_count)
        task_category = self.operations.normalize_task_category(request.task_category)
        emit_run_event(
            "collection_started",
            RunPhase.COLLECTION,
            "비전 작업자가 채용공고 수집을 시작했습니다.",
            data={
                "site": request.site or "",
                "target_count": target_count,
                "task_category": task_category,
                "count_mode": intent.count_mode.value,
            },
        )

        try:
            max_review_retries = self.operations.review_retries()
            attempt = max(0, int(request.review_attempt or 0))
            pending_feedback = request.review_feedback or ""
            worker_run_id: str | None = None
            aggregate_validation: dict[str, Any] = {}
            observed_job_ids: set[int] = set()

            while True:
                persisted_ids = {
                    int(item["job_id"])
                    for item in aggregate_validation.get("persisted_items", [])
                    if isinstance(item, dict) and str(item.get("job_id", "")).isdigit()
                }
                resolved_count = max(
                    int(aggregate_validation.get("persisted_count") or 0),
                    len(persisted_ids | observed_job_ids),
                )
                remaining_target = (
                    max(0, target_count - resolved_count)
                    if target_count > 0
                    else 0
                )
                attempt_intent = dict(intent_payload)
                if target_count > 0:
                    attempt_intent["target_count"] = remaining_target
                with measure_step(
                    "vision_worker",
                    site=request.site or "",
                    review_attempt=attempt,
                ):
                    worker_result = self.operations.run_worker(
                        keyword,
                        site=request.site,
                        target_count=remaining_target if target_count > 0 else target_count,
                        task_category=task_category,
                        search_intent_resolved=request.search_intent_resolved,
                        collection_intent=attempt_intent,
                        review_feedback=pending_feedback,
                        review_attempt=attempt,
                        run_id=worker_run_id,
                    )
                submission = worker_result["submission"]
                worker_run_id = submission.get("run_id") or worker_run_id

                emit_run_event(
                    "review_started",
                    RunPhase.REVIEW,
                    "작업자 수집 결과를 검토하고 있습니다.",
                    data={"worker_run_id": worker_run_id or "", "attempt": attempt},
                )
                with measure_step("worker_review", review_attempt=attempt):
                    review, submission_id = self.operations.review_worker(submission)

                emit_run_event(
                    "persistence_started",
                    RunPhase.PERSISTENCE,
                    "검증된 채용공고를 저장하고 있습니다.",
                    data={"decision": review.get("decision", "")},
                )
                with measure_step("job_persistence"):
                    (
                        _persisted_count,
                        submission,
                        review,
                        persisted_submission_id,
                    ) = self.operations.persist_result(worker_result, review)
                if persisted_submission_id:
                    submission_id = persisted_submission_id
                current_validation = dict(worker_result.get("persistence_validation") or {})
                aggregate_validation = self._merge_persistence_validation(
                    aggregate_validation,
                    current_validation,
                )
                observed_job_ids.update(
                    int(job_id)
                    for job_id in (worker_result.get("observed_job_ids") or [])
                    if str(job_id).isdigit() and int(job_id) > 0
                )
                persisted_ids = {
                    int(item["job_id"])
                    for item in aggregate_validation.get("persisted_items", [])
                    if isinstance(item, dict) and str(item.get("job_id", "")).isdigit()
                }
                resolved_count = max(
                    int(aggregate_validation.get("persisted_count") or 0),
                    len(persisted_ids | observed_job_ids),
                )
                scope_exhausted, _availability = self._search_scope_exhausted(
                    submission,
                    worker_result,
                    resolved_count,
                )

                should_retry = bool(
                    review.get("decision") == "revise"
                    and review.get("continue_collection", True)
                    and attempt < max_review_retries
                    and not scope_exhausted
                )
                if should_retry:
                    pending_feedback = self.operations.render_review_feedback(review)
                    attempt += 1
                    logger.info(
                        "Retrying worker after commander feedback",
                        attempt=attempt,
                        worker_run_id=worker_run_id,
                    )
                    continue

                worker_result["persistence_validation"] = aggregate_validation
                worker_result["observed_job_ids"] = sorted(observed_job_ids)

                return self._build_result(
                    request=request,
                    keyword=keyword,
                    target_count=target_count,
                    task_category=task_category,
                    worker_result=worker_result,
                    submission=submission,
                    submission_id=submission_id,
                    review=review,
                    persisted_count=int(aggregate_validation.get("persisted_count") or 0),
                    resolved_count=resolved_count,
                    collection_intent=(
                        worker_result.get("collection_intent")
                        or intent_payload
                    ),
                )
        except Exception as exc:
            from agent.application.run_context import (
                ModelRequestTimeout,
                RunCancelled,
                RunDeadlineExceeded,
            )

            if isinstance(
                exc,
                (
                    RunCancelled,
                    RunDeadlineExceeded,
                    ModelRequestTimeout,
                ),
            ):
                raise
            logger.exception("Vision worker execution failed", error=str(exc))
            return {
                "message": f"collection error: {exc}",
                "review": {"decision": "reject"},
            }
        finally:
            with measure_step("browser_cleanup"):
                self.operations.close_browser()

    def _build_result(
        self,
        *,
        request: CollectionRequest,
        keyword: str,
        target_count: int,
        task_category: str,
        worker_result: dict,
        submission: dict,
        submission_id: str,
        review: dict,
        persisted_count: int,
        resolved_count: int,
        collection_intent: dict[str, Any],
    ) -> dict[str, Any]:
        item_count = int(submission.get("collected_count") or 0)
        site_name = worker_result.get("site_name", request.site or "unknown")
        site_slug = worker_result.get("site_slug", request.site or "unknown")
        effective_keyword = worker_result.get("keyword") or keyword
        hit_recursion_limit = bool(worker_result.get("hit_recursion_limit", False))
        is_finished = bool(worker_result.get("is_finished", False))
        recursion_limit = int(worker_result.get("recursion_limit") or 0)
        effective_target_count = int(
            target_count
            or worker_result.get("target_count")
            or submission.get("target_count")
            or 0
        )
        effective_task_category = (
            worker_result.get("task_category")
            or submission.get("task_category")
            or task_category
        )
        base_needs_approval = self.operations.needs_approval(
            hit_recursion_limit=hit_recursion_limit,
            is_finished=is_finished,
            persisted_count=resolved_count,
            target_count=effective_target_count,
        )
        intermediate_report = (
            self.operations.build_intermediate_report(
                worker_result,
                submission,
                persisted_count=persisted_count,
                current_limit=recursion_limit,
                target_count=effective_target_count,
            )
            if base_needs_approval
            else {}
        )
        scope_exhausted, availability = self._search_scope_exhausted(
            submission,
            worker_result,
            resolved_count,
        )
        needs_approval = (
            base_needs_approval
            and not scope_exhausted
            and self.operations.report_requires_more_collection(intermediate_report)
        )
        validation = dict(worker_result.get("persistence_validation") or {})
        rejected_count = int(validation.get("rejected_count") or 0)
        outcome = build_collection_outcome(
            is_finished=is_finished,
            hit_recursion_limit=hit_recursion_limit,
            review=review,
            persisted_count=persisted_count,
            resolved_count=resolved_count,
            rejected_count=rejected_count,
            target_count=effective_target_count,
            scope_exhausted=scope_exhausted,
        )
        outcome_fields = outcome.as_dict()
        completion_status = outcome_fields["completion_status"]
        missing_count = (
            max(0, effective_target_count - resolved_count)
            if effective_target_count > 0
            else 0
        )
        collection_document_ids = sorted(
            {
                *(
                    int(item["job_id"])
                    for item in validation.get("persisted_items", [])
                    if isinstance(item, dict)
                    and str(item.get("job_id", "")).isdigit()
                    and int(item["job_id"]) > 0
                ),
                *(
                    int(job_id)
                    for job_id in (worker_result.get("observed_job_ids") or [])
                    if str(job_id).isdigit() and int(job_id) > 0
                ),
            }
        )

        if needs_approval:
            message = (
                "intermediate report: recursion limit reached with partial collection persisted; "
                f"keyword={effective_keyword!r}, site={site_name}, collected={item_count}, "
                f"persisted={persisted_count}; approval required to raise limit to "
                f"{intermediate_report.get('suggested_recursion_limit')}"
            )
        elif scope_exhausted:
            message = (
                f"search scope exhausted: keyword={effective_keyword!r}, site={site_name}, "
                f"available={availability.get('available_job_count', 0)}, "
                f"resolved={resolved_count}, persisted={persisted_count}"
            )
        elif persisted_count > 0:
            completion_type = "partial collection persisted" if completion_status == "partial" else "vision collection persisted"
            message = (
                f"{completion_type}: keyword={effective_keyword!r}, site={site_name}, "
                f"collected={item_count}, persisted={persisted_count}"
            )
        elif resolved_count > 0:
            message = (
                f"existing database jobs confirmed: keyword={effective_keyword!r}, "
                f"site={site_name}, resolved={resolved_count}"
            )
        elif review.get("decision") == "revise":
            feedback_text = review.get("feedback_to_worker") or "; ".join(
                review.get("reasons") or []
            )
            message = f"worker submission needs revision: {feedback_text}"
        elif hit_recursion_limit:
            message = (
                "collection stopped at recursion limit without accepted data: "
                f"site={site_name}, keyword={effective_keyword!r}"
            )
        else:
            message = (
                f"collection finished without accepted data: site={site_name}, "
                f"keyword={effective_keyword!r}"
            )

        emit_run_event(
            "collection_completed",
            RunPhase.COLLECTION,
            "채용공고 수집과 저장을 마쳤습니다.",
            data={
                "site": site_slug,
                "item_count": item_count,
                "persisted_count": persisted_count,
                "document_ids": collection_document_ids,
                "needs_human_approval": needs_approval,
                "completion_status": completion_status,
                "stage_statuses": outcome_fields,
            },
        )
        return {
            "message": message,
            "site": site_slug,
            "site_name": site_name,
            "keyword": effective_keyword,
            "target_count": effective_target_count,
            "task_category": self.operations.normalize_task_category(
                effective_task_category
            ),
            "item_count": item_count,
            "persisted_count": persisted_count,
            "submission_id": submission_id,
            "worker_run_id": submission.get("run_id") or "",
            "review": review,
            "hit_recursion_limit": hit_recursion_limit,
            "is_finished": is_finished,
            "needs_human_approval": needs_approval,
            "intermediate_report": intermediate_report,
            "collection_intent": collection_intent,
            "completion_status": completion_status,
            "worker_status": outcome_fields["worker_status"],
            "review_status": outcome_fields["review_status"],
            "persistence_status": outcome_fields["persistence_status"],
            "target_status": outcome_fields["target_status"],
            "stage_statuses": outcome_fields,
            "search_scope_exhausted": scope_exhausted,
            "job_results_availability": availability,
            "missing_count": missing_count,
            "document_ids": collection_document_ids,
            "persistence_validation": validation,
            "observed_job_ids": sorted(
                {
                    int(job_id)
                    for job_id in (worker_result.get("observed_job_ids") or [])
                    if str(job_id).isdigit() and int(job_id) > 0
                }
            ),
        }


def build_collection_operations(
    worker_runtime: Any,
) -> CollectionOperations:
    """애플리케이션 수집 서비스가 사용할 외부 작업을 조립한다."""

    from agent.application.collection_request_builder import (
        normalize_target_count,
    )
    from agent.application.collection_submission_service import (
        commit_worker_review,
        persist_accepted_worker_result,
    )
    from agent.application.collection_worker_runner import (
        build_limit_intermediate_report,
        limit_report_requires_more_collection,
        needs_human_limit_approval,
        run_worker_once,
        worker_review_retries,
    )
    from agent.application.worker_execution_service import (
        close_browser_after_run,
    )
    from agent.recipe.reviewer import render_review_feedback
    from agent.recipe.task_category import normalize_task_category

    return CollectionOperations(
        normalize_target_count=normalize_target_count,
        normalize_task_category=normalize_task_category,
        review_retries=worker_review_retries,
        run_worker=partial(
            run_worker_once,
            worker_runtime=worker_runtime,
        ),
        review_worker=commit_worker_review,
        persist_result=persist_accepted_worker_result,
        render_review_feedback=render_review_feedback,
        needs_approval=needs_human_limit_approval,
        build_intermediate_report=build_limit_intermediate_report,
        report_requires_more_collection=limit_report_requires_more_collection,
        close_browser=partial(
            close_browser_after_run,
            worker_runtime=worker_runtime,
        ),
    )


__all__ = [
    "CollectionOperations",
    "CollectionRequest",
    "CollectionService",
    "build_collection_operations",
]

"""비전 작업자 실행, 검토, 저장을 조율하는 채용공고 수집 서비스."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.application.run_context import emit_run_event, measure_step
from agent.application.run_contracts import RunPhase
from agent.utils.logger import logger


@dataclass(frozen=True)
class CollectionRequest:
    search_keyword: str
    site: str | None = None
    target_count: int = 0
    task_category: str = ""
    review_feedback: str = ""
    review_attempt: int = 0


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

    def collect(self, request: CollectionRequest) -> dict[str, Any]:
        keyword = str(request.search_keyword or "").strip()
        if not keyword:
            return {
                "message": "collection failed: missing search keyword",
                "review": {"decision": "reject"},
            }

        target_count = self.operations.normalize_target_count(request.target_count)
        task_category = self.operations.normalize_task_category(request.task_category)
        emit_run_event(
            "collection_started",
            RunPhase.COLLECTION,
            "비전 작업자가 채용공고 수집을 시작했습니다.",
            data={
                "site": request.site or "",
                "target_count": target_count,
                "task_category": task_category,
            },
        )

        try:
            max_review_retries = self.operations.review_retries()
            attempt = max(0, int(request.review_attempt or 0))
            pending_feedback = request.review_feedback or ""
            worker_run_id: str | None = None

            while True:
                with measure_step(
                    "vision_worker",
                    site=request.site or "",
                    review_attempt=attempt,
                ):
                    worker_result = self.operations.run_worker(
                        keyword,
                        site=request.site,
                        target_count=target_count,
                        task_category=task_category,
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

                if review.get("decision") == "revise" and attempt < max_review_retries:
                    pending_feedback = self.operations.render_review_feedback(review)
                    attempt += 1
                    logger.info(
                        "Retrying worker after commander feedback",
                        attempt=attempt,
                        worker_run_id=worker_run_id,
                    )
                    continue

                emit_run_event(
                    "persistence_started",
                    RunPhase.PERSISTENCE,
                    "승인된 채용공고를 저장하고 있습니다.",
                    data={"decision": review.get("decision", "")},
                )
                with measure_step("job_persistence"):
                    (
                        persisted_count,
                        submission,
                        review,
                        persisted_submission_id,
                    ) = self.operations.persist_result(worker_result, review)
                if persisted_submission_id:
                    submission_id = persisted_submission_id

                return self._build_result(
                    request=request,
                    keyword=keyword,
                    target_count=target_count,
                    task_category=task_category,
                    worker_result=worker_result,
                    submission=submission,
                    submission_id=submission_id,
                    review=review,
                    persisted_count=persisted_count,
                )
        except Exception as exc:
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
    ) -> dict[str, Any]:
        item_count = int(submission.get("collected_count") or 0)
        site_name = worker_result.get("site_name", request.site or "unknown")
        site_slug = worker_result.get("site_slug", request.site or "unknown")
        effective_keyword = worker_result.get("keyword") or keyword
        hit_recursion_limit = bool(worker_result.get("hit_recursion_limit", False))
        is_finished = bool(worker_result.get("is_finished", False))
        recursion_limit = int(worker_result.get("recursion_limit") or 0)
        effective_target_count = int(
            worker_result.get("target_count")
            or submission.get("target_count")
            or target_count
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
            persisted_count=persisted_count,
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
        needs_approval = base_needs_approval and self.operations.report_requires_more_collection(
            intermediate_report
        )

        if needs_approval:
            message = (
                "intermediate report: recursion limit reached with partial collection persisted; "
                f"keyword={effective_keyword!r}, site={site_name}, collected={item_count}, "
                f"persisted={persisted_count}; approval required to raise limit to "
                f"{intermediate_report.get('suggested_recursion_limit')}"
            )
        elif review.get("decision") == "accept" and persisted_count > 0:
            completion_type = (
                "partial collection persisted"
                if hit_recursion_limit and not is_finished
                else "vision collection persisted"
            )
            message = (
                f"{completion_type}: keyword={effective_keyword!r}, site={site_name}, "
                f"collected={item_count}, persisted={persisted_count}"
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
                "needs_human_approval": needs_approval,
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
        }


__all__ = ["CollectionOperations", "CollectionRequest", "CollectionService"]

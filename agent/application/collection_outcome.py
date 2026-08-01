"""수집 실행의 서로 다른 성공 경계를 명시적으로 표현한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from agent.application.run_contracts import RunStatus


class WorkerStatus(str, Enum):
    FINISHED = "finished"
    LIMIT_REACHED = "limit_reached"
    STOPPED = "stopped"


class ReviewStatus(str, Enum):
    ACCEPTED = "accepted"
    REVISION_REQUIRED = "revision_required"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class PersistenceStatus(str, Enum):
    PERSISTED = "persisted"
    EXISTING_ONLY = "existing_only"
    PARTIAL = "partial"
    REJECTED = "rejected"
    EMPTY = "empty"


class TargetStatus(str, Enum):
    MET = "met"
    SCOPE_EXHAUSTED = "scope_exhausted"
    UNMET = "unmet"
    UNSPECIFIED = "unspecified"


class CollectionStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    REJECTED = "rejected"


@dataclass(frozen=True)
class CollectionOutcome:
    worker_status: WorkerStatus
    review_status: ReviewStatus
    persistence_status: PersistenceStatus
    target_status: TargetStatus
    completion_status: CollectionStatus

    def as_dict(self) -> dict[str, str]:
        return {
            key: value.value if isinstance(value, Enum) else str(value)
            for key, value in asdict(self).items()
        }


def collection_run_status(
    completion_status: CollectionStatus | str,
    *,
    needs_human_approval: bool = False,
) -> RunStatus:
    """수집 도메인 결과를 백엔드 실행 상태로 변환한다."""

    if needs_human_approval:
        return RunStatus.WAITING_APPROVAL
    try:
        status = CollectionStatus(completion_status)
    except (TypeError, ValueError):
        return RunStatus.FAILED
    return {
        CollectionStatus.COMPLETE: RunStatus.COMPLETED,
        CollectionStatus.PARTIAL: RunStatus.PARTIAL,
        CollectionStatus.REJECTED: RunStatus.FAILED,
    }[status]


def _review_status(review: dict[str, Any]) -> ReviewStatus:
    decision = str(review.get("decision") or "").strip().casefold()
    return {
        "accept": ReviewStatus.ACCEPTED,
        "revise": ReviewStatus.REVISION_REQUIRED,
        "reject": ReviewStatus.REJECTED,
    }.get(decision, ReviewStatus.UNKNOWN)


def build_collection_outcome(
    *,
    is_finished: bool,
    hit_recursion_limit: bool,
    review: dict[str, Any],
    persisted_count: int,
    resolved_count: int,
    rejected_count: int,
    target_count: int,
    scope_exhausted: bool,
) -> CollectionOutcome:
    """작업자, 검토, 저장과 목표 충족 상태를 독립적으로 계산한다."""

    if is_finished:
        worker_status = WorkerStatus.FINISHED
    elif hit_recursion_limit:
        worker_status = WorkerStatus.LIMIT_REACHED
    else:
        worker_status = WorkerStatus.STOPPED

    review_status = _review_status(review)
    if persisted_count > 0 and rejected_count > 0:
        persistence_status = PersistenceStatus.PARTIAL
    elif persisted_count > 0:
        persistence_status = PersistenceStatus.PERSISTED
    elif resolved_count > 0:
        persistence_status = PersistenceStatus.EXISTING_ONLY
    elif rejected_count > 0:
        persistence_status = PersistenceStatus.REJECTED
    else:
        persistence_status = PersistenceStatus.EMPTY

    if scope_exhausted:
        target_status = TargetStatus.SCOPE_EXHAUSTED
    elif target_count <= 0:
        target_status = TargetStatus.UNSPECIFIED
    elif resolved_count >= target_count:
        target_status = TargetStatus.MET
    else:
        target_status = TargetStatus.UNMET

    review_allows_result = (
        review_status == ReviewStatus.ACCEPTED
        or bool(review.get("accept_collected_data"))
    )
    if not review_allows_result:
        completion_status = CollectionStatus.REJECTED
    elif resolved_count <= 0:
        completion_status = CollectionStatus.REJECTED
    elif target_status == TargetStatus.SCOPE_EXHAUSTED:
        completion_status = CollectionStatus.COMPLETE
    elif (
        target_status == TargetStatus.MET
        and worker_status == WorkerStatus.FINISHED
    ):
        completion_status = (
            CollectionStatus.PARTIAL
            if persistence_status == PersistenceStatus.PARTIAL
            else CollectionStatus.COMPLETE
        )
    else:
        completion_status = CollectionStatus.PARTIAL

    return CollectionOutcome(
        worker_status=worker_status,
        review_status=review_status,
        persistence_status=persistence_status,
        target_status=target_status,
        completion_status=completion_status,
    )


__all__ = [
    "CollectionOutcome",
    "CollectionStatus",
    "PersistenceStatus",
    "ReviewStatus",
    "TargetStatus",
    "WorkerStatus",
    "build_collection_outcome",
    "collection_run_status",
]

"""비전 작업자 노드 테스트의 상태와 런타임 생성 도우미."""

from __future__ import annotations

from typing import Any, cast

from langgraph.runtime import Runtime

from agent.runtime.vision_worker_runtime import (
    VisionWorkerRuntime,
    WorkerDependencies,
)
from agent.runtime.worker_contracts import (
    WorkerState,
    WorkerStateUpdate,
    apply_worker_state_update,
    create_worker_state,
)
from agent.runtime.worker_data_services import WorkerDataServices
from shared.schema.jd_schema import JobReview, JobReviewStatus


def worker_state(
    *,
    goal: str = "",
    request: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    transition: dict[str, Any] | None = None,
    replay: dict[str, Any] | None = None,
    collection: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    lifecycle: dict[str, Any] | None = None,
) -> WorkerState:
    return create_worker_state(
        goal,
        request=request,
        observation=observation,
        decision=decision,
        transition=transition,
        replay=replay,
        collection=collection,
        progress=progress,
        lifecycle=lifecycle,
    )


def apply_update(
    state: WorkerState,
    update: WorkerStateUpdate,
) -> WorkerState:
    return apply_worker_state_update(state, update)


def worker_data_services(
    *,
    mark_existing_job_cards=None,
    find_existing_job_url=None,
    load_experience_rules=None,
    record_recipe_replay=None,
    review_job_draft=None,
) -> WorkerDataServices:
    """노드 단위 테스트에서 외부 DB와 모델 호출을 제거한다."""

    return WorkerDataServices(
        mark_existing_job_cards=(
            mark_existing_job_cards or (lambda queue, _url: (queue, []))
        ),
        find_existing_job_url=(
            find_existing_job_url
            or (lambda _url, _jobs: {"matched": False})
        ),
        load_experience_rules=(
            load_experience_rules
            or (lambda _site, *, task_category=None: [])
        ),
        record_recipe_replay=(
            record_recipe_replay or (lambda _recipe_key, _succeeded: True)
        ),
        review_job_draft=(
            review_job_draft
            or (
                lambda draft, _intent: JobReview(
                    detail_key=draft.detail_key,
                    url=draft.url,
                    status=JobReviewStatus.NEEDS_MORE,
                    missing_fields=draft.required_fields,
                    reason="테스트 기본 검토 결과",
                )
            )
        ),
    )


def node_runtime(
    vision: Any = None,
    data: WorkerDataServices | None = None,
) -> Runtime[WorkerDependencies]:
    return Runtime(
        context=WorkerDependencies(
            vision=cast(VisionWorkerRuntime, vision or object()),
            data=data or worker_data_services(),
        )
    )


__all__ = [
    "apply_update",
    "node_runtime",
    "worker_data_services",
    "worker_state",
]

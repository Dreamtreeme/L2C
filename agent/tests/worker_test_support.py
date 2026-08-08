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


def worker_state(
    *,
    goal: str = "",
    request: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    transition: dict[str, Any] | None = None,
    replay: dict[str, Any] | None = None,
    collection: dict[str, Any] | None = None,
    lifecycle: dict[str, Any] | None = None,
    safety: dict[str, Any] | None = None,
) -> WorkerState:
    return create_worker_state(
        goal,
        request=request,
        observation=observation,
        decision=decision,
        transition=transition,
        replay=replay,
        collection=collection,
        lifecycle=lifecycle,
        safety=safety,
    )


def apply_update(
    state: WorkerState,
    update: WorkerStateUpdate,
) -> WorkerState:
    return apply_worker_state_update(state, update)


def worker_data_services(
    *,
    extract_job_detail=None,
    mark_existing_job_cards=None,
    find_existing_job_url=None,
) -> WorkerDataServices:
    """노드 단위 테스트에서 외부 DB와 모델 호출을 제거한다."""

    return WorkerDataServices(
        extract_job_detail=(
            extract_job_detail or (lambda _state, _url: {})
        ),
        mark_existing_job_cards=(
            mark_existing_job_cards or (lambda queue, _url: (queue, []))
        ),
        find_existing_job_url=(
            find_existing_job_url
            or (lambda _url, _jobs: {"matched": False})
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

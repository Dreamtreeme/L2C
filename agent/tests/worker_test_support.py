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


def node_runtime(vision: Any = None) -> Runtime[WorkerDependencies]:
    return Runtime(
        context=WorkerDependencies(
            vision=cast(VisionWorkerRuntime, vision or object())
        )
    )


__all__ = ["apply_update", "node_runtime", "worker_state"]

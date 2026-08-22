"""네 책임 노드 안에서 작업자 관찰과 행동 결정을 순서대로 조율한다."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from langgraph.runtime import Runtime

from agent.config import get_settings
from agent.graph.worker_observation import capture_node, ocr_node
from agent.graph.worker_reasoning import reasoning_node
from agent.graph.worker_selection import selection_node
from agent.graph.worker_transition import (
    complete_action_effect,
    inspect_action_effect,
)
from agent.observability.graph_events import graph_step
from agent.observability.reflex_paths import (
    reflex_selection_observation,
    reflex_step_observation,
)
from agent.recipe.replay_runtime import attempt_reflex_replay
from agent.runtime.job_card_queue import has_unresolved_job_card_queue
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.runtime.worker_contracts import (
    WorkerState,
    WorkerStateUpdate,
    action_event_results,
    apply_worker_state_update,
)
from agent.runtime.worker_state import current_observation_ready
from shared.schema.jd_schema import JobReviewStatus


WorkerStep = Callable[
    [WorkerState, Runtime[WorkerDependencies]],
    Mapping[str, Any],
]


def _action_source(update: Mapping[str, Any]) -> str:
    decision = update.get("decision")
    if not isinstance(decision, Mapping):
        return ""
    request = decision.get("pending_action")
    return str(request.source if request is not None else "")


def _apply_step(
    name: str,
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
    step: WorkerStep,
) -> WorkerState:
    """내부 단계 하나를 측정하고 LangGraph와 같은 방식으로 상태를 병합한다."""

    with graph_step(name) as observation:
        raw_update = dict(step(state, runtime))
        source = _action_source(raw_update)
        if source:
            observation["action_source"] = source
        if name == "reflex":
            observation.update(reflex_selection_observation(raw_update))
        elif name == "transition":
            observation.update(reflex_step_observation(raw_update))
        elif name == "reasoning":
            observation["reasoning_mode"] = "general"
        return apply_worker_state_update(
            state,
            cast(WorkerStateUpdate, raw_update),
        )


def _has_pending_action(state: WorkerState) -> bool:
    return state["decision"].get("pending_action") is not None


def _reflex_missed_current_observation(state: WorkerState) -> bool:
    trace = dict(state["replay"].get("reflex_trace") or {})
    observation_id = str(state["observation"].get("observation_id") or "")
    return bool(
        observation_id
        and trace.get("hit") is False
        and str(trace.get("observation_id") or "") == observation_id
    )


def _review_follow_up_due(state: WorkerState) -> bool:
    review = state["collection"].get("last_job_review")
    if review is None or review.status != JobReviewStatus.NEEDS_MORE:
        return False
    results = action_event_results(
        state["transition"].get("action_events", []) or []
    )
    return bool(results and results[-1].get("action") == "review_job_detail")


def _run_selection(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> WorkerState:
    return _apply_step("selection", state, runtime, selection_node)


def _run_reasoning(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> WorkerState:
    return _apply_step("reasoning", state, runtime, reasoning_node)


def _try_reflex(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> tuple[WorkerState, bool]:
    if not get_settings().reflex.enabled or _reflex_missed_current_observation(state):
        return state, False
    updated = _apply_step("reflex", state, runtime, attempt_reflex_replay)
    request = updated["decision"].get("pending_action")
    return updated, bool(request is not None and request.source == "reflex")


def _run_ocr(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> WorkerState:
    updated = _apply_step("ocr", state, runtime, ocr_node)
    if updated["transition"].get("transition_request") is None:
        return updated
    if (
        updated["transition"].get("transition_result", {}).get("status")
        != "needs_ocr"
    ):
        return updated
    return _apply_step(
        "transition",
        updated,
        runtime,
        complete_action_effect,
    )


def _decision_is_resolved(state: WorkerState) -> bool:
    """선택 결과가 종료·행동·대기 중 하나로 확정됐는지 반환한다."""

    transition = state["transition"].get("transition_result", {}) or {}
    return bool(
        state["lifecycle"].get("is_finished")
        or _has_pending_action(state)
        or state["observation"].get("low_information_screen")
        or transition.get("status") == "pending"
    )


def _prepare_current_observation(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> WorkerState:
    """경험 규칙을 먼저 시도하고 실패한 현재 캡처만 OCR한다."""

    if current_observation_ready(state):
        return state
    transition = state["transition"].get("transition_result", {}) or {}
    if transition.get("status") not in {"needs_ocr", "unknown"}:
        replayed, hit = _try_reflex(state, runtime)
        if hit:
            return replayed
        state = replayed
    return _run_selection(_run_ocr(state, runtime), runtime)


def observation_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> WorkerState:
    """화면 안정화, 캡처와 직전 행동의 저비용 효과 판정을 수행한다."""

    captured = _apply_step("capture", state, runtime, capture_node)
    return _apply_step(
        "transition",
        captured,
        runtime,
        inspect_action_effect,
    )


def decision_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> WorkerState:
    """확정 정책, 경험 규칙, OCR, LLM 순서로 다음 행동 하나를 정한다."""

    if state["lifecycle"].get("is_finished") or _has_pending_action(state):
        return state
    if _review_follow_up_due(state):
        return _run_reasoning(state, runtime)

    working = _run_selection(state, runtime)
    if _decision_is_resolved(working):
        return working

    working = _prepare_current_observation(working, runtime)
    if _decision_is_resolved(working):
        return working

    transition = working["transition"].get("transition_result", {}) or {}
    if has_unresolved_job_card_queue(working) or transition.get("status") == "unknown":
        return _run_reasoning(working, runtime)

    working, hit = _try_reflex(working, runtime)
    if hit:
        return working
    return _run_reasoning(working, runtime)


__all__ = ["decision_node", "observation_node"]

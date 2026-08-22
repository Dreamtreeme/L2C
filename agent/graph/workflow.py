"""비전 작업자의 관찰-결정-실행-검토 그래프."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agent.graph.worker_cycle import decision_node, observation_node
from agent.graph.worker_execution import execution_node
from agent.graph.worker_review import review_node
from agent.observability.graph_events import graph_step
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.runtime.worker_contracts import WorkerState
from agent.runtime.worker_state import current_observation_ready
from agent.utils.logger import logger


def route_after_start(state: WorkerState) -> str:
    """완료된 현재 관찰이 있으면 결정하고, 없으면 새 화면을 관찰한다."""

    if state["observation"].get("low_information_screen"):
        return "decision"
    return "decision" if current_observation_ready(state) else "observation"


def route_after_decision(state: WorkerState) -> str:
    """결정 결과를 실행하거나 화면을 다시 관찰한다."""

    if state["lifecycle"].get("is_finished", False):
        return "end"
    if state["decision"].get("pending_action") is not None:
        return "execution"
    if state["observation"].get("low_information_screen"):
        return "observation"
    raise RuntimeError("행동 결정 노드가 실행할 행동이나 종료 결과를 만들지 못했습니다.")


def route_after_execution(state: WorkerState) -> str:
    """실행 결과에 따라 새 화면 관찰, 공고 검토 또는 다음 결정을 선택한다."""

    if state["lifecycle"].get("is_finished", False):
        logger.info("Task marked as finished. Ending workflow.")
        return "end"
    if state["transition"].get("error_count", 0) >= 3:
        logger.error("Too many errors. Forcing workflow to end.")
        return "end"
    if state["collection"].get("pending_job_draft") is not None:
        return "review"
    if state["transition"].get("transition_request"):
        return "observation"
    return "decision"


def route_after_review(state: WorkerState) -> str:
    """검토로 목표가 끝났으면 종료하고, 아니면 다음 행동을 결정한다."""

    return "end" if state["lifecycle"].get("is_finished", False) else "decision"


def _instrument_node(
    name: str,
    node: Callable[
        [WorkerState, Runtime[WorkerDependencies]],
        Mapping[str, Any],
    ],
):
    """물리 실행과 공고 검토 경계만 그래프 단계로 측정한다."""

    @wraps(node)
    def observed(
        state: WorkerState,
        runtime: Runtime[WorkerDependencies],
    ) -> Mapping[str, Any]:
        with graph_step(name) as observation:
            request = state["decision"].get("pending_action")
            if request is not None:
                observation["action_source"] = str(request.source)
                observation["action_names"] = [
                    str(call.name) for call in request.tool_calls if call.name
                ]
            return node(state, runtime)

    return observed


def build_graph():
    """네 작업자 책임을 연결하고 컴파일한다."""

    logger.info("Building four-stage worker StateGraph")
    workflow = StateGraph(
        WorkerState,
        context_schema=WorkerDependencies,
    )
    workflow.add_node("observation", observation_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("execution", _instrument_node("execution", execution_node))
    workflow.add_node("review", _instrument_node("review", review_node))

    workflow.add_conditional_edges(
        START,
        route_after_start,
        {"observation": "observation", "decision": "decision"},
    )
    workflow.add_edge("observation", "decision")
    workflow.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "execution": "execution",
            "observation": "observation",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "execution",
        route_after_execution,
        {
            "observation": "observation",
            "review": "review",
            "decision": "decision",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {"decision": "decision", "end": END},
    )
    app = workflow.compile()
    logger.info("Four-stage worker StateGraph compiled")
    return app


__all__ = [
    "build_graph",
    "route_after_decision",
    "route_after_execution",
    "route_after_review",
    "route_after_start",
]

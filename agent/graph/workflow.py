"""비전 작업자의 원자 관찰-판정-선택-실행 그래프."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agent.config import get_settings
from agent.graph.worker_reasoning import reasoning_node
from agent.runtime.worker_contracts import WorkerState
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.graph.worker_observation import (
    capture_node,
    ocr_node,
)
from agent.graph.worker_selection import selection_node
from agent.graph.worker_transition import transition_node
from agent.graph.worker_execution import execution_node
from agent.runtime.worker_state import (
    current_observation_ready,
    return_to_job_results_for_url,
)
from agent.observability.graph_events import graph_step
from agent.observability.reflex_paths import (
    reflex_selection_observation,
    reflex_transition_observation,
)
from agent.recipe.replay_runtime import attempt_reflex_replay
from agent.utils.logger import logger


def route_after_start(state: WorkerState) -> str:
    """현재 캡처에 속한 관찰만 재사용하고 나머지는 다시 캡처한다."""

    if state["observation"].get("low_information_screen"):
        return "selection"
    if current_observation_ready(state):
        return "selection"
    return "capture"


def route_after_selection(state: WorkerState) -> str:
    """결정론적 정책, 선택적 OCR, Reflex, LLM 순서로 다음 경로를 고른다."""

    if state["decision"].get("pending_action") is not None:
        return "execution"
    if state["observation"].get("low_information_screen"):
        return "capture"

    transition_result = dict(state["transition"].get("transition_result", {}) or {})
    if transition_result.get("status") == "pending":
        return "capture"
    if not current_observation_ready(state):
        return "ocr" if transition_result.get("needs_ocr") else "reasoning"
    if return_to_job_results_for_url(state):
        return "reasoning"
    if state["collection"].get("job_detail_followup"):
        return "reasoning"
    if transition_result.get("status") == "unknown" and (
        str(transition_result.get("source") or "").startswith("reflex")
        or transition_result.get("source")
        in {
            "page_policy",
            "duplicate_job_policy",
        }
        or transition_result.get("reason")
        in {"no_screen_change", "reflex_no_screen_change"}
    ):
        return "reasoning"
    if not get_settings().reflex.enabled:
        return "reasoning"
    return "reflex"


def route_after_reflex(state: WorkerState) -> str:
    """Reflex가 요청을 만들었을 때만 실행하고 나머지는 LLM으로 보낸다."""

    request = state["decision"].get("pending_action")
    return (
        "execution"
        if request is not None and request.source == "reflex"
        else "reasoning"
    )


def route_after_reasoning(state: WorkerState) -> str:
    """로딩 화면은 행동 없이 재관찰하고, 그 외 판단 결과만 실행한다."""

    if (
        state["decision"].get("pending_action") is None
        and (state["decision"].get("job_card_selection_trace") or {}).get("reason")
        == "screen_loading"
    ):
        return "capture"
    return "execution"


def route_after_execution(state: WorkerState) -> str:
    """원자 실행 뒤 종료, 후속 정책, 새 관찰 또는 새 판단을 선택한다."""

    if state["lifecycle"].get("is_finished", False):
        logger.info("Task marked as finished. Ending workflow.")
        return "end"
    if state["safety"].get("pending_human_approval", False):
        logger.info("Human approval required. Ending worker loop.")
        return "end"
    if state["transition"].get("error_count", 0) >= 3:
        logger.error("Too many errors. Forcing workflow to end.")
        return "end"
    if state["decision"].get("pending_action") is not None:
        return "execution"
    if state["transition"].get("transition_request"):
        return "capture"
    return "reasoning"


def _instrument_node(
    name: str,
    node: Callable[
        [WorkerState, Runtime[WorkerDependencies]],
        dict[str, Any],
    ],
):
    @wraps(node)
    def observed(
        state: WorkerState,
        runtime: Runtime[WorkerDependencies],
    ) -> dict[str, Any]:
        with graph_step(name) as observation:
            result = node(state, runtime)
            request = (
                state["decision"].get("pending_action")
                if name == "execution"
                else (
                    (result.get("decision") or {}).get("pending_action")
                    if isinstance(result, dict)
                    else None
                )
            )
            action_source = str(request.source if request is not None else "")
            observation.update(
                action_source=action_source,
                success=True,
            )
            if name == "execution" and request is not None:
                observation["action_names"] = [
                    str(call.name) for call in request.tool_calls if call.name
                ]
            if name == "ocr" and isinstance(result, dict):
                observation["analysis_mode"] = str(
                    (result.get("observation") or {}).get("analysis_mode") or "full"
                )
            if name == "reflex" and isinstance(result, dict):
                observation.update(reflex_selection_observation(result))
            if name == "transition" and isinstance(result, dict):
                observation.update(reflex_transition_observation(result))
            if name == "reasoning":
                observation["reasoning_mode"] = (
                    "card_selection" if action_source == "card_selector" else "general"
                )
            return result

    return observed


def build_graph():
    """책임별 작업자 노드를 연결하고 컴파일한다."""

    logger.info("Building atomic worker StateGraph")
    workflow = StateGraph(
        WorkerState,
        context_schema=WorkerDependencies,
    )

    nodes = {
        "capture": capture_node,
        "transition": transition_node,
        "ocr": ocr_node,
        "selection": selection_node,
        "reflex": attempt_reflex_replay,
        "reasoning": reasoning_node,
        "execution": execution_node,
    }
    for name, node in nodes.items():
        workflow.add_node(name, _instrument_node(name, node))

    workflow.add_conditional_edges(
        START,
        route_after_start,
        {"selection": "selection", "capture": "capture"},
    )
    workflow.add_edge("capture", "transition")
    workflow.add_edge("transition", "selection")
    workflow.add_edge("ocr", "transition")
    workflow.add_conditional_edges(
        "selection",
        route_after_selection,
        {
            "execution": "execution",
            "capture": "capture",
            "ocr": "ocr",
            "reflex": "reflex",
            "reasoning": "reasoning",
        },
    )
    workflow.add_conditional_edges(
        "reflex",
        route_after_reflex,
        {"execution": "execution", "reasoning": "reasoning"},
    )
    workflow.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {"execution": "execution", "capture": "capture"},
    )
    workflow.add_conditional_edges(
        "execution",
        route_after_execution,
        {
            "execution": "execution",
            "capture": "capture",
            "reasoning": "reasoning",
            "end": END,
        },
    )
    app = workflow.compile()
    logger.info("Atomic worker StateGraph compiled")
    return app


__all__ = [
    "build_graph",
    "route_after_execution",
    "route_after_reasoning",
    "route_after_reflex",
    "route_after_selection",
    "route_after_start",
]

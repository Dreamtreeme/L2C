"""비전 작업자의 원자 관찰-판정-선택-실행 그래프."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from agent.config import get_settings
from agent.graph.worker_reasoning import reasoning_node
from agent.graph.worker_reflex import reflex_node
from agent.graph.state import GraphState
from agent.graph.worker_collection import collection_node
from agent.graph.worker_observation import capture_node, ocr_node
from agent.graph.worker_recording import recording_node
from agent.graph.worker_selection import selection_node
from agent.graph.worker_transition import transition_node
from agent.graph.worker_execution import execution_node
from agent.graph.worker_state import return_to_job_results_for_url
from agent.graph.worker_state_contract import current_observation_matches_capture
from agent.observability.graph_events import graph_step
from agent.observability.reflex_paths import (
    reflex_selection_observation,
    reflex_transition_observation,
)
from agent.utils.logger import logger


def route_after_start(state: GraphState) -> str:
    """기존 관찰이 있으면 재사용하고, 없으면 첫 화면을 캡처한다."""

    if state.get("low_information_screen"):
        return "selection"
    if current_observation_matches_capture(state) or (
        state.get("current_markers")
        and state.get("current_page_role")
        and state.get("recent_images")
    ):
        return "selection"
    return "capture"


def route_after_transition(state: GraphState) -> str:
    """OCR 완료 관찰만 수집 상태에 반영하고 나머지는 바로 선택한다."""

    return (
        "collection"
        if current_observation_matches_capture(state)
        else "selection"
    )


def route_after_selection(state: GraphState) -> str:
    """결정론적 정책, 선택적 OCR, Reflex, LLM 순서로 다음 경로를 고른다."""

    if state.get("pending_action") is not None:
        return "execution"
    if state.get("low_information_screen"):
        return "capture"

    transition_result = dict(state.get("transition_result", {}) or {})
    if transition_result.get("status") == "pending":
        return "capture"
    if not current_observation_matches_capture(state):
        return (
            "ocr"
            if transition_result.get("needs_ocr")
            else "reasoning"
        )
    if return_to_job_results_for_url(state):
        return "reasoning"
    if state.get("job_detail_followup"):
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


def route_after_reflex(state: GraphState) -> str:
    """Reflex가 요청을 만들었을 때만 실행하고 나머지는 LLM으로 보낸다."""

    return (
        "execution"
        if str(getattr(state.get("pending_action"), "source", "") or "") == "reflex"
        else "reasoning"
    )


def route_after_reasoning(state: GraphState) -> str:
    """로딩 화면은 행동 없이 재관찰하고, 그 외 판단 결과만 실행한다."""

    if (
        state.get("pending_action") is None
        and (state.get("job_card_selection_trace") or {}).get("reason")
        == "screen_loading"
    ):
        return "capture"
    return "execution"


def route_after_recording(state: GraphState) -> str:
    """원자 실행 뒤 종료, 후속 정책, 새 관찰 또는 새 판단을 선택한다."""

    if state.get("is_finished", False):
        logger.info("Task marked as finished. Ending workflow.")
        return "end"
    if state.get("pending_human_approval", False):
        logger.info("Human approval required. Ending worker loop.")
        return "end"
    if state.get("error_count", 0) >= 3:
        logger.error("Too many errors. Forcing workflow to end.")
        return "end"
    if state.get("pending_action") is not None:
        return "execution"
    if state.get("transition_request"):
        return "capture"
    return "reasoning"


def _instrument_node(name: str, node: Callable[[GraphState], dict[str, Any]]):
    @wraps(node)
    def observed(state: GraphState) -> dict[str, Any]:
        with graph_step(name) as observation:
            result = node(state)
            request = (
                state.get("pending_action")
                if name == "execution"
                else (
                    result.get("pending_action")
                    if isinstance(result, dict)
                    else None
                )
            )
            action_source = str(getattr(request, "source", "") or "")
            observation.update(
                action_source=action_source,
                success=True,
            )
            if name == "execution" and request is not None:
                observation["action_names"] = [
                    str(call.name)
                    for call in getattr(request, "tool_calls", [])
                    if getattr(call, "name", "")
                ]
            if name == "ocr" and isinstance(result, dict):
                observation["analysis_mode"] = str(result.get("analysis_mode") or "full")
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


def build_graph(*, worker_runtime=None):
    """책임별 작업자 노드를 연결하고 컴파일한다."""

    logger.info("Building atomic worker StateGraph")
    workflow = StateGraph(GraphState)
    bind = worker_runtime.bind_node if worker_runtime is not None else lambda node: node

    nodes = {
        "capture": capture_node,
        "transition": transition_node,
        "ocr": ocr_node,
        "collection": collection_node,
        "selection": selection_node,
        "reflex": reflex_node,
        "reasoning": reasoning_node,
        "execution": execution_node,
        "recording": recording_node,
    }
    for name, node in nodes.items():
        workflow.add_node(name, bind(_instrument_node(name, node)))

    workflow.add_conditional_edges(
        START,
        route_after_start,
        {"selection": "selection", "capture": "capture"},
    )
    workflow.add_edge("capture", "transition")
    workflow.add_conditional_edges(
        "transition",
        route_after_transition,
        {"collection": "collection", "selection": "selection"},
    )
    workflow.add_edge("ocr", "transition")
    workflow.add_edge("collection", "selection")
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
    workflow.add_edge("execution", "recording")
    workflow.add_conditional_edges(
        "recording",
        route_after_recording,
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
    "route_after_recording",
    "route_after_reasoning",
    "route_after_reflex",
    "route_after_selection",
    "route_after_start",
    "route_after_transition",
]

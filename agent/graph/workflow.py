"""비전 작업자의 원자 관찰-판정-선택-실행 그래프."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from agent.config import get_settings
from agent.graph.worker_reasoning import reasoning_node
from agent.graph.state import GraphState
from agent.graph.worker_collection import apply_observation_node
from agent.graph.worker_observation import analyze_screen_node, capture_screen_node
from agent.graph.worker_recording import record_execution_node
from agent.graph.worker_selection import select_deterministic_action_node
from agent.graph.worker_transition import evaluate_transition_node
from agent.graph.worker_execution import action_node
from agent.graph.worker_state import detail_return_pending_for_url
from agent.runtime.reflex_runtime import reflex_node
from agent.observability.graph_events import graph_step
from agent.utils.logger import logger


def _reflex_enabled() -> bool:
    return get_settings().reflex.enabled


def _request_source(state: GraphState) -> str:
    request = state.get("pending_action")
    return str(getattr(request, "source", "") or "")


def route_after_start(state: GraphState) -> str:
    """준비된 시작 화면은 선택 단계로, 없으면 기존 LLM 시작 경로로 보낸다."""

    if state.get("low_information_screen"):
        return "selection"
    if state.get("ocr_complete") or (
        state.get("current_markers")
        and state.get("current_page_role")
        and state.get("recent_images")
    ):
        return "selection"
    return "reasoning"


def route_after_transition(state: GraphState) -> str:
    """OCR 완료 관찰만 수집 상태에 반영하고 나머지는 바로 선택한다."""

    return "collection" if state.get("ocr_complete") else "selection"


def route_after_selection(state: GraphState) -> str:
    """결정론적 정책, 선택적 OCR, Reflex, LLM 순서로 다음 경로를 고른다."""

    if state.get("pending_action") is not None:
        return "action"
    if state.get("low_information_screen"):
        return "capture"

    if state.get("transition_status") == "pending" and state.get("ocr_complete"):
        return "capture"
    if not state.get("ocr_complete"):
        return "ocr" if state.get("ocr_required") else "reasoning"
    if detail_return_pending_for_url(state):
        return "reasoning"
    if state.get("detail_followup_required"):
        return "reasoning"
    if state.get("transition_status") == "unknown" and (
        str(state.get("transition_source") or "").startswith("reflex")
        or state.get("transition_source") in {"page_policy", "duplicate_job_policy"}
        or state.get("transition_reason") in {"no_screen_change", "reflex_no_screen_change"}
    ):
        return "reasoning"
    if not _reflex_enabled():
        return "reasoning"
    return "reflex"


def route_after_reflex(state: GraphState) -> str:
    """Reflex가 요청을 만들었을 때만 실행하고 나머지는 LLM으로 보낸다."""

    return "action" if _request_source(state) == "reflex" else "reasoning"


def route_after_reasoning(state: GraphState) -> str:
    """로딩 화면은 행동 없이 재관찰하고, 그 외 판단 결과만 실행한다."""

    if (
        state.get("pending_action") is None
        and (state.get("result_card_selector_trace") or {}).get("reason")
        == "screen_loading"
    ):
        return "capture"
    return "action"


def route_after_action(state: GraphState) -> str:
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
        return "action"
    if state.get("pending_transition"):
        return "capture"
    return "reasoning"


def _instrument_node(name: str, node: Callable[[GraphState], dict[str, Any]]):
    @wraps(node)
    def observed(state: GraphState) -> dict[str, Any]:
        with graph_step(name) as observation:
            result = node(state)
            request = result.get("pending_action") if isinstance(result, dict) else None
            action_source = str(getattr(request, "source", "") or "")
            observation.update(
                action_source=action_source,
                success=True,
            )
            if name == "ocr" and isinstance(result, dict):
                observation["analysis_mode"] = str(result.get("analysis_mode") or "full")
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
        "capture": capture_screen_node,
        "transition": evaluate_transition_node,
        "ocr": analyze_screen_node,
        "collection": apply_observation_node,
        "selection": select_deterministic_action_node,
        "reflex": reflex_node,
        "reasoning": reasoning_node,
        "action": action_node,
        "recording": record_execution_node,
    }
    for name, node in nodes.items():
        workflow.add_node(name, bind(_instrument_node(name, node)))

    workflow.add_conditional_edges(
        START,
        route_after_start,
        {"selection": "selection", "reasoning": "reasoning"},
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
            "action": "action",
            "capture": "capture",
            "ocr": "ocr",
            "reflex": "reflex",
            "reasoning": "reasoning",
        },
    )
    workflow.add_conditional_edges(
        "reflex",
        route_after_reflex,
        {"action": "action", "reasoning": "reasoning"},
    )
    workflow.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {"action": "action", "capture": "capture"},
    )
    workflow.add_edge("action", "recording")
    workflow.add_conditional_edges(
        "recording",
        route_after_action,
        {
            "action": "action",
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
    "route_after_action",
    "route_after_reasoning",
    "route_after_reflex",
    "route_after_selection",
    "route_after_start",
    "route_after_transition",
]

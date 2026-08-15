"""비전 작업자의 원자 관찰-판정-선택-실행 그래프."""

from __future__ import annotations

from collections.abc import Mapping
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
from agent.graph.worker_review import review_node
from shared.schema.jd_schema import JobReviewStatus
from agent.runtime.job_card_queue import (
    can_select_pending_job_card,
    has_unresolved_job_card_queue,
    needs_job_results_navigation,
)
from agent.runtime.worker_state import current_observation_ready
from agent.observability.graph_events import graph_step
from agent.observability.reflex_paths import (
    reflex_selection_observation,
    reflex_step_observation,
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


def _route_before_ocr(state: WorkerState, transition_result: dict[str, Any]) -> str:
    """행동 결과 확인은 OCR로, 새 화면의 경험 규칙은 OCR 전에 검사한다."""

    status = str(transition_result.get("status") or "")
    if status == "unknown":
        return "reasoning"
    if status == "needs_ocr":
        return "ocr"
    if get_settings().reflex.enabled and not _reflex_missed_current_observation(state):
        return "reflex"
    return "ocr" if transition_result.get("needs_ocr") else "reasoning"


def _reflex_missed_current_observation(state: WorkerState) -> bool:
    trace = dict(state["replay"].get("reflex_trace") or {})
    observation_id = str(state["observation"].get("observation_id") or "")
    return bool(
        observation_id
        and trace.get("hit") is False
        and str(trace.get("observation_id") or "") == observation_id
    )


def route_after_selection(state: WorkerState) -> str:
    """결정론적 정책, 선택적 OCR, Reflex, LLM 순서로 다음 경로를 고른다."""

    if state["lifecycle"].get("is_finished", False):
        logger.info("Task completed by selection policy. Ending workflow.")
        return "end"
    if state["decision"].get("pending_action") is not None:
        return "execution"
    if state["observation"].get("low_information_screen"):
        return "capture"

    transition_result = dict(state["transition"].get("transition_result", {}) or {})
    if transition_result.get("status") == "pending":
        return "capture"
    if not current_observation_ready(state):
        return _route_before_ocr(state, transition_result)
    if has_unresolved_job_card_queue(state):
        return "reasoning"
    if transition_result.get("status") == "unknown":
        return "reasoning"
    if not get_settings().reflex.enabled:
        return "reasoning"
    if _reflex_missed_current_observation(state):
        return "reasoning"
    return "reflex"


def route_after_reflex(state: WorkerState) -> str:
    """적중하면 실행하고, OCR 전 불일치는 전체 화면 해석으로 보낸다."""

    request = state["decision"].get("pending_action")
    if request is not None and request.source == "reflex":
        return "execution"
    return "reasoning" if current_observation_ready(state) else "ocr"


def route_after_execution(state: WorkerState) -> str:
    """원자 실행 뒤 종료, 후속 정책, 새 관찰 또는 새 판단을 선택한다."""

    if state["lifecycle"].get("is_finished", False):
        logger.info("Task marked as finished. Ending workflow.")
        return "end"
    if state["transition"].get("error_count", 0) >= 3:
        logger.error("Too many errors. Forcing workflow to end.")
        return "end"
    if state["collection"].get("pending_job_draft") is not None:
        return "review"
    if state["transition"].get("transition_request"):
        return "capture"
    if needs_job_results_navigation(state) or can_select_pending_job_card(state):
        return "selection"
    return "reasoning"


def route_after_review(state: WorkerState) -> str:
    """검토 결과에 따라 같은 상세를 계속 읽거나 다음 카드로 이동한다."""

    if state["lifecycle"].get("is_finished", False):
        return "end"
    review = state["collection"].get("last_job_review")
    if review and review.status == JobReviewStatus.NEEDS_MORE:
        return "reasoning"
    return "selection"


def _instrument_node(
    name: str,
    node: Callable[
        [WorkerState, Runtime[WorkerDependencies]],
        Mapping[str, Any],
    ],
):
    @wraps(node)
    def observed(
        state: WorkerState,
        runtime: Runtime[WorkerDependencies],
    ) -> Mapping[str, Any]:
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
            if name == "reflex" and isinstance(result, dict):
                observation.update(reflex_selection_observation(result))
            if name == "transition" and isinstance(result, dict):
                observation.update(reflex_step_observation(result))
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
        "review": review_node,
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
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "reflex",
        route_after_reflex,
        {"execution": "execution", "ocr": "ocr", "reasoning": "reasoning"},
    )
    workflow.add_edge("reasoning", "execution")
    workflow.add_conditional_edges(
        "execution",
        route_after_execution,
        {
            "capture": "capture",
            "review": "review",
            "selection": "selection",
            "reasoning": "reasoning",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {"selection": "selection", "reasoning": "reasoning", "end": END},
    )
    app = workflow.compile()
    logger.info("Atomic worker StateGraph compiled")
    return app


__all__ = [
    "build_graph",
    "route_after_execution",
    "route_after_review",
    "route_after_reflex",
    "route_after_selection",
    "route_after_start",
]

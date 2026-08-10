"""결정론적 카드 선택과 LLM 호출로 다음 작업자 행동을 선택한다."""

from __future__ import annotations

import json
import time
from typing import Any

from langgraph.runtime import Runtime

from agent.observability.run_context import invoke_with_metrics, raise_if_cancelled
from agent.runtime.worker_contracts import (
    ActionRequest,
    WorkerState,
    action_event_results,
    action_event_transitions,
    action_request_from_model_response,
)
from agent.runtime.tool_schema import ACTION_TOOL_SCHEMAS
from agent.graph.worker_reasoning_prompt import build_reasoning_messages
from agent.runtime.job_card_selector import select_job_cards
from agent.runtime.transition_runtime import (
    detect_two_screen_transition_cycle,
)
from agent.utils.logger import logger
from agent.runtime.vision_worker_runtime import WorkerDependencies


def _get_ui_llm_with_tools(runtime: Runtime[WorkerDependencies]):
    """작업자 원자 도구를 바인딩한 모델을 재사용한다."""

    return runtime.context.vision.get_ui_model_with_tools(
        tuple(ACTION_TOOL_SCHEMAS),
        ACTION_TOOL_SCHEMAS,
    )


def _is_repeating(history: list[Any], count: int) -> bool:
    """최근 행동이 모두 같은 도구와 인자인지 검사한다."""

    if len(history) < count:
        return False
    recent = history[-count:]
    actions = {
        (
            action.get("action"),
            json.dumps(action.get("args", {}), sort_keys=True),
        )
        for action in recent
        if isinstance(action, dict)
    }
    return len(actions) == 1


def _loop_warning(
    state: WorkerState,
) -> tuple[str, int]:
    """최근 행동과 화면 전환 기록에서 반복 경고를 만든다."""

    action_history = action_event_results(
        state["transition"].get("action_events", []) or []
    )
    warning = ""
    error_increment = 0
    if _is_repeating(action_history, 3):
        repeated = action_history[-1]
        logger.warning(
            "Repeated action loop detected",
            action=repeated.get("action"),
            args=repeated.get("args"),
        )
        warning = (
            "\n\n[경고: 무한 루프 감지됨] 당신은 직전 3회 동안 동일한 행동"
            f"({repeated.get('action')}: {repeated.get('args')})을 반복했습니다. "
            "절대 동일한 행동(동일 마커 클릭 등)을 다시 수행하지 마십시오. "
            "새로운 마커를 클릭하거나, 스크롤을 하거나, 다른 방식으로 목표를 해결해야 합니다."
        )

    transition_cycle = detect_two_screen_transition_cycle(
        action_event_transitions(state["transition"].get("action_events", []) or [])
    )
    if transition_cycle.get("detected"):
        logger.warning(
            "Two-screen transition cycle detected",
            action_cycle=transition_cycle.get("action_cycle", []),
            same_screen_distances=transition_cycle.get(
                "same_screen_distances",
                [],
            ),
        )
        warning += (
            "\n\n[경고: 두 화면 왕복 반복 감지] 최근 화면이 A-B-A-B 순서로 반복됐습니다. "
            f"반복된 전환 행동: {transition_cycle.get('action_cycle', [])}. "
            "이전 입력 대상이나 이동 경로가 목표에 맞지 않을 가능성이 높습니다. "
            "같은 행동 순서를 다시 실행하지 말고 현재 화면에서 의미가 다른 입력창, 버튼 또는 이동 경로를 선택하십시오."
        )

    if _is_repeating(action_history, 4):
        logger.error("Persistent loop detected. Increasing error count to terminate.")
        error_increment = 1
    return warning, error_increment


def reasoning_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> dict[str, Any]:
    """카드 선택기로 먼저 판단하고 필요할 때만 LLM을 호출한다."""

    raise_if_cancelled()
    started = time.perf_counter()
    logger.info("Executing Reasoning Node")
    loop_warning, error_increment = _loop_warning(state)

    selector_request, selector_trace = select_job_cards(state)
    if selector_request is not None:
        logger.info(
            "Reasoning Node completed",
            component="reasoning",
            duration_sec=round(time.perf_counter() - started, 6),
            reasoning_mode="card_selection",
        )
        result = {
            "decision": {
                "pending_action": selector_request,
                "job_card_selection_trace": selector_trace,
            },
            "replay": {
                "reflex_trace": {
                    "hit": False,
                    "source": "card_selector",
                },
            },
        }
        if error_increment > 0:
            result["transition"] = {
                "error_count": (
                    state["transition"].get("error_count", 0) + error_increment
                )
            }
        return result

    reasoning_mode = (
        "general_after_card_selector" if selector_trace.get("attempted") else "general"
    )
    response = invoke_with_metrics(
        _get_ui_llm_with_tools(runtime),
        build_reasoning_messages(
            state,
            loop_warning,
        ),
        "vision_reasoning",
    )
    try:
        pending_action = action_request_from_model_response(
            response,
            allowed_tool_names=tuple(ACTION_TOOL_SCHEMAS),
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Model action request rejected", error=str(exc))
        pending_action = ActionRequest(
            source="llm",
            summary="모델이 유효하지 않은 도구 호출을 반환했습니다.",
            metadata={"validation_error": str(exc)},
        )
        error_increment += 1

    logger.info(
        "Reasoning Node completed",
        component="reasoning",
        duration_sec=round(time.perf_counter() - started, 6),
        reasoning_mode=reasoning_mode,
    )

    result = {
        "decision": {
            "pending_action": pending_action,
            "job_card_selection_trace": selector_trace,
        },
        "replay": {"reflex_trace": {"hit": False, "source": "reasoning"}},
    }
    if error_increment > 0:
        result["transition"] = {
            "error_count": (state["transition"].get("error_count", 0) + error_increment)
        }
    return result


__all__ = ["reasoning_node"]

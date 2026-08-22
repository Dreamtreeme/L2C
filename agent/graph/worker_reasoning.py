"""현재 화면을 해석해 다음 작업자 행동을 선택한다."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

from langgraph.runtime import Runtime

from agent.config import get_settings
from agent.observability.run_context import (
    RunCancelled,
    RunDeadlineExceeded,
    current_run_context,
    invoke_with_metrics,
    raise_if_cancelled,
)
from agent.runtime.worker_contracts import (
    ActionRequest,
    DecisionPatch,
    WorkerState,
    WorkerStage,
    action_event_results,
    action_event_transitions,
    action_request_from_model_response,
    build_action_request,
)
from agent.runtime.tool_schema import (
    ACTION_TOOL_SCHEMAS,
)
from agent.graph.worker_reasoning_prompt import build_reasoning_messages
from agent.graph.worker_execution_dispatch import can_request_job_detail_review
from agent.runtime.job_card_queue import has_unresolved_job_card_queue
from agent.runtime.transition_runtime import (
    detect_two_screen_transition_cycle,
)
from agent.utils.logger import logger
from agent.runtime.vision_worker_runtime import WorkerDependencies


_REASONING_COMPONENTS = {
    "vision_reasoning_lightweight",
    "vision_reasoning",
}


@dataclass
class _ReasoningUsage:
    """전체 관측치와 현재 업무 단계의 중단 예산을 분리한다."""

    stage: WorkerStage
    total_call_count: int
    stage_call_count: int
    estimated_cost_usd: float

    def record_call(self) -> None:
        self.total_call_count += 1
        self.stage_call_count += 1


def _reasoning_tool_names(state: WorkerState) -> tuple[str, ...]:
    """화면 역할 대신 실제 실행 전제조건으로 모델 도구를 제한한다."""

    observation = state["observation"]
    markers_ready = bool(
        observation.get("ocr_complete") and observation.get("current_markers")
    )
    unresolved_queue = has_unresolved_job_card_queue(state)
    review_available = can_request_job_detail_review(
        state,
        str(observation.get("current_url") or ""),
    )
    allowed = {
        "scroll",
        "press_key",
        "open_browser",
        "go_back",
        "close_current_tab",
        "switch_tab",
    }
    if markers_ready:
        allowed.update({"click_marker", "type_in_marker"})
        if not unresolved_queue:
            allowed.add("set_job_card_queue")
    if review_available:
        allowed.add("review_job_detail")
    elif not unresolved_queue:
        allowed.add("finish_task")
    return tuple(name for name in ACTION_TOOL_SCHEMAS if name in allowed)


def _get_ui_llm_with_tools(
    runtime: Runtime[WorkerDependencies],
    state: WorkerState,
    *,
    tier: str,
):
    """현재 실행 전제조건과 모델 단계가 같은 바인딩을 재사용한다."""

    tool_names = _reasoning_tool_names(state)
    return runtime.context.vision.get_ui_model_with_tools(
        tool_names,
        ACTION_TOOL_SCHEMAS,
        tier=tier,
    )


def _observed_reasoning_usage() -> tuple[int, float]:
    """현재 실행 컨텍스트가 기록한 화면 판단 호출 수와 비용을 읽는다."""

    observed_calls = 0
    spent_usd = 0.0
    context = current_run_context()
    if context is not None:
        usage = context.llm_budget_usage(_REASONING_COMPONENTS)
        observed_calls = int(usage.get("call_count") or 0)
        spent_usd = float(usage.get("estimated_cost_usd") or 0.0)
    return observed_calls, spent_usd


def _reasoning_usage(state: WorkerState) -> _ReasoningUsage:
    """상태의 누적 호출 수와 현재 단계 호출 수를 복원한다."""

    stage = state["progress"]["stage"]
    decision = state["decision"]
    observed_calls, spent_usd = _observed_reasoning_usage()
    stage_call_count = (
        int(decision.get("reasoning_stage_call_count") or 0)
        if decision.get("reasoning_stage") == stage
        else 0
    )
    return _ReasoningUsage(
        stage=stage,
        total_call_count=max(
            int(decision.get("reasoning_call_count") or 0),
            observed_calls,
        ),
        stage_call_count=stage_call_count,
        estimated_cost_usd=spent_usd,
    )


def _reasoning_stop_reason(usage: _ReasoningUsage) -> str:
    """현재 단계 호출 한도와 전체 실행 비용 한도를 검사한다."""

    observed_calls, spent_usd = _observed_reasoning_usage()
    usage.total_call_count = max(usage.total_call_count, observed_calls)
    usage.estimated_cost_usd = spent_usd
    settings = get_settings().vision
    if usage.stage_call_count >= settings.reasoning_call_limit:
        return "reasoning_call_limit"
    if (
        settings.reasoning_cost_limit_usd > 0
        and usage.estimated_cost_usd >= settings.reasoning_cost_limit_usd
    ):
        return "reasoning_cost_limit"
    return ""


def _decision_patch(
    request: ActionRequest,
    usage: _ReasoningUsage,
) -> DecisionPatch:
    return {
        "pending_action": request,
        "reasoning_call_count": usage.total_call_count,
        "reasoning_stage": usage.stage,
        "reasoning_stage_call_count": usage.stage_call_count,
    }


def _reasoning_stop(
    usage: _ReasoningUsage,
    reason: str,
) -> dict[str, Any]:
    _reasoning_stop_reason(usage)
    model_failed = reason == "primary_reasoning_failed"
    summary = (
        "화면 판단 모델의 복구 호출이 실패해 현재까지 수집한 결과로 종료합니다."
        if model_failed
        else "화면 판단 예산에 도달해 현재까지 수집한 결과로 종료합니다."
    )
    result = (
        "화면 판단 모델이 유효한 행동을 반환하지 않아 현재까지 확보한 "
        "정보만으로 수집을 종료했습니다."
        if model_failed
        else "화면 판단 호출 한도에 도달해 현재까지 확보한 정보만으로 수집을 종료했습니다."
    )
    request = build_action_request(
        "reasoning_policy",
        summary,
        [
            {
                "name": "finish_task",
                "args": {"result": result},
                "id": "reasoning_policy_stop",
            }
        ],
        metadata={
            "reason": reason,
            "call_count": usage.total_call_count,
            "stage": usage.stage,
            "stage_call_count": usage.stage_call_count,
            "estimated_cost_usd": usage.estimated_cost_usd,
        },
    )
    logger.warning(
        "Reasoning stopped by policy",
        reason=reason,
        call_count=usage.total_call_count,
        stage=usage.stage,
        stage_call_count=usage.stage_call_count,
        estimated_cost_usd=usage.estimated_cost_usd,
    )
    return {
        "decision": _decision_patch(request, usage),
        "replay": {
            "reflex_trace": {"hit": False, "source": "reasoning_policy"}
        },
    }


def _invoke_reasoning_model(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
    loop_warning: str,
    *,
    tier: str,
) -> ActionRequest:
    tool_names = _reasoning_tool_names(state)
    component = (
        "vision_reasoning_lightweight"
        if tier == "lightweight"
        else "vision_reasoning"
    )
    response = invoke_with_metrics(
        _get_ui_llm_with_tools(runtime, state, tier=tier),
        build_reasoning_messages(state, loop_warning),
        component,
    )
    request = action_request_from_model_response(
        response,
        allowed_tool_names=tool_names,
    )
    if not request.tool_calls:
        raise ValueError("모델이 실행할 도구를 선택하지 않았습니다.")
    return request


def _choose_reasoning_action(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
    loop_warning: str,
    *,
    initial_tier: str,
    usage: _ReasoningUsage,
) -> tuple[ActionRequest | None, str, str]:
    """경량 판단이 실패한 경우에만 고성능 모델로 한 번 복구한다."""

    tiers = (
        ("primary",)
        if initial_tier == "primary"
        else ("lightweight", "primary")
    )
    retry_warning = loop_warning
    for tier in tiers:
        stop_reason = _reasoning_stop_reason(usage)
        if stop_reason:
            return None, tier, stop_reason
        try:
            request = _invoke_reasoning_model(
                state,
                runtime,
                retry_warning,
                tier=tier,
            )
            usage.record_call()
            return request, tier, ""
        except (RunCancelled, RunDeadlineExceeded):
            raise
        except Exception as exc:
            usage.record_call()
            logger.warning(
                "Worker reasoning attempt failed",
                tier=tier,
                error=str(exc),
            )
            retry_warning = (
                f"{loop_warning}\n\n[직전 화면 판단의 도구 호출이 거부됨]\n"
                f"- 오류: {exc}\n"
                "같은 잘못된 도구 인자를 반복하지 말고 현재 화면 근거로 수정하십시오."
            )
    _reasoning_stop_reason(usage)
    return None, "primary", "primary_reasoning_failed"


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
    """현재 화면에서 다음 원자 행동 또는 공고 카드 큐를 선택한다."""

    raise_if_cancelled()
    started = time.perf_counter()
    logger.info("Executing Reasoning Node")
    loop_warning, error_increment = _loop_warning(state)
    usage = _reasoning_usage(state)
    budget_reason = _reasoning_stop_reason(usage)
    if budget_reason:
        return _reasoning_stop(usage, budget_reason)

    transition_status = str(
        state["transition"].get("transition_result", {}).get("status") or ""
    )
    initial_tier = (
        "primary"
        if transition_status == "unknown" or loop_warning
        else "lightweight"
    )
    pending_action, tier, stop_reason = (
        _choose_reasoning_action(
            state,
            runtime,
            loop_warning,
            initial_tier=initial_tier,
            usage=usage,
        )
    )
    if pending_action is None:
        return _reasoning_stop(usage, stop_reason)
    reasoning_mode = "general_recovery" if tier != initial_tier else "general"

    logger.info(
        "Reasoning Node completed",
        component="reasoning",
        duration_sec=round(time.perf_counter() - started, 6),
        reasoning_mode=reasoning_mode,
        model_tier=tier,
    )

    result = {
        "decision": _decision_patch(pending_action, usage),
        "replay": {"reflex_trace": {"hit": False, "source": "reasoning"}},
    }
    if error_increment > 0:
        result["transition"] = {
            "error_count": (state["transition"].get("error_count", 0) + error_increment)
        }
    return result


__all__ = ["reasoning_node"]

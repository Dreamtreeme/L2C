"""LLM 호출 전에 실행하는 결정론적 행동 선택 정책."""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from agent.config import get_settings
from agent.runtime.worker_contracts import (
    TransitionResult,
    WorkerState,
    build_action_request,
)
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.runtime.worker_state import (
    count_mode_from_state,
    target_count_from_state,
)
from agent.runtime.job_card_queue import (
    active_job_card,
    job_card_queue_scope_complete,
    needs_job_results_navigation,
    next_job_card_request,
    skip_active_job_card,
)
from agent.runtime.site_context import looks_like_job_detail_url
from agent.utils.logger import logger


def _low_information_stop() -> dict[str, Any]:
    request = build_action_request(
        "screen_policy",
        "stop after repeated low-information captures",
        [
            {
                "name": "finish_task",
                "args": {
                    "result": (
                        "브라우저 화면이 준비되지 않아 현재까지 확보한 정보만으로 "
                        "수집을 종료했습니다."
                    )
                },
                "id": "screen_policy_low_information_stop",
            }
        ],
    )
    return {
        "decision": {"pending_action": request},
    }


def _skip_duplicate_detail(
    state: WorkerState,
    *,
    transition_result: TransitionResult,
    current_url: str,
    duplicate_trace: dict[str, Any],
) -> dict[str, Any]:
    queue = skip_active_job_card(
        [
            dict(item)
            for item in state["collection"].get("job_card_queue", []) or []
            if isinstance(item, dict)
        ],
        reason="existing_detail_url",
        url=current_url,
        job_id=duplicate_trace.get("job_id"),
    )
    queue_complete = job_card_queue_scope_complete(
        queue,
        count_mode=count_mode_from_state(state),
        target_count=target_count_from_state(state),
    )
    update: dict[str, Any] = {
        "transition": {
            "transition_request": None,
            "transition_result": {
                **transition_result,
                "status": "ready",
                "outcome": "existing_job_detail",
                "reason": "existing_job_detail",
                "needs_ocr": False,
            },
        },
        "collection": {"job_card_queue": queue},
    }
    if queue_complete:
        update["progress"] = {
            "stage": "finished",
        }
        update["lifecycle"] = {
            "is_finished": True,
            "completion_reason": (
                "visible_scope_completed"
                if count_mode_from_state(state) == "visible_all"
                else "target_reached"
            ),
        }
    return update


def _job_results_navigation_action(action_name: str) -> dict[str, Any]:
    reason = (
        "뒤로가기가 상세 화면을 바꾸지 않아 현재 상세 탭을 닫습니다."
        if action_name == "close_current_tab"
        else "수집이 끝난 상세 화면에서 검색 결과로 돌아갑니다."
    )
    request = build_action_request(
        "job_results_navigation",
        reason,
        [
            {
                "name": action_name,
                "args": {
                    "reason": reason,
                    "expected_after": "검색 결과 목록이 보인다.",
                    "page_role": "job_detail",
                    "risk_level": "safe_navigation",
                },
                "id": f"job_results_navigation_{action_name}",
            }
        ],
    )
    return {"decision": {"pending_action": request}}


def _select_job_results_navigation(
    state: WorkerState,
    transition_result: TransitionResult,
) -> dict[str, Any] | None:
    """상세 완료 후 목록 복귀의 결정된 두 단계를 선택한다."""

    if not needs_job_results_navigation(state):
        return None
    action = str(transition_result.get("action") or "")
    status = str(transition_result.get("status") or "")
    reason = str(transition_result.get("reason") or "")
    if (
        action == "go_back"
        and status == "unknown"
        and reason in {"no_screen_change", "reflex_no_screen_change"}
    ):
        return _job_results_navigation_action("close_current_tab")
    if action in {"go_back", "close_current_tab"}:
        return None
    return _job_results_navigation_action("go_back")


def _select_duplicate_detail(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
    *,
    current_url: str,
    transition_result: TransitionResult,
) -> dict[str, Any] | None:
    active_card = active_job_card(
        list(state["collection"].get("job_card_queue", []) or [])
    )
    if not active_card or not looks_like_job_detail_url(current_url):
        return None
    duplicate_trace = runtime.context.data.find_existing_job_url(
        current_url,
        list(state["collection"].get("job_captures", [])),
    )
    if not duplicate_trace.get("matched"):
        return None
    logger.info(
        "Existing job detail selected for skip",
        url=current_url,
        source=duplicate_trace.get("source", ""),
    )
    return _skip_duplicate_detail(
        state,
        transition_result=transition_result,
        current_url=current_url,
        duplicate_trace=duplicate_trace,
    )


def _select_queued_card(
    state: WorkerState,
) -> dict[str, Any] | None:
    observation = state["observation"]
    if not observation.get("ocr_complete"):
        return None
    markers = list(observation.get("current_markers") or [])
    request, trace = next_job_card_request(
        state,
        markers,
    )
    if request is None:
        logger.debug(
            "Job card queue selection skipped",
            reason=trace.get("reason", ""),
        )
        return None
    logger.info(
        "Job card queue action selected",
        queue_id=trace.get("queue_id", ""),
        ocr_skipped=False,
    )
    return {"decision": {"pending_action": request}}


def selection_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> dict[str, Any]:
    """중복 공고와 확인된 목록 화면의 큐를 검사해 원자 행동 하나를 선택한다."""

    decision = state["decision"]
    observation = state["observation"]
    transition = state["transition"]
    replay = state["replay"]
    if decision.get("pending_action") is not None:
        return {}

    capture_count = int(observation.get("low_information_capture_count") or 0)
    if observation.get("low_information_screen"):
        if capture_count >= get_settings().vision.low_information_max_capture_cycles:
            return _low_information_stop()
        return {}

    if replay.get("replay_session"):
        return {}

    current_url = str(observation.get("current_url") or "")
    transition_result = transition.get("transition_result", {}) or {}
    duplicate_update = _select_duplicate_detail(
        state,
        runtime,
        current_url=current_url,
        transition_result=transition_result,
    )
    if duplicate_update is not None:
        return duplicate_update
    navigation_update = _select_job_results_navigation(state, transition_result)
    if navigation_update is not None:
        return navigation_update
    return (
        _select_queued_card(state)
        or {}
    )


__all__ = ["selection_node"]

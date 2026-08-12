"""작업자 행동 실행 결과를 그래프 상태와 후속 행동에 반영한다."""

from __future__ import annotations

from typing import Any

from agent.graph import worker_execution_dispatch
from agent.graph.worker_transition_recording import set_transition_request
from agent.runtime.worker_contracts import (
    ActionRequest,
    apply_worker_state_update,
    build_action_request,
)
from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.runtime.worker_state import (
    job_capture_count,
    count_mode_from_state,
    target_count_from_state,
)
from agent.runtime.job_card_queue import (
    activate_job_card,
    complete_active_job_card,
    job_card_click_matches_queue,
    job_card_queue_scope_complete,
    pending_job_cards,
    resolved_job_card_count,
)
from agent.runtime.worker_actions import (
    DIRECT_SCREEN_ACTION_SOURCES,
    URL_STALE_ACTIONS,
)


def raise_for_action_failure(result: dict[str, Any]) -> None:
    """도구가 반환값으로 보고한 실패를 실행 예외로 통일한다."""

    if result.get("status") != "error":
        return
    message = result.get("error") or result.get("result") or "action failed"
    raise RuntimeError(str(message))


def execute_ui_action(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    call_metadata: dict[str, Any],
    tool_call_id: str,
    action_sequence: int,
) -> tuple[dict[str, Any], bool]:
    """물리 행동을 실행하고 다음 캡처가 확인할 전환 정보를 만든다."""

    state = context.state
    observation = state["observation"]
    current_url = str(observation.get("current_url") or "")
    result = worker_execution_dispatch.dispatch_ui_action(
        action_name,
        args,
        context.marker_bbox,
        action_tools=context.worker_runtime.get_action_tools(),
        current_url=current_url,
    )
    raise_for_action_failure(result)
    screen_changed = True
    if action_name == "open_browser":
        raw_result_payload = result.get("result")
        result_payload = (
            raw_result_payload if isinstance(raw_result_payload, dict) else {}
        )
        screen_changed = bool(result_payload.get("opened"))
        if not screen_changed and not observation.get("ui_context"):
            screen_changed = True
        observation["current_url"] = str(result_payload.get("url") or args["url"])
        observation["current_url_stale"] = screen_changed
    else:
        observation["current_url_stale"] = (
            bool(observation.get("current_url_stale", True))
            or action_name in URL_STALE_ACTIONS
        )

    context.screen_changed = context.screen_changed or screen_changed
    if screen_changed:
        transition_source = (
            context.action_request.source
            if context.action_request.source in DIRECT_SCREEN_ACTION_SOURCES
            else "autonomous"
        )
        transition_source = str(
            call_metadata.get("transition_source") or transition_source
        )
        set_transition_request(
            context,
            action_sequence,
            action_name,
            args,
            transition_source,
        )
    return result, screen_changed


def activate_clicked_job_card(
    context: WorkerExecutionContext,
    result: dict[str, Any],
    action_name: str,
    action_context_args: dict[str, Any],
) -> None:
    """성공한 큐 카드 클릭을 활성 공고로 표시한다."""

    collection = context.state["collection"]
    job_card_queue = list(collection.get("job_card_queue", []) or [])
    if (
        result.get("status") != "success"
        or action_name != "click_marker"
        or not job_card_click_matches_queue(
            job_card_queue,
            action_context_args,
        )
    ):
        return
    collection["job_card_queue"] = activate_job_card(
        job_card_queue,
        action_context_args,
    )


def _first_pending_job_card_action(
    cards: list[dict[str, Any]],
) -> ActionRequest | None:
    if not cards:
        return None
    first_card = cards[0]
    marker_id = first_card.get("source_marker_id")
    if marker_id is None:
        return None
    queue_id = str(first_card.get("queue_id") or "")
    return build_action_request(
        "job_card_queue",
        "job_card_queue_first_item",
        [
            {
                "name": "click_marker",
                "args": {
                    "marker_id": marker_id,
                    "target_label": first_card.get("title", ""),
                    "target_role": "job_card",
                    "target_component": "job_card_title",
                    "page_role": "search",
                    "reason": ("job card queue stored; open the first pending card"),
                    "expected_after": "selected job detail page is visible",
                },
                "id": f"job_card_queue_{queue_id or 'first'}",
                "metadata": {"queue_id": queue_id},
            }
        ],
    )


def _apply_job_card_queue_result(
    context: WorkerExecutionContext,
    result: dict[str, Any],
) -> ActionRequest | None:
    state = context.state
    collection = state["collection"]
    job_card_queue = list(collection.get("job_card_queue", []) or [])
    pending_cards = pending_job_cards(job_card_queue)
    first_card_action = _first_pending_job_card_action(pending_cards)
    if first_card_action is not None:
        return first_card_action

    if job_card_queue_scope_complete(
        job_card_queue,
        count_mode=count_mode_from_state(state),
        target_count=target_count_from_state(state),
    ):
        state["lifecycle"]["is_finished"] = True
        resolved_count = resolved_job_card_count(job_card_queue)
        result["auto_finished"] = True
        result["resolved_count"] = resolved_count
    return None


def _apply_job_detail_completion(
    context: WorkerExecutionContext,
    result: dict[str, Any],
) -> None:
    state = context.state
    collection = state["collection"]
    job_card_queue = list(collection.get("job_card_queue", []) or [])
    target_count = target_count_from_state(state)
    collected_count = job_capture_count(state)

    job_card_queue = complete_active_job_card(job_card_queue)
    collection["job_card_queue"] = job_card_queue
    result["detail_policy"] = "required_fields_complete"
    pending_cards = pending_job_cards(job_card_queue)
    resolved_count = max(
        collected_count,
        resolved_job_card_count(job_card_queue),
    )
    if pending_cards or (target_count > 0 and resolved_count < target_count):
        result["next_step"] = "navigate_to_job_results"
        result["pending_count"] = len(pending_cards)
        return

    if (
        count_mode_from_state(state) == "visible_all"
        and job_card_queue
        and not pending_cards
    ):
        state["lifecycle"]["is_finished"] = True
        result["auto_finished"] = True
        result["count_mode"] = "visible_all"
        result["collected_count"] = collected_count


def _apply_collection_target_completion(
    context: WorkerExecutionContext,
    result: dict[str, Any],
) -> None:
    state = context.state
    collection = state["collection"]
    job_card_queue = list(collection.get("job_card_queue", []) or [])
    target_count = target_count_from_state(state)
    collected_count = job_capture_count(state)
    resolved_count = max(
        collected_count,
        resolved_job_card_count(job_card_queue),
    )
    if target_count <= 0 or resolved_count < target_count:
        return

    state["lifecycle"]["is_finished"] = True
    result["auto_finished"] = True
    result["target_count"] = target_count
    result["collected_count"] = collected_count
    result["resolved_count"] = resolved_count


def execute_state_action(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
) -> tuple[dict[str, Any], ActionRequest | None]:
    """상태 행동을 실행하고 카드·상세 완료 후속 효과를 반영한다."""

    state = context.state
    observation = state["observation"]
    collection = state["collection"]
    current_captures = list(collection.get("job_captures", []))
    outcome = worker_execution_dispatch.dispatch_state_action(
        action_name,
        args,
        current_captures,
        current_url=str(observation.get("current_url") or ""),
        state=state,
        data_services=context.data_services,
    )
    raise_for_action_failure(outcome.result)
    context.state = apply_worker_state_update(context.state, outcome.state_update)
    result = outcome.result
    follow_up: ActionRequest | None = None
    if action_name == "set_job_card_queue":
        follow_up = _apply_job_card_queue_result(context, result)

    is_successful_detail_update = (
        action_name == "finish_detail_reading" and result.get("status") == "success"
    )
    if is_successful_detail_update:
        _apply_job_detail_completion(context, result)
        _apply_collection_target_completion(context, result)
    return result, follow_up


__all__ = [
    "activate_clicked_job_card",
    "execute_state_action",
    "execute_ui_action",
    "raise_for_action_failure",
]

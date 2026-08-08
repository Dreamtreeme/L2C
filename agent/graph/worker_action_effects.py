"""작업자 행동 실행 결과를 그래프 상태와 후속 행동에 반영한다."""

from __future__ import annotations

from typing import Any

from agent.graph import worker_execution_dispatch
from agent.runtime.worker_contracts import ActionRequest, build_action_request
from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.runtime.worker_state import (
    count_mode_from_state,
    extracted_job_count,
    target_count_from_state,
)
from agent.runtime.job_card_queue import (
    activate_job_card,
    complete_active_job_card,
    job_card_click_matches_queue,
    job_card_queue_scope_complete,
    normalized_return_action,
    pending_job_cards,
    resolved_job_card_count,
)
from agent.runtime.transition_runtime import latest_no_effect_transition
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

    result = worker_execution_dispatch.dispatch_ui_action(
        action_name,
        args,
        context.marker_bbox,
        current_url=context.current_url,
    )
    raise_for_action_failure(result)
    screen_changed = True
    if action_name == "open_browser":
        result_payload = (
            result.get("result")
            if isinstance(result.get("result"), dict)
            else {}
        )
        screen_changed = bool(result_payload.get("opened"))
        if not screen_changed and not context.state.get("ui_context"):
            screen_changed = True
        context.current_url = result_payload.get("url") or args["url"]
        context.current_url_stale = screen_changed
    else:
        context.current_url_stale = (
            context.current_url_stale
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
        context.set_transition_request(
            action_sequence,
            action_name,
            args,
            transition_source,
            tool_call_id,
        )
    return result, screen_changed


def activate_clicked_job_card(
    context: WorkerExecutionContext,
    result: dict[str, Any],
    action_name: str,
    action_context_args: dict[str, Any],
) -> None:
    """성공한 큐 카드 클릭을 활성 공고로 표시한다."""

    if (
        result.get("status") != "success"
        or action_name != "click_marker"
        or not job_card_click_matches_queue(
            context.job_card_queue,
            action_context_args,
        )
    ):
        return
    context.job_card_queue = activate_job_card(
        context.job_card_queue,
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
                    "reason": (
                        "job card queue stored; open the first pending card"
                    ),
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
    pending_cards = pending_job_cards(context.job_card_queue)
    first_card_action = _first_pending_job_card_action(pending_cards)
    if first_card_action is not None:
        return first_card_action

    if job_card_queue_scope_complete(
        context.job_card_queue,
        count_mode=count_mode_from_state(context.state),
        target_count=target_count_from_state(context.state),
    ):
        context.is_finished = True
        resolved_count = resolved_job_card_count(
            context.job_card_queue
        )
        result["auto_finished"] = True
        result["resolved_count"] = resolved_count
    return None


def _apply_state_action_update(
    context: WorkerExecutionContext,
    update: worker_execution_dispatch.StateActionUpdate,
) -> None:
    """명시된 필드만 실행 문맥에 반영한다."""

    if update.job_card_queue is not None:
        context.job_card_queue = list(update.job_card_queue)
    if update.job_results_memory is not None:
        context.job_results_memory = dict(update.job_results_memory)
    if update.job_results_availability is not None:
        context.job_results_availability = dict(
            update.job_results_availability
        )
    if update.job_detail_buffer is not None:
        context.job_detail_buffer = dict(update.job_detail_buffer)
    if update.job_detail_coverage is not None:
        context.job_detail_coverage = dict(update.job_detail_coverage)
    if update.job_detail_followup is not None:
        context.job_detail_followup = dict(update.job_detail_followup)


def _confirmed_job_results_return_action(
    context: WorkerExecutionContext,
    result: dict[str, Any],
) -> ActionRequest | None:
    no_effect_return = latest_no_effect_transition(context.state)
    failed_return_action = str(no_effect_return.get("action") or "")
    return_action = normalized_return_action(
        context.job_results_memory.get("return_action")
    )
    if (
        not return_action
        or failed_return_action == return_action.get("name")
    ):
        result["detail_policy"] = "return_requires_reasoning"
        if failed_return_action:
            result["failed_return_action"] = failed_return_action
        return None

    return_name = str(return_action["name"])
    return_args = {
        **dict(return_action.get("args") or {}),
        "reason": "이전에 확인된 검색 결과 복귀 행동을 재사용합니다.",
        "expected_after": "검색 결과 목록이 표시된다.",
    }
    return build_action_request(
        "page_policy",
        "reuse_confirmed_result_return_action",
        [
            {
                "name": return_name,
                "args": return_args,
                "id": f"detail_policy_{return_name}",
            }
        ],
    )


def _apply_job_detail_completion(
    context: WorkerExecutionContext,
    result: dict[str, Any],
    action_sequence: int,
) -> ActionRequest | None:
    target_count = target_count_from_state(context.state)
    collected_count = extracted_job_count(context.current_jobs)

    context.job_card_queue = complete_active_job_card(
        context.job_card_queue
    )
    result["detail_policy"] = "required_fields_complete"
    pending_cards = pending_job_cards(context.job_card_queue)
    resolved_count = max(
        collected_count,
        resolved_job_card_count(context.job_card_queue),
    )
    if pending_cards or (
        target_count > 0 and resolved_count < target_count
    ):
        context.return_to_job_results = {
            "url": context.current_url,
            "reason": "required_fields_complete",
            "pending_count": len(pending_cards),
            "completed_action_seq": action_sequence,
        }
        return _confirmed_job_results_return_action(context, result)

    if (
        count_mode_from_state(context.state) == "visible_all"
        and context.job_card_queue
        and not pending_cards
    ):
        context.is_finished = True
        result["auto_finished"] = True
        result["count_mode"] = "visible_all"
        result["collected_count"] = collected_count
    return None


def _apply_collection_target_completion(
    context: WorkerExecutionContext,
    result: dict[str, Any],
) -> None:
    target_count = target_count_from_state(context.state)
    collected_count = extracted_job_count(context.current_jobs)
    resolved_count = max(
        collected_count,
        resolved_job_card_count(context.job_card_queue),
    )
    if target_count <= 0 or resolved_count < target_count:
        return

    context.return_to_job_results = {}
    context.is_finished = True
    result["auto_finished"] = True
    result["target_count"] = target_count
    result["collected_count"] = collected_count
    result["resolved_count"] = resolved_count


def execute_state_action(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    action_sequence: int,
) -> tuple[dict[str, Any], ActionRequest | None]:
    """상태 행동을 실행하고 카드·상세 완료 후속 효과를 반영한다."""

    outcome = worker_execution_dispatch.dispatch_state_action(
        action_name,
        args,
        context.current_jobs,
        current_url=context.current_url,
        state=context.state_for_dispatch(),
    )
    raise_for_action_failure(outcome.result)
    context.current_jobs = outcome.jobs
    _apply_state_action_update(context, outcome.state_update)
    result = outcome.result
    follow_up: ActionRequest | None = None
    if action_name == "set_job_card_queue":
        follow_up = _apply_job_card_queue_result(context, result)

    is_successful_detail_update = (
        action_name == "finish_detail_reading"
        and result.get("status") == "success"
    )
    if is_successful_detail_update:
        follow_up = (
            _apply_job_detail_completion(
                context,
                result,
                action_sequence,
            )
            or follow_up
        )

    if (
        action_name in {"update_extracted_info", "finish_detail_reading"}
        and result.get("status") == "success"
    ):
        _apply_collection_target_completion(context, result)
    return result, follow_up


__all__ = [
    "activate_clicked_job_card",
    "execute_state_action",
    "execute_ui_action",
    "raise_for_action_failure",
]

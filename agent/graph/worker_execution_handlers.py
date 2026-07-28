"""행동 종류별 검증, 실행, 후속 정책을 처리한다."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.action_request import ActionRequest, build_action_request
from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph import worker_execution_dispatch
from agent.graph.worker_execution_policy import (
    auto_finish_on_target_enabled,
    compact_action_args,
    is_detail_update,
    repeats_no_effect_target,
    sensitive_action_reason,
)
from agent.graph.worker_resources import (
    check_current_reasoning_screen,
    get_action_tools,
)
from agent.graph.worker_state import (
    count_mode_from_state,
    return_to_job_results_for_url,
    extracted_job_count,
    target_count_from_state,
)
from agent.runtime.action_validation import text_input_target_rejection
from agent.runtime.followup_runtime import select_followup_action
from agent.runtime.job_card_queue import (
    completed_job_card_count,
    complete_active_job_card,
    activate_job_card,
    normalized_return_action,
    pending_job_cards,
    job_card_click_matches_queue,
    job_card_queue_scope_complete,
)
from agent.runtime.transition_runtime import latest_no_effect_transition
from agent.utils.logger import logger


UI_ACTIONS = frozenset(
    {
        "click_marker",
        "type_in_marker",
        "scroll",
        "press_key",
        "open_browser",
        "close_browser",
        "close_current_tab",
        "switch_tab",
        "go_back",
    }
)
STATE_ACTIONS = frozenset(
    {
        "update_extracted_info",
        "finish_detail_reading",
        "set_job_card_queue",
    }
)
RETURN_ACTIONS = frozenset({"go_back", "close_current_tab", "switch_tab"})
URL_STALE_ACTIONS = frozenset(
    {
        "click_marker",
        "press_key",
        "open_browser",
        "close_browser",
        "close_current_tab",
        "switch_tab",
        "go_back",
    }
)
DIRECT_SCREEN_ACTION_SOURCES = frozenset(
    {
        "reflex",
        "job_card_queue",
        "page_policy",
        "duplicate_job_policy",
        "followup_strategy",
    }
)


def _guard_return_to_results(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    before_snapshot: dict[str, Any],
    step_started: float,
) -> bool:
    return_pending = return_to_job_results_for_url(
        {
            **context.state,
            "return_to_job_results": context.return_to_job_results,
            "current_url": context.current_url,
        },
        context.current_url,
    )
    if not return_pending:
        return False
    if action_name not in STATE_ACTIONS and (
        action_name not in UI_ACTIONS or action_name in RETURN_ACTIONS
    ):
        return False

    context.append_guard_result(
        action_name,
        args,
        before_snapshot,
        status="skipped",
        reason="return_to_job_results",
        message=(
            "상세 수집이 이미 완료되었습니다. 같은 공고를 더 읽지 말고 "
            "검색 결과 화면으로 복귀해야 합니다."
        ),
        step_started=step_started,
    )
    return True


def _guard_ui_action(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    before_snapshot: dict[str, Any],
    step_started: float,
) -> bool:
    if (
        action_name in {"click_marker", "type_in_marker"}
        and context.action_request.source not in DIRECT_SCREEN_ACTION_SOURCES
    ):
        guard_result = check_current_reasoning_screen(
            context.state,
            marker_id=args.get("marker_id"),
        )
        if guard_result.get("stale"):
            context.append_guard_result(
                action_name,
                args,
                before_snapshot,
                status="skipped",
                reason="screen_changed_during_reasoning",
                message=(
                    "Skipped UI action because the screen changed while reasoning; "
                    "a fresh perception is required."
                ),
                step_started=step_started,
                observation_required=True,
                details=guard_result,
            )
            return True

    no_effect_transition = latest_no_effect_transition(context.state)
    if repeats_no_effect_target(
        no_effect_transition,
        action_name,
        args,
    ):
        context.append_guard_result(
            action_name,
            args,
            before_snapshot,
            status="skipped",
            reason="same_screen_no_effect_action_blocked",
            message=(
                "Blocked an atomic UI action that already had no effect on this "
                "screen. Choose another navigation method."
            ),
            step_started=step_started,
        )
        return True

    sensitive_reason = sensitive_action_reason(
        {
            **context.state,
            "current_markers": context.current_markers,
        },
        action_name,
        args,
    )
    if sensitive_reason:
        context.require_human_approval(
            action_name,
            args,
            sensitive_reason,
            before_snapshot,
            step_started,
        )
        return True

    if action_name == "type_in_marker":
        target_rejection = text_input_target_rejection(
            context.current_markers,
            args.get("marker_id"),
        )
        if target_rejection:
            context.append_guard_result(
                action_name,
                args,
                before_snapshot,
                status="error",
                reason=target_rejection["reason"],
                message=(
                    "Blocked type_in_marker because the selected marker does not "
                    "look like a text input target. Choose the visible input "
                    "container or placeholder marker."
                ),
                step_started=step_started,
                increments_error=True,
            )
            return True

    return False


def _execute_ui_action(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    call_metadata: dict[str, Any],
    tool_call_id: str,
    action_sequence: int,
) -> tuple[dict[str, Any], bool]:
    result = worker_execution_dispatch.dispatch_ui_action(
        action_name,
        args,
        context.marker_bbox,
        current_url=context.current_url,
    )
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
        contract = (
            call_metadata.get("transition_contract")
            or context.reflex_transition_contracts.get(tool_call_id)
        )
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
            contract,
            transition_source,
            tool_call_id,
            str(call_metadata.get("strategy_key") or ""),
        )
    return result, screen_changed


def _activate_clicked_job_card(
    context: WorkerExecutionContext,
    result: dict[str, Any],
    action_name: str,
    action_context_args: dict[str, Any],
) -> None:
    if (
        result.get("status") != "success"
        or action_name != "click_marker"
        or not job_card_click_matches_queue(
            context.job_card_queue,
            action_context_args,
        )
    ):
        return
    (
        context.job_card_queue,
        context.active_job_card,
    ) = activate_job_card(
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
    context.job_card_queue = list(
        result.pop("_job_card_queue", []) or []
    )
    context.job_results_memory = dict(
        result.pop("_job_results_memory", {}) or {}
    )
    observed_availability = dict(
        result.pop("_job_results_availability", {}) or {}
    )
    if observed_availability:
        context.job_results_availability = observed_availability

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
        resolved_count = completed_job_card_count(
            context.job_card_queue
        )
        context.collected_data.append(
            "Auto-finished after resolving "
            f"{resolved_count} existing job card(s)."
        )
        result["auto_finished"] = True
        result["resolved_count"] = resolved_count
    return None


def _confirmed_job_results_return_action(
    context: WorkerExecutionContext,
    result: dict[str, Any],
) -> ActionRequest | None:
    no_effect_return = latest_no_effect_transition(context.state)
    followup_request, followup_trace = select_followup_action(
        {
            **context.state,
            "current_url": context.current_url,
            "current_page_role": "job_detail",
        },
        trigger_action="finish_detail_reading",
        trigger_page_role="job_detail",
        page_role="job_detail",
        current_url=context.current_url,
    )
    failed_return_action = str(no_effect_return.get("action") or "")
    if followup_request is not None:
        followup_name = str(followup_request.tool_calls[0].name)
        if failed_return_action != followup_name:
            result["detail_policy"] = "reuse_contextual_followup"
            result["followup_strategy"] = followup_trace
            return followup_request

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
    args: dict[str, Any],
    action_sequence: int,
) -> ActionRequest | None:
    target_count = target_count_from_state(context.state)
    collected_count = extracted_job_count(context.current_jobs)
    if args.get("detail_complete") is not True:
        return None

    (
        context.job_card_queue,
        context.active_job_card,
    ) = complete_active_job_card(
        context.job_card_queue,
        context.active_job_card,
    )
    result["detail_policy"] = "detail_complete"
    pending_cards = pending_job_cards(context.job_card_queue)
    resolved_count = max(
        collected_count,
        completed_job_card_count(context.job_card_queue),
    )
    if pending_cards or (
        target_count > 0 and resolved_count < target_count
    ):
        context.return_to_job_results = {
            "url": context.current_url,
            "reason": "detail_complete",
            "pending_count": len(pending_cards),
            "completed_action_seq": action_sequence,
        }
        return _confirmed_job_results_return_action(context, result)

    if (
        auto_finish_on_target_enabled()
        and count_mode_from_state(context.state) == "visible_all"
        and context.job_card_queue
        and not pending_cards
    ):
        context.is_finished = True
        context.collected_data.append(
            "Auto-finished after collecting all "
            f"{collected_count} visible jobs."
        )
        result["auto_finished"] = True
        result["count_mode"] = "visible_all"
        result["collected_count"] = collected_count
    return None


def _apply_collection_target_completion(
    context: WorkerExecutionContext,
    result: dict[str, Any],
    args: dict[str, Any],
) -> None:
    if (
        not auto_finish_on_target_enabled()
        or args.get("detail_complete") is False
    ):
        return

    target_count = target_count_from_state(context.state)
    collected_count = extracted_job_count(context.current_jobs)
    resolved_count = max(
        collected_count,
        completed_job_card_count(context.job_card_queue),
    )
    if target_count <= 0 or resolved_count < target_count:
        return

    context.return_to_job_results = {}
    context.is_finished = True
    context.collected_data.append(
        f"Auto-finished after collecting target_count={target_count} jobs."
    )
    result["auto_finished"] = True
    result["target_count"] = target_count
    result["collected_count"] = collected_count
    result["resolved_count"] = resolved_count


def _execute_state_action(
    context: WorkerExecutionContext,
    action_name: str,
    args: dict[str, Any],
    action_sequence: int,
) -> tuple[dict[str, Any], ActionRequest | None]:
    result, context.current_jobs = (
        worker_execution_dispatch.dispatch_state_action(
            action_name,
            args,
            context.current_jobs,
            current_url=context.current_url,
            state=context.state_for_dispatch(),
        )
    )
    follow_up: ActionRequest | None = None
    if action_name == "set_job_card_queue":
        follow_up = _apply_job_card_queue_result(context, result)

    if action_name == "finish_detail_reading":
        context.job_detail_buffer = dict(
            result.pop(
                "_job_detail_buffer",
                context.job_detail_buffer,
            )
            or {}
        )
        context.job_detail_followup = dict(
            result.pop(
                "_job_detail_followup",
                context.job_detail_followup,
            )
            or {}
        )

    is_successful_detail_update = (
        action_name in {"update_extracted_info", "finish_detail_reading"}
        and result.get("status") == "success"
        and is_detail_update(args)
    )
    if is_successful_detail_update:
        follow_up = (
            _apply_job_detail_completion(
                context,
                result,
                args,
                action_sequence,
            )
            or follow_up
        )

    if (
        action_name in {"update_extracted_info", "finish_detail_reading"}
        and result.get("status") == "success"
    ):
        _apply_collection_target_completion(context, result, args)
    return result, follow_up


def _record_successful_call(
    context: WorkerExecutionContext,
    *,
    action_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    before_snapshot: dict[str, Any],
    action_sequence: int,
    screen_changed: bool,
    tool_call_id: str,
    call_metadata: dict[str, Any],
) -> None:
    enriched = context.enrich_result(
        result,
        action_name,
        args,
        before_snapshot,
        screen_change_expected=screen_changed,
        tool_call_id=tool_call_id,
        tool_call_metadata=call_metadata,
    )
    context.new_actions.append(enriched)
    context.append_execution_record(
        action_name,
        args,
        enriched,
        before_snapshot,
        context.after_context(screen_changed=screen_changed),
        action_sequence,
        record_ui=action_name in UI_ACTIONS,
    )


def _record_failed_call(
    context: WorkerExecutionContext,
    *,
    action_name: str,
    args: dict[str, Any],
    error: Exception,
    action_sequence: int,
    tool_call_id: str,
    call_metadata: dict[str, Any],
) -> None:
    before_snapshot = context.before_snapshot()
    result = {
        "action": action_name,
        "status": "error",
        "error": str(error),
    }
    enriched = context.enrich_result(
        result,
        action_name,
        args,
        before_snapshot,
        tool_call_id=tool_call_id,
        tool_call_metadata=call_metadata,
    )
    context.new_actions.append(enriched)
    context.append_execution_record(
        action_name,
        args,
        enriched,
        before_snapshot,
        context.after_context(screen_changed=False),
        action_sequence,
    )
    context.error_count += 1


def execute_action_request(
    context: WorkerExecutionContext,
) -> WorkerExecutionContext:
    """검증된 요청의 원자 행동을 순서대로 실행한다."""

    for tool_call in context.action_request.tool_calls:
        action_name = tool_call.name
        args = dict(tool_call.args)
        call_metadata = dict(tool_call.metadata)
        action_context_args = {**args, **call_metadata}
        if action_name == "finish_detail_reading":
            args.setdefault("page_role", "job_detail")
            args.setdefault("detail_complete", True)

        logger.info(
            "Executing requested tool",
            source=context.action_request.source,
            action=action_name,
            args=compact_action_args(action_name, args),
        )
        step_started = time.perf_counter()
        before_snapshot = context.before_snapshot()
        action_sequence = context.next_action_sequence()
        follow_up: ActionRequest | None = None

        try:
            if _guard_return_to_results(
                context,
                action_name,
                args,
                before_snapshot,
                step_started,
            ):
                break

            if action_name in UI_ACTIONS:
                if _guard_ui_action(
                    context,
                    action_name,
                    args,
                    before_snapshot,
                    step_started,
                ):
                    break
                result, screen_changed = _execute_ui_action(
                    context,
                    action_name,
                    args,
                    call_metadata,
                    tool_call.id,
                    action_sequence,
                )
                _activate_clicked_job_card(
                    context,
                    result,
                    action_name,
                    action_context_args,
                )
            elif action_name in STATE_ACTIONS:
                result, follow_up = _execute_state_action(
                    context,
                    action_name,
                    args,
                    action_sequence,
                )
                screen_changed = False
            elif action_name == "finish_task":
                result = get_action_tools().finish_task(args["result"])
                context.is_finished = True
                context.collected_data.append(args["result"])
                screen_changed = False
            else:
                raise ValueError(f"Unknown tool: {action_name}")

            _record_successful_call(
                context,
                action_name=action_name,
                args=args,
                result=result,
                before_snapshot=before_snapshot,
                action_sequence=action_sequence,
                screen_changed=screen_changed,
                tool_call_id=tool_call.id,
                call_metadata=call_metadata,
            )
            logger.info(
                "Action execution completed",
                action=action_name,
                duration_sec=round(
                    time.perf_counter() - step_started,
                    6,
                ),
            )

            if follow_up is not None and not context.is_finished:
                context.next_pending_action = follow_up
                logger.info(
                    "Deterministic follow-up action queued",
                    source=follow_up.source,
                    reason=follow_up.summary,
                )
                break
            if context.is_finished:
                break
        except Exception as exc:
            logger.error(
                "Failed to execute action",
                action=action_name,
                error=str(exc),
            )
            _record_failed_call(
                context,
                action_name=action_name,
                args=args,
                error=exc,
                action_sequence=action_sequence,
                tool_call_id=tool_call.id,
                call_metadata=call_metadata,
            )
            break

    return context


__all__ = [
    "DIRECT_SCREEN_ACTION_SOURCES",
    "execute_action_request",
    "RETURN_ACTIONS",
    "STATE_ACTIONS",
    "UI_ACTIONS",
    "URL_STALE_ACTIONS",
]

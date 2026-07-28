"""검증, 실행, 상태 반영을 순서대로 조율하는 작업자 실행기."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.action_request import ActionRequest
from agent.graph.worker_action_effects import (
    activate_clicked_job_card,
    execute_state_action,
    execute_ui_action,
)
from agent.graph.worker_action_guard import (
    guard_return_to_results,
    guard_ui_action,
)
from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph.worker_execution_policy import compact_action_args
from agent.graph.worker_resources import get_action_tools
from agent.runtime.worker_actions import (
    STATE_UPDATE_ACTIONS,
    TERMINAL_ACTIONS,
    UI_ACTIONS,
)
from agent.utils.logger import logger


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
            if guard_return_to_results(
                context,
                action_name,
                args,
                before_snapshot,
                step_started,
            ):
                break

            if action_name in UI_ACTIONS:
                if guard_ui_action(
                    context,
                    action_name,
                    args,
                    before_snapshot,
                    step_started,
                ):
                    break
                result, screen_changed = execute_ui_action(
                    context,
                    action_name,
                    args,
                    call_metadata,
                    tool_call.id,
                    action_sequence,
                )
                activate_clicked_job_card(
                    context,
                    result,
                    action_name,
                    action_context_args,
                )
            elif action_name in STATE_UPDATE_ACTIONS:
                result, follow_up = execute_state_action(
                    context,
                    action_name,
                    args,
                    action_sequence,
                )
                screen_changed = False
            elif action_name in TERMINAL_ACTIONS:
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


__all__ = ["execute_action_request"]

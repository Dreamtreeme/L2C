"""물리·상태 행동의 실행 결과와 학습 근거를 같은 이벤트로 기록한다."""

from __future__ import annotations

from typing import Any

from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph.worker_execution_policy import compact_action_args
from agent.recipe.record import build_physical_action, build_screen_checkpoint
from agent.runtime.worker_contracts import build_action_event
from agent.utils.text import normalize_text
from agent.vision.target_snapshot import build_action_target_snapshot


def _enrich_action_result(
    context: WorkerExecutionContext,
    result: dict[str, Any],
    action_name: str,
    args: dict[str, Any],
    before_snapshot: dict[str, Any],
    *,
    screen_changed: bool,
    tool_call_id: str,
    tool_call_metadata: dict[str, Any] | None,
    action_source: str,
) -> dict[str, Any]:
    state = context.state
    action_request = context.action_request
    enriched = dict(result)
    enriched["args"] = compact_action_args(action_name, args)
    enriched["action_source"] = action_source or action_request.source
    if tool_call_id:
        enriched["tool_call_id"] = tool_call_id
    if tool_call_metadata:
        enriched["execution_metadata"] = dict(tool_call_metadata)
    enriched["before_url"] = before_snapshot.get("url", "")
    enriched["before_screenshot"] = before_snapshot.get("screenshot", "")
    enriched["before_marked_image"] = before_snapshot.get("marked_image", "")
    enriched["screen_change_expected"] = screen_changed
    target = build_action_target_snapshot(state, action_name, args)
    if target:
        enriched["target"] = target
    if action_request.source == "reflex":
        trace = state["replay"].get("reflex_trace") or {}
        if trace:
            enriched["reflex_recipe_key"] = trace.get("recipe_key", "")
            tool_calls = trace.get("tool_calls") or {}
            call_trace = tool_calls.get(tool_call_id) if tool_call_id else None
            if call_trace:
                enriched["reflex_match"] = dict(call_trace)
    if enriched.get("action") != action_name:
        enriched["requested_action"] = action_name
    return enriched


def record_action_result(
    context: WorkerExecutionContext,
    *,
    action_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    before_snapshot: dict[str, Any],
    action_sequence: int,
    screen_changed: bool = False,
    record_ui: bool = False,
    tool_call_id: str = "",
    tool_call_metadata: dict[str, Any] | None = None,
    action_source: str = "",
) -> dict[str, Any]:
    """행동 결과를 실행 로그와 레시피 학습 근거에 한 번만 기록한다."""

    enriched = _enrich_action_result(
        context,
        result,
        action_name,
        args,
        before_snapshot,
        screen_changed=screen_changed,
        tool_call_id=tool_call_id,
        tool_call_metadata=tool_call_metadata,
        action_source=action_source,
    )
    candidate_action = (
        build_physical_action(
            context.state,
            action_name,
            args,
            action_sequence,
        )
        if record_ui
        else None
    )
    before_checkpoint = (
        build_screen_checkpoint(
            context.state,
            observation_id=str(before_snapshot.get("observation_id") or ""),
            current_url=str(before_snapshot.get("url") or ""),
        )
        if candidate_action
        else None
    )
    before_marker_texts = [
        str(marker.get("text") or "")
        for marker in context.state["observation"].get("current_markers", []) or []
        if isinstance(marker, dict) and marker.get("text")
    ]
    context.new_actions.append(enriched)
    context.new_events.append(
        build_action_event(
            action_sequence,
            enriched,
            observation_id=context.action_request.observation_id,
            candidate_action=candidate_action,
            before_checkpoint=before_checkpoint,
            before_marker_texts=before_marker_texts,
            expected_after=normalize_text(args.get("expected_after")),
            intent=normalize_text(args.get("reason")),
        )
    )
    return enriched


__all__ = ["record_action_result"]

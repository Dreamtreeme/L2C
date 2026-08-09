"""물리·상태 행동의 실행 결과와 학습 근거를 같은 이벤트로 기록한다."""

from __future__ import annotations

from typing import Any

from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph.worker_execution_policy import compact_action_args
from agent.recipe.feedback import record_action_episode
from agent.recipe.record import record_ui_step
from agent.runtime.worker_contracts import build_action_event
from agent.runtime.job_collection import job_count
from agent.vision.target_snapshot import build_action_target_snapshot


def _after_context(
    context: WorkerExecutionContext,
    *,
    screen_changed: bool,
) -> dict[str, Any]:
    state = context.result.state
    observation = state["observation"]
    return {
        "current_url": str(observation.get("current_url") or ""),
        "current_url_stale": bool(observation.get("current_url_stale", True)),
        "screen_changed": screen_changed,
        "collected_job_count": job_count(
            state["collection"].get("collected_jobs", [])
        ),
        "is_finished": bool(state["lifecycle"].get("is_finished", False)),
    }


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
    state = context.result.state
    action_request = context.input.action_request
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
    enriched["decision_capture_id"] = str(
        action_request.metadata.get("decision_capture_id")
        or before_snapshot.get("capture_id")
        or ""
    )
    enriched["screen_change_expected"] = screen_changed
    target = build_action_target_snapshot(state, action_name, args)
    if target:
        enriched["target"] = target
    if action_request.source == "reflex":
        trace = dict(state["replay"].get("reflex_trace", {}) or {})
        if trace:
            enriched["reflex_recipe_key"] = trace.get("recipe_key", "")
            call_trace = (
                (trace.get("tool_calls") or {}).get(tool_call_id)
                if tool_call_id
                else None
            )
            if call_trace:
                enriched["reflex_match"] = dict(call_trace)
    if enriched.get("action") != action_name:
        enriched["requested_action"] = action_name
    return enriched


def _record_state(
    context: WorkerExecutionContext,
    before_snapshot: dict[str, Any],
) -> dict[str, Any]:
    state = context.result.state
    observation = state["observation"]
    return {
        "goal": state["request"].get("goal", ""),
        "current_capture_id": str(before_snapshot.get("capture_id") or ""),
        "current_markers": list(observation.get("current_markers", []) or []),
        "current_url": before_snapshot.get("url", ""),
        "current_page_role": observation.get("current_page_role", ""),
        "screen_signature": dict(observation.get("screen_signature", {}) or {}),
        "current_screenshot": str(observation.get("current_screenshot") or ""),
        "marked_image": observation.get("marked_image", ""),
    }


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
    record_state = _record_state(context, before_snapshot)
    recipe_steps: list[dict[str, Any]] = []
    if record_ui:
        record_ui_step(
            recipe_steps,
            record_state,
            action_name,
            args,
            action_sequence,
        )
    feedback: list[dict[str, Any]] = []
    record_action_episode(
        feedback,
        record_state,
        context.input.action_request,
        action_name,
        args,
        enriched,
        before_snapshot,
        _after_context(context, screen_changed=screen_changed),
        action_sequence,
    )
    context.result.new_actions.append(enriched)
    context.result.new_events.append(
        build_action_event(
            action_sequence,
            enriched,
            recipe_step=recipe_steps[0] if recipe_steps else None,
            feedback_episode=feedback[0] if feedback else None,
        )
    )
    return enriched


__all__ = ["record_action_result"]

"""행동 전후 화면을 연결하는 전환 요청을 기록한다."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.graph.worker_execution_policy import compact_action_args


def _transition_step(
    context: WorkerExecutionContext,
    action_sequence: int,
    action_name: str,
    args: dict[str, Any],
    tool_call_id: str,
) -> dict[str, Any]:
    state = context.state
    action_request = context.action_request
    step: dict[str, Any] = {
        "seq": action_sequence,
        "action": action_name,
        "args": compact_action_args(action_name, args),
        "page_role": (
            args.get("page_role") or state["observation"].get("current_page_role", "")
        ),
        "target_role": args.get("target_role") or "",
        "component": args.get("target_component") or "",
        "expected_after": args.get("expected_after") or "",
    }
    if tool_call_id:
        step["tool_call_id"] = tool_call_id
    if action_request.source == "reflex":
        trace = dict(state["replay"].get("reflex_trace", {}) or {})
        call_trace = (
            (trace.get("tool_calls") or {}).get(tool_call_id) if tool_call_id else None
        )
        if isinstance(call_trace, dict):
            step.update(
                {
                    "recipe_key": trace.get("recipe_key", ""),
                    "recipe_seq": call_trace.get("seq"),
                    "replay_mode": call_trace.get("replay_mode", ""),
                    "match_mode": call_trace.get("match_mode", ""),
                    "target_text": call_trace.get("target_text", ""),
                    "marker_id": call_trace.get("marker_id"),
                    "phash": call_trace.get("phash", {}),
                }
            )
    return {
        key: value for key, value in step.items() if value not in (None, "", {}, [])
    }


def set_transition_request(
    context: WorkerExecutionContext,
    action_sequence: int,
    action_name: str,
    args: dict[str, Any],
    source: str,
    tool_call_id: str = "",
) -> None:
    """다음 캡처가 검증할 화면 전환 기대값을 상태에 저장한다."""

    state = context.state
    action_request = context.action_request
    observation = state["observation"]
    recipe_key = ""
    if source == "reflex":
        recipe_key = str(
            (state["replay"].get("reflex_trace", {}) or {}).get("recipe_key") or ""
        )
    request_metadata = dict(action_request.metadata or {})
    before_state = (
        dict(request_metadata.get("before_state") or {})
        if isinstance(request_metadata.get("before_state"), dict)
        else {}
    )
    state["transition"]["transition_request"] = {
        "action_seq": action_sequence,
        "action": action_name,
        "before_observation_id": str(
            action_request.observation_id
            or observation.get("observation_id")
            or ""
        ),
        "source": source,
        "recipe_key": recipe_key,
        "recipe_transition_index": request_metadata.get("transition_index"),
        "recipe_transition_count": request_metadata.get("transition_count"),
        "expected_after_state": dict(
            request_metadata.get("expected_after_state") or {}
        ),
        "before_page_role": str(before_state.get("page_role") or ""),
        "transition_actions": list(request_metadata.get("transition_actions") or []),
        "step": _transition_step(
            context,
            action_sequence,
            action_name,
            args,
            tool_call_id,
        ),
        "before_url": str(observation.get("current_url") or ""),
        "before_screenshot": str(observation.get("current_screenshot") or ""),
        "started_at": time.time(),
    }


__all__ = ["set_transition_request"]

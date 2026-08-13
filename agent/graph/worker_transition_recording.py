"""행동 전후 화면을 연결하는 전환 요청을 기록한다."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.worker_execution_context import WorkerExecutionContext
from agent.runtime.worker_contracts import TransitionRequest
from shared.schema.recipe_schema import ScreenCheckpoint


def set_transition_request(
    context: WorkerExecutionContext,
    action_sequence: int,
    action_name: str,
    args: dict[str, Any],
    source: str,
) -> None:
    """같은 행동 요청의 실행 단계를 다음 캡처가 검증할 묶음으로 저장한다."""

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
    raw_expected_after_state = request_metadata.get("expected_after_state")
    expected_after_state = (
        ScreenCheckpoint.model_validate(raw_expected_after_state)
        if raw_expected_after_state
        else None
    )
    before_observation_id = str(
        action_request.observation_id or observation.get("observation_id") or ""
    )
    existing = state["transition"].get("transition_request")
    group = existing if (
        existing
        and existing.get("before_observation_id") == before_observation_id
        and existing.get("source") == source
    ) else None
    action_seqs = list(group.get("action_seqs") or []) if group else []
    if action_sequence not in action_seqs:
        action_seqs.append(action_sequence)
    recorded_actions = (
        list(group.get("transition_actions") or []) if group else []
    )
    metadata_actions = list(request_metadata.get("transition_actions") or [])
    if metadata_actions:
        recorded_actions = [str(name) for name in metadata_actions]
    elif action_name not in recorded_actions:
        recorded_actions.append(action_name)

    marker_id = args.get("marker_id")
    transition_request: TransitionRequest = {
        "action_seq": action_sequence,
        "action_seqs": action_seqs,
        "action": action_name,
        "before_observation_id": before_observation_id,
        "source": source,
        "recipe_key": recipe_key,
        "expected_after_state": expected_after_state,
        "expected_after": str(
            args.get("expected_after")
            or (group.get("expected_after") if group else "")
            or ""
        ),
        "input_text": str(
            args.get("text")
            or (group.get("input_text") if group else "")
            or ""
        ),
        "target_marker_id": (
            marker_id
            if isinstance(marker_id, int)
            else (
                group.get("target_marker_id") if group else None
            )
        ),
        "before_page_role": str(
            before_state.get("page_role")
            or observation.get("current_page_role")
            or ""
        ),
        "transition_actions": recorded_actions,
        "before_url": str(
            (group.get("before_url") if group else "")
            or observation.get("current_url")
            or ""
        ),
        "before_screenshot": str(
            (
                group.get("before_screenshot") if group else ""
            )
            or observation.get("current_screenshot")
            or ""
        ),
        "started_at": float(
            (group.get("started_at") if group else 0.0)
            or time.time()
        ),
    }
    transition_index = request_metadata.get("transition_index")
    if isinstance(transition_index, int):
        transition_request["recipe_transition_index"] = transition_index
    transition_count = request_metadata.get("transition_count")
    if isinstance(transition_count, int):
        transition_request["recipe_transition_count"] = transition_count
    state["transition"]["transition_request"] = transition_request


__all__ = ["set_transition_request"]

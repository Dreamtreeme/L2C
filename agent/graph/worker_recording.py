"""원자 행동 실행 결과를 피드백과 Reflex 후보 기록으로 변환한다."""

from __future__ import annotations

from typing import Any

from agent.graph.action_request import ActionRequest
from agent.graph.state import GraphState
from agent.recipe.feedback import record_action_episode
from agent.recipe.record import commit_if_finished, record_ui_step


def record_execution_node(state: GraphState) -> dict[str, Any]:
    """실행 노드가 남긴 결과만 읽어 학습용 기록을 생성한다."""

    records = [
        dict(item)
        for item in state.get("execution_records", []) or []
        if isinstance(item, dict)
    ]
    if not records:
        return {"execution_records": []}

    recorded_steps: list[dict[str, Any]] = []
    feedback_episodes: list[dict[str, Any]] = []
    for item in records:
        request = ActionRequest.model_validate(item.get("request") or {})
        action_name = str(item.get("action_name") or "")
        args = dict(item.get("args", {}) or {})
        result = dict(item.get("result", {}) or {})
        before_snapshot = dict(item.get("before_snapshot", {}) or {})
        after_context = dict(item.get("after_context", {}) or {})
        record_state = dict(item.get("record_state", {}) or {})
        seq = int(item.get("seq") or 0)
        if item.get("record_ui"):
            record_ui_step(recorded_steps, record_state, action_name, args, seq)
        record_action_episode(
            feedback_episodes,
            record_state,
            request,
            action_name,
            args,
            result,
            before_snapshot,
            after_context,
            seq,
        )

    if state.get("is_finished"):
        commit_if_finished(
            list(state.get("recorded_steps", []) or []) + recorded_steps,
            state,
            str(state.get("current_url") or ""),
        )
    return {
        "execution_records": [],
        "recorded_steps": recorded_steps,
        "feedback_episodes": feedback_episodes,
    }


__all__ = ["record_execution_node"]

"""향후 반사 레시피 승격을 위한 피드백 기록(feedback episode).

이 모듈은 스크립트를 강제하지 않고 관찰 결과만 남긴다.
반복 패턴의 재사용 여부는 이후 비평가/메모리(Critic/Memory) 계층이 판단한다.
"""

from __future__ import annotations

from typing import Any

from agent.runtime.worker_contracts import WorkerState
from agent.runtime.worker_actions import (
    STATE_UPDATE_ACTIONS,
    TERMINAL_ACTIONS,
    UI_ACTIONS,
)
from shared.schema.feedback_schema import (
    ActionFeedback,
    ActionObservation,
    ActionProposal,
    FeedbackEpisode,
)


def _feedback_label(
    action_name: str, result: dict[str, Any], after: dict[str, Any]
) -> ActionFeedback:
    status = result.get("status", "")
    reason = result.get("reason", "") or ""
    if status == "skipped":
        return ActionFeedback(label="no_effect", reason=reason or "skipped")
    if status == "error":
        return ActionFeedback(
            label="error",
            reason=str(result.get("error") or "action_error"),
        )

    if action_name == "finish_task":
        return ActionFeedback(label="success", reason="task finished")
    if (
        action_name == "open_browser"
        and isinstance(result.get("result"), dict)
        and result["result"].get("opened") is False
    ):
        return ActionFeedback(
            label="no_effect",
            reason=result["result"].get("reason", "browser already at target"),
        )
    if action_name in UI_ACTIONS:
        if after.get("screen_changed"):
            return ActionFeedback(
                label="partial",
                reason="screen-changing action executed; next perception must validate",
            )
        return ActionFeedback(
            label="no_effect",
            reason="no screen change expected or observed in action result",
        )
    if action_name in STATE_UPDATE_ACTIONS | TERMINAL_ACTIONS:
        return ActionFeedback(label="success", reason="state action executed")
    return ActionFeedback(
        label="partial", reason="unclassified successful action"
    )


def record_action_episode(
    episodes: list[FeedbackEpisode],
    state: WorkerState,
    action_name: str,
    args: dict[str, Any],
    enriched_result: dict[str, Any],
    before_snapshot: dict[str, Any],
    after_context: dict[str, Any],
    seq: int,
) -> None:
    """피드백 기록을 하나 추가한다."""

    observation_state = state["observation"]
    proposal = ActionProposal(
        action=action_name,
        args=dict(args),
    )
    before = {
        "observation_id": str(before_snapshot.get("observation_id") or ""),
        "url": before_snapshot.get("url", ""),
        "screenshot": before_snapshot.get("screenshot", ""),
        "page_role": str(observation_state.get("current_page_role") or ""),
        "marker_texts": [
            str(marker.get("text") or "")
            for marker in observation_state.get("current_markers", []) or []
            if isinstance(marker, dict) and marker.get("text")
        ],
    }
    after = {
        "url": after_context.get("current_url", ""),
        "screen_changed": bool(after_context.get("screen_changed", False)),
    }
    observation = ActionObservation(
        before=before,
        after=after,
        result=dict(enriched_result),
    )
    feedback = _feedback_label(action_name, enriched_result, after)
    episode = FeedbackEpisode(
        seq=seq,
        proposal=proposal,
        observation=observation,
        feedback=feedback,
    )
    episodes.append(episode)

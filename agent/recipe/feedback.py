"""향후 반사 레시피 승격을 위한 피드백 기록(feedback episode).

이 모듈은 스크립트를 강제하지 않고 관찰 결과만 남긴다.
반복 패턴의 재사용 여부는 이후 비평가/메모리(Critic/Memory) 계층이 판단한다.
"""

from __future__ import annotations

from typing import Any

from agent.utils.text import site_of
from agent.runtime.worker_contracts import ActionRequest
from agent.runtime.worker_actions import (
    STATE_UPDATE_ACTIONS,
    TERMINAL_ACTIONS,
    UI_ACTIONS,
)
from agent.utils.model_conversion import dump_model
from agent.vision.target_snapshot import build_action_target_snapshot
from shared.schema.feedback_schema import (
    ActionFeedback,
    ActionObservation,
    ActionProposal,
    FeedbackEpisode,
)

def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return "" if content is None else str(content).strip()


def _feedback_label(action_name: str, result: dict[str, Any], after: dict[str, Any]) -> ActionFeedback:
    status = result.get("status", "")
    reason = result.get("reason", "") or ""
    if status == "skipped":
        if reason == "same_state_repeat_blocked":
            return ActionFeedback(label="loop_risk", reason=reason, confidence=0.85)
        return ActionFeedback(label="no_effect", reason=reason or "skipped", confidence=0.7)
    if status == "error":
        return ActionFeedback(label="error", reason=str(result.get("error") or "action_error"), confidence=0.9)

    if action_name == "finish_task":
        return ActionFeedback(label="success", reason="task finished", confidence=0.8)
    if action_name == "open_browser" and isinstance(result.get("result"), dict) and result["result"].get("opened") is False:
        return ActionFeedback(label="no_effect", reason=result["result"].get("reason", "browser already at target"), confidence=0.75)
    if action_name in UI_ACTIONS:
        if after.get("screen_changed"):
            return ActionFeedback(label="partial", reason="screen-changing action executed; next perception must validate", confidence=0.45)
        return ActionFeedback(label="no_effect", reason="no screen change expected or observed in action result", confidence=0.45)
    if action_name in STATE_UPDATE_ACTIONS | TERMINAL_ACTIONS:
        return ActionFeedback(label="success", reason="state action executed", confidence=0.5)
    return ActionFeedback(label="partial", reason="unclassified successful action", confidence=0.2)


def record_action_episode(
    episodes: list[dict[str, Any]],
    state: dict,
    action_request: ActionRequest,
    action_name: str,
    args: dict[str, Any],
    enriched_result: dict[str, Any],
    before_snapshot: dict[str, Any],
    after_context: dict[str, Any],
    seq: int,
) -> None:
    """피드백 기록을 하나 추가한다."""

    goal = str(state.get("goal") or "")
    target = enriched_result.get("target") or build_action_target_snapshot(
        state,
        action_name,
        args,
    )
    proposal = ActionProposal(
        action=action_name,
        args=dict(args),
        llm_thought=_message_text(action_request.summary),
        reason=str(args.get("reason") or ""),
        target=target,
        target_label=args.get("target_label"),
        component_candidate=args.get("target_component"),
        target_role_candidate=args.get("target_role"),
        expected_after=str(args.get("expected_after") or ""),
    )
    before = {
        "capture_id": str(before_snapshot.get("capture_id") or ""),
        "url": before_snapshot.get("url", ""),
        "screenshot": before_snapshot.get("screenshot", ""),
        "marked_image": before_snapshot.get("marked_image", ""),
        "page_role": str(state.get("current_page_role") or ""),
        "screen_signature": dict(before_snapshot.get("screen_signature", {}) or {}),
        "marker_texts": [
            str(marker.get("text") or "")
            for marker in state.get("current_markers", []) or []
            if isinstance(marker, dict) and marker.get("text")
        ],
    }
    after = {
        "url": after_context.get("current_url", ""),
        "current_url_stale": bool(after_context.get("current_url_stale", True)),
        "screen_changed": bool(after_context.get("screen_changed", False)),
        "is_finished": bool(after_context.get("is_finished", False)),
        "collected_job_count": int(after_context.get("collected_job_count") or 0),
    }
    observation = ActionObservation(
        before=before,
        after=after,
        result=dict(enriched_result),
    )
    feedback = _feedback_label(action_name, enriched_result, after)
    episode = FeedbackEpisode(
        seq=seq,
        goal=goal,
        site=site_of(before.get("url", "")),
        proposal=proposal,
        observation=observation,
        feedback=feedback,
    )
    episodes.append(dump_model(episode))

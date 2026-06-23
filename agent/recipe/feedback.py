"""향후 반사 레시피 승격을 위한 피드백 기록(feedback episode).

이 모듈은 스크립트를 강제하지 않고 관찰 결과만 남긴다.
반복 패턴의 재사용 여부는 이후 비평가/메모리(Critic/Memory) 계층이 판단한다.
"""

from __future__ import annotations

import json
from typing import Any

from agent.recipe.state_key import normalize_text, site_of
from agent.utils.logger import logger
from shared.schema.feedback_schema import (
    ActionFeedback,
    ActionObservation,
    ActionProposal,
    FeedbackEpisode,
    ParameterCandidate,
)

_UI_ACTIONS = {"click_marker", "type_in_marker", "scroll", "press_key", "open_browser", "close_browser", "go_back"}
_STATE_ACTIONS = {"update_plan_progress", "update_extracted_info", "finish_task"}


def _dump_model(model) -> dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


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


def _compact_args(action_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if action_name == "update_extracted_info":
        raw = args.get("data_json", "")
        try:
            data = json.loads(raw or "{}")
        except Exception:
            return {"data_json": "<invalid json>", "payload_chars": len(raw)}
        jobs = data.get("공고목록")
        if isinstance(jobs, dict):
            jobs = [jobs]
        fields = []
        if isinstance(jobs, list):
            for job in jobs:
                if isinstance(job, dict):
                    fields.extend(job.keys())
        fields.extend(key for key in data.keys() if key != "공고목록")
        return {
            "incoming_jobs": len(jobs) if isinstance(jobs, list) else 0,
            "fields": sorted({str(field) for field in fields}),
            "payload_chars": len(raw),
        }
    return dict(args or {})


def _marker_by_id(markers: list[dict], marker_id: int | None) -> dict | None:
    for marker in markers or []:
        if isinstance(marker, dict) and marker.get("id") == marker_id:
            return marker
    return None


def _target_snapshot(state: dict, action_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    if action_name not in {"click_marker", "type_in_marker"}:
        return None
    marker = _marker_by_id(state.get("current_markers", []) or [], args.get("marker_id"))
    if not marker:
        return {"marker_id": args.get("marker_id"), "missing": True}
    target = {
        "marker_id": marker.get("id"),
        "text": marker.get("text", ""),
        "bbox": marker.get("bbox", []),
    }
    target_label = args.get("target_label") or args.get("semantic_label")
    if target_label:
        target["target_label"] = target_label
    return target


def _text_in_goal(value: str, goal: str) -> bool:
    value_key = normalize_text(value).lower().replace(" ", "")
    goal_key = normalize_text(goal).lower().replace(" ", "")
    return bool(value_key and goal_key and value_key in goal_key)


def _parameter_candidates(action_name: str, args: dict[str, Any], goal: str) -> list[ParameterCandidate]:
    candidates: list[ParameterCandidate] = []
    if action_name == "type_in_marker":
        value = str(args.get("text") or "").strip()
        if value:
            slot_name = str(args.get("slot_name") or "").strip()
            candidates.append(
                ParameterCandidate(
                    slot_candidate=slot_name or ("query" if _text_in_goal(value, goal) else "text_input"),
                    value=value,
                    reason=str(args.get("reason") or "")
                    or ("typed text overlaps user goal" if _text_in_goal(value, goal) else "typed text may vary by run"),
                    confidence=0.65 if slot_name else (0.45 if _text_in_goal(value, goal) else 0.25),
                )
            )
    return candidates


def _extracted_job_count(extracted_jd: Any) -> int:
    if not isinstance(extracted_jd, dict):
        return 0
    jobs = extracted_jd.get("공고목록")
    if isinstance(jobs, list):
        return len(jobs)
    return 1 if extracted_jd else 0


def _feedback_label(action_name: str, result: dict[str, Any], after: dict[str, Any]) -> ActionFeedback:
    status = result.get("status", "")
    reason = result.get("reason", "") or ""
    if status == "skipped":
        if reason in {"same_state_repeat_blocked", "unsafe_ui_action_chain", "chain_boundary_after_screen_change"}:
            return ActionFeedback(label="loop_risk", reason=reason, confidence=0.85)
        return ActionFeedback(label="no_effect", reason=reason or "skipped", confidence=0.7)
    if status == "error":
        return ActionFeedback(label="error", reason=str(result.get("error") or "action_error"), confidence=0.9)

    if action_name == "update_extracted_info":
        if after.get("extracted_job_count", 0) > 0:
            return ActionFeedback(label="success", reason="extracted data changed", confidence=0.65)
        return ActionFeedback(label="partial", reason="state update executed but no job count observed", confidence=0.35)
    if action_name == "finish_task":
        return ActionFeedback(label="success", reason="task finished", confidence=0.8)
    if action_name == "open_browser" and isinstance(result.get("result"), dict) and result["result"].get("opened") is False:
        return ActionFeedback(label="no_effect", reason=result["result"].get("reason", "browser already at target"), confidence=0.75)
    if action_name in _UI_ACTIONS:
        if after.get("screen_changed"):
            return ActionFeedback(label="partial", reason="screen-changing action executed; next perception must validate", confidence=0.45)
        return ActionFeedback(label="no_effect", reason="no screen change expected or observed in action result", confidence=0.45)
    if action_name in _STATE_ACTIONS:
        return ActionFeedback(label="success", reason="state action executed", confidence=0.5)
    return ActionFeedback(label="partial", reason="unclassified successful action", confidence=0.2)


def record_action_episode(
    episodes: list[dict[str, Any]],
    state: dict,
    ai_msg: Any,
    action_name: str,
    args: dict[str, Any],
    enriched_result: dict[str, Any],
    before_snapshot: dict[str, Any],
    after_context: dict[str, Any],
    seq: int,
) -> None:
    """피드백 기록(feedback episode)을 하나 추가한다. 실패해도 실행을 막지 않는다."""
    try:
        goal = state.get("goal", "") or ""
        target = enriched_result.get("target") or _target_snapshot(state, action_name, args)
        proposal = ActionProposal(
            action=action_name,
            args=_compact_args(action_name, args),
            llm_thought=_message_text(getattr(ai_msg, "content", "")),
            reason=str(args.get("reason") or ""),
            target=target,
            target_label=(args.get("target_label") or args.get("semantic_label")),
            component_candidate=args.get("target_component") or args.get("component_candidate"),
            target_role_candidate=args.get("target_role") or args.get("target_role_candidate"),
            expected_after=str(args.get("expected_after") or ""),
            parameter_candidates=_parameter_candidates(action_name, args, goal),
            fixed_candidate=action_name in {"scroll", "press_key", "go_back"},
        )
        before = {
            "state_key": before_snapshot.get("state_key", ""),
            "url": before_snapshot.get("url", ""),
            "screenshot": before_snapshot.get("screenshot", ""),
            "marked_image": before_snapshot.get("marked_image", ""),
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
            "extracted_job_count": _extracted_job_count(after_context.get("extracted_jd", {})),
        }
        observation = ActionObservation(before=before, after=after, result=dict(enriched_result))
        feedback = _feedback_label(action_name, enriched_result, after)
        episode = FeedbackEpisode(
            seq=seq,
            goal=goal,
            site=site_of(before.get("url", "")),
            page_state_key=before.get("state_key", ""),
            proposal=proposal,
            observation=observation,
            feedback=feedback,
        )
        episodes.append(_dump_model(episode))
    except Exception as e:
        logger.debug("feedback record_action_episode skipped", error=str(e))

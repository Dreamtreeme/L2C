"""활성 Reflex 승격 전에 명백한 실행 결함을 차단한다."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from agent.runtime.replay_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
    is_supported_recipe_action_group,
)
_BLOCKING_FEEDBACK_LABELS = {"wrong_target", "no_effect", "loop_risk", "error"}
_BLOCKING_RESULT_STATUSES = {"error", "skipped"}
_BLOCKING_TRANSITION_REASONS = {
    "no_screen_change",
    "reflex_no_screen_change",
}
_CODE_MANAGED_TRANSITION_SOURCES = {
    "page_policy": "managed_by_page_policy",
    "job_card_queue": "managed_by_card_queue",
    "duplicate_job_policy": "managed_by_duplicate_policy",
    "screen_policy": "managed_by_screen_policy",
    "reflex": "already_managed_by_reflex",
}
_DEFERRED_GROUP_EFFECT_REASONS = {
    "no_screen_change",
    "transition_not_ready",
}


def _seq(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _step_before_capture_id(step: dict[str, Any]) -> str:
    before_state = (
        step.get("before_state")
        if isinstance(step.get("before_state"), dict)
        else {}
    )
    return str(
        before_state.get("capture_id")
        or step.get("decision_capture_id")
        or ""
    )


def _transition_after_capture_id(
    observations: list[dict[str, Any]],
) -> str:
    for observation in reversed(observations):
        after_state = (
            observation.get("after_state")
            if isinstance(observation.get("after_state"), dict)
            else {}
        )
        capture_id = str(
            after_state.get("capture_id")
            or observation.get("to_capture_id")
            or ""
        )
        if capture_id:
            return capture_id
    return ""


def _apply_verified_action_groups(
    candidate: dict[str, Any],
    transitions_by_seq: dict[int, list[dict[str, Any]]],
    verdicts: dict[int, dict[str, Any]],
) -> None:
    """최종 행동이 전환을 검증한 연속 행동 묶음의 선행 무변화를 허용한다."""

    reviewable_steps = [
        step
        for step in candidate.get("steps", []) or []
        if (
            isinstance(step, dict)
            and step.get("action") in REVIEWABLE_REPLAY_ACTIONS
            and _seq(step.get("seq")) is not None
        )
    ]
    for first, second in zip(reviewable_steps, reviewable_steps[1:]):
        if not is_supported_recipe_action_group([first, second]):
            continue
        first_seq = int(first["seq"])
        second_seq = int(second["seq"])
        first_verdict = verdicts.get(first_seq)
        second_verdict = verdicts.get(second_seq)
        if not first_verdict or not second_verdict or not second_verdict["eligible"]:
            continue

        first_after_capture = _transition_after_capture_id(
            transitions_by_seq.get(first_seq, [])
        )
        second_before_capture = _step_before_capture_id(second)
        if (
            not first_after_capture
            or not second_before_capture
            or first_after_capture != second_before_capture
        ):
            continue

        blocking_reasons = list(first_verdict.get("blocking_reasons") or [])
        remaining_reasons = [
            reason
            for reason in blocking_reasons
            if reason not in _DEFERRED_GROUP_EFFECT_REASONS
        ]
        if remaining_reasons:
            continue

        group_seqs = [first_seq, second_seq]
        first_verdict.update(
            {
                "eligible": True,
                "blocking_reasons": [],
                "execution_group_seqs": group_seqs,
                "effect_verified_by_seq": second_seq,
                "evidence_mode": "deferred_group_effect",
            }
        )
        second_verdict.update(
            {
                "execution_group_seqs": group_seqs,
                "effect_verified_by_seq": second_seq,
                "evidence_mode": "group_commit_effect",
            }
        )


def evaluate_candidate_step_evidence(candidate: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """재사용 가능한 행동별 증거를 검사해 명백한 차단 사유를 반환한다."""

    payload = dict(candidate.get("payload", {}) or {})
    feedback_by_seq: dict[int, list[dict[str, Any]]] = defaultdict(list)
    transitions_by_seq: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for episode in payload.get("feedback_episodes", []) or []:
        if not isinstance(episode, dict):
            continue
        seq = _seq(episode.get("seq"))
        if seq is not None:
            feedback_by_seq[seq].append(episode)

    for observation in payload.get("transition_records", []) or []:
        if not isinstance(observation, dict):
            continue
        seq = _seq(observation.get("action_seq"))
        if seq is not None:
            transitions_by_seq[seq].append(observation)

    verdicts: dict[int, dict[str, Any]] = {}
    for step in candidate.get("steps", []) or []:
        if (
            not isinstance(step, dict)
            or step.get("action") not in REVIEWABLE_REPLAY_ACTIONS
        ):
            continue
        seq = _seq(step.get("seq"))
        if seq is None:
            continue

        feedback_items = feedback_by_seq.get(seq, [])
        transition_items = transitions_by_seq.get(seq, [])
        feedback_labels: list[str] = []
        transition_sources: list[str] = []
        transition_statuses: list[str] = []
        reasons: list[str] = []
        if str(step.get("risk_level") or "").strip().casefold() == "sensitive":
            reasons.append("sensitive_action")
        if step.get("needs_user_confirmation") is True:
            reasons.append("user_confirmation_required")

        for episode in feedback_items:
            feedback = episode.get("feedback") if isinstance(episode.get("feedback"), dict) else {}
            observation = episode.get("observation") if isinstance(episode.get("observation"), dict) else {}
            result = observation.get("result") if isinstance(observation.get("result"), dict) else {}
            label = str(feedback.get("label") or "").strip()
            result_status = str(result.get("status") or "").strip()
            if label:
                feedback_labels.append(label)
            if label in _BLOCKING_FEEDBACK_LABELS:
                reasons.append(f"feedback_{label}")
            if result_status in _BLOCKING_RESULT_STATUSES:
                reasons.append(f"action_{result_status}")

        for observation in transition_items:
            source = str(observation.get("source") or "").strip()
            status = str(observation.get("status") or "").strip()
            reason = str(observation.get("reason") or "").strip()
            if source:
                transition_sources.append(source)
            if status:
                transition_statuses.append(status)
            managed_reason = _CODE_MANAGED_TRANSITION_SOURCES.get(source)
            if managed_reason:
                reasons.append(managed_reason)
            if reason in _BLOCKING_TRANSITION_REASONS:
                reasons.append(reason)

        if step.get("action") in CONTEXTUAL_REPLAY_ACTIONS:
            if not feedback_items:
                reasons.append("action_evidence_missing")
            if not transition_items:
                reasons.append("transition_evidence_missing")
            elif "ready" not in transition_statuses:
                reasons.append("transition_not_ready")
        elif not feedback_items and not transition_items:
            reasons.append("action_evidence_missing")

        verdicts[seq] = {
            "seq": seq,
            "action": str(step.get("action") or ""),
            "eligible": not reasons,
            "blocking_reasons": list(dict.fromkeys(reasons)),
            "feedback_labels": list(dict.fromkeys(feedback_labels)),
            "transition_sources": list(dict.fromkeys(transition_sources)),
            "transition_statuses": list(dict.fromkeys(transition_statuses)),
        }
    _apply_verified_action_groups(
        candidate,
        transitions_by_seq,
        verdicts,
    )
    return verdicts


def compact_step_evidence_verdicts(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Critic 입력과 검증 로그에 사용할 정렬된 판정 목록을 만든다."""

    verdicts = evaluate_candidate_step_evidence(candidate)
    return [verdicts[seq] for seq in sorted(verdicts)]


__all__ = [
    "compact_step_evidence_verdicts",
    "evaluate_candidate_step_evidence",
]

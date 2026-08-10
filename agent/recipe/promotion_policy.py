"""활성 Reflex 승격 전에 명백한 실행 결함을 차단한다."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from agent.runtime.worker_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
    is_supported_recipe_action_group,
)
from shared.schema.feedback_schema import (
    FeedbackEpisode,
    RecipeCandidate,
    RecordedRecipeStep,
    RecordedTransition,
)

_BLOCKING_FEEDBACK_LABELS = {"no_effect", "error"}
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


def _step_before_observation_id(step: RecordedRecipeStep) -> str:
    return str(step.before_state.get("observation_id") or "")


def _transition_after_observation_id(
    observations: list[RecordedTransition],
) -> str:
    for observation in reversed(observations):
        observation_id = str(
            observation.after_state.get("observation_id") or ""
        )
        if observation_id:
            return observation_id
    return ""


def _apply_verified_action_groups(
    candidate: RecipeCandidate,
    transitions_by_seq: dict[int, list[RecordedTransition]],
    verdicts: dict[int, dict[str, Any]],
) -> None:
    """최종 행동이 전환을 검증한 연속 행동 묶음의 선행 무변화를 허용한다."""

    reviewable_steps = [
        step
        for step in candidate.steps
        if (step.action in REVIEWABLE_REPLAY_ACTIONS and step.seq is not None)
    ]
    for first, second in zip(reviewable_steps, reviewable_steps[1:]):
        if not is_supported_recipe_action_group(
            [first.model_dump(mode="json"), second.model_dump(mode="json")]
        ):
            continue
        first_seq = int(first.seq)
        second_seq = int(second.seq)
        first_verdict = verdicts.get(first_seq)
        second_verdict = verdicts.get(second_seq)
        if not first_verdict or not second_verdict or not second_verdict["eligible"]:
            continue

        first_after_observation = _transition_after_observation_id(
            transitions_by_seq.get(first_seq, [])
        )
        second_before_observation = _step_before_observation_id(second)
        if (
            not first_after_observation
            or not second_before_observation
            or first_after_observation != second_before_observation
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


def _feedback_evidence(
    feedback_items: list[FeedbackEpisode],
) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    reasons: list[str] = []
    for episode in feedback_items:
        result = episode.observation.result
        label = episode.feedback.label
        result_status = str(result.get("status") or "").strip()
        if label:
            labels.append(label)
        if label in _BLOCKING_FEEDBACK_LABELS:
            reasons.append(f"feedback_{label}")
        if result_status in _BLOCKING_RESULT_STATUSES:
            reasons.append(f"action_{result_status}")
    return labels, reasons


def _transition_evidence(
    transition_items: list[RecordedTransition],
) -> tuple[list[str], list[str], list[str]]:
    sources: list[str] = []
    statuses: list[str] = []
    reasons: list[str] = []
    for observation in transition_items:
        source = observation.source.strip()
        status = observation.status.strip()
        reason = observation.reason.strip()
        if source:
            sources.append(source)
        if status:
            statuses.append(status)
        managed_reason = _CODE_MANAGED_TRANSITION_SOURCES.get(source)
        if managed_reason:
            reasons.append(managed_reason)
        if reason in _BLOCKING_TRANSITION_REASONS:
            reasons.append(reason)
    return sources, statuses, reasons


def _step_policy_reasons(step: RecordedRecipeStep) -> list[str]:
    reasons = []
    if step.risk_level.strip().casefold() == "sensitive":
        reasons.append("sensitive_action")
    return reasons


def _missing_evidence_reasons(
    action: str,
    feedback_items: list[FeedbackEpisode],
    transition_items: list[RecordedTransition],
    transition_statuses: list[str],
) -> list[str]:
    if action not in CONTEXTUAL_REPLAY_ACTIONS:
        return (
            ["action_evidence_missing"]
            if not feedback_items and not transition_items
            else []
        )
    reasons = []
    if not feedback_items:
        reasons.append("action_evidence_missing")
    if not transition_items:
        reasons.append("transition_evidence_missing")
    elif "ready" not in transition_statuses:
        reasons.append("transition_not_ready")
    return reasons


def _step_evidence_verdict(
    step: RecordedRecipeStep,
    seq: int,
    feedback_items: list[FeedbackEpisode],
    transition_items: list[RecordedTransition],
) -> dict[str, Any]:
    action = step.action
    feedback_labels, feedback_reasons = _feedback_evidence(feedback_items)
    transition_sources, transition_statuses, transition_reasons = _transition_evidence(
        transition_items
    )
    reasons = _step_policy_reasons(step)
    reasons.extend(feedback_reasons)
    reasons.extend(transition_reasons)
    reasons.extend(
        _missing_evidence_reasons(
            action,
            feedback_items,
            transition_items,
            transition_statuses,
        )
    )
    return {
        "seq": seq,
        "action": action,
        "eligible": not reasons,
        "blocking_reasons": list(dict.fromkeys(reasons)),
        "feedback_labels": list(dict.fromkeys(feedback_labels)),
        "transition_sources": list(dict.fromkeys(transition_sources)),
        "transition_statuses": list(dict.fromkeys(transition_statuses)),
    }


def evaluate_candidate_step_evidence(
    candidate: RecipeCandidate,
) -> dict[int, dict[str, Any]]:
    """재사용 가능한 행동별 증거를 검사해 명백한 차단 사유를 반환한다."""

    feedback_by_seq: dict[int, list[FeedbackEpisode]] = defaultdict(list)
    for episode in candidate.submission.feedback_episodes:
        feedback_by_seq[episode.seq].append(episode)
    transitions_by_seq: dict[int, list[RecordedTransition]] = defaultdict(list)
    for transition in candidate.submission.transition_records:
        if transition.action_seq is not None:
            transitions_by_seq[transition.action_seq].append(transition)
    verdicts: dict[int, dict[str, Any]] = {}
    reviewable_steps = (
        step for step in candidate.steps if step.action in REVIEWABLE_REPLAY_ACTIONS
    )
    for step in reviewable_steps:
        seq = step.seq
        if seq is None:
            continue
        verdicts[seq] = _step_evidence_verdict(
            step,
            seq,
            feedback_by_seq.get(seq, []),
            transitions_by_seq.get(seq, []),
        )
    _apply_verified_action_groups(
        candidate,
        transitions_by_seq,
        verdicts,
    )
    return verdicts


def compact_step_evidence_verdicts(candidate: RecipeCandidate) -> list[dict[str, Any]]:
    """Critic 입력과 검증 로그에 사용할 정렬된 판정 목록을 만든다."""

    verdicts = evaluate_candidate_step_evidence(candidate)
    return [verdicts[seq] for seq in sorted(verdicts)]


__all__ = [
    "compact_step_evidence_verdicts",
    "evaluate_candidate_step_evidence",
]

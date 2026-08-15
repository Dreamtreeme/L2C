"""자율탐색 원본 기록에서 비평가가 남긴 연속 성공 경로를 고른다."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from agent.runtime.site_context import normalize_page_role
from agent.runtime.target_matching import screen_context_signature_match
from agent.runtime.worker_actions import is_supported_recipe_action_group
from shared.schema.execution_record_schema import (
    ObservedTransition,
    PhysicalActionName,
    ScreenCheckpoint,
)
from shared.schema.feedback_schema import RecipeCandidate, RecipeCandidateReview


class PrunedTransition(BaseModel):
    """경험 규칙 생성 대상에서 제외된 전이와 이유."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    actions: list[PhysicalActionName]
    reason: str


def _has_verifiable_result(transition: ObservedTransition) -> bool:
    before = transition.before
    after = transition.after
    if after.has_anchor() or after.has_context_phash():
        return True
    before_role = normalize_page_role(before.page_role)
    after_role = normalize_page_role(after.page_role)
    if before_role and after_role and before_role != after_role:
        return True
    return bool(after.url_template and after.url_template != before.url_template)


def _has_target_evidence(transition: ObservedTransition) -> bool:
    return all(
        action.action not in {"click_marker", "type_in_marker", "scroll"}
        or (action.target is not None and bool(action.roi_signature))
        for action in transition.actions
    )


def is_preparation_transition(
    transition: ObservedTransition,
    following: ObservedTransition | None,
) -> bool:
    """화면은 유지됐지만 바로 다음 성공 행동에 입력을 제공했는지 확인한다."""

    evidence = transition.evidence
    following_evidence = following.evidence if following else None
    return bool(
        evidence
        and evidence.source == "autonomous"
        and evidence.result_status == "success"
        and evidence.status == "unknown"
        and evidence.reason == "no_screen_change"
        and len(transition.actions) == 1
        and transition.actions[0].action == "type_in_marker"
        and _has_target_evidence(transition)
        and following
        and following_evidence
        and transition.after.same_observation_as(following.before)
        and following_evidence.source == "autonomous"
        and following_evidence.result_status == "success"
        and following_evidence.status == "ready"
        and is_supported_recipe_action_group(following.actions)
        and not any(
            action.risk_level.strip().casefold() == "sensitive"
            for action in following.actions
        )
        and _has_target_evidence(following)
        and _has_verifiable_result(following)
    )


def transition_rejection_reason(
    transition: ObservedTransition,
    following: ObservedTransition | None = None,
) -> str:
    """비평가에게 전달할 수 없는 원본 전이의 첫 번째 이유를 반환한다."""

    evidence = transition.evidence
    if evidence is None:
        return "transition_evidence_missing"
    if evidence.source != "autonomous":
        return "not_autonomous"
    if evidence.result_status != "success":
        return f"action_{evidence.result_status or 'result_missing'}"
    if evidence.status != "ready" and not is_preparation_transition(
        transition,
        following,
    ):
        return "transition_not_ready"
    if not normalize_page_role(transition.before.page_role):
        return "page_role_missing"
    if any(
        action.risk_level.strip().casefold() == "sensitive"
        for action in transition.actions
    ):
        return "sensitive_action"
    if not is_supported_recipe_action_group(transition.actions):
        return "unsupported_action_group"
    if not _has_target_evidence(transition):
        return "target_evidence_missing"
    if not _has_verifiable_result(transition):
        return "after_state_unverifiable"
    return ""


def reviewable_candidate_transitions(
    candidate: RecipeCandidate,
) -> tuple[list[ObservedTransition], list[PrunedTransition]]:
    """원본 근거가 완성된 자율탐색 전이만 비평가 입력으로 고른다."""

    reviewable: list[ObservedTransition] = []
    pruned: list[PrunedTransition] = []
    transitions = candidate.transitions
    for index, transition in enumerate(transitions):
        following = transitions[index + 1] if index + 1 < len(transitions) else None
        reason = transition_rejection_reason(transition, following)
        if reason:
            pruned.append(
                PrunedTransition(
                    seq=transition.seq,
                    actions=[action.action for action in transition.actions],
                    reason=reason,
                )
            )
            continue
        reviewable.append(transition)
    return reviewable, pruned


def _same_screen(left: ScreenCheckpoint, right: ScreenCheckpoint) -> bool:
    if left.same_observation_as(right):
        return True
    if normalize_page_role(left.page_role) != normalize_page_role(right.page_role):
        return False
    if left.url_template != right.url_template:
        return False
    match = screen_context_signature_match(
        dict(left.screen_context_signature),
        dict(right.screen_context_signature),
    )
    distance = match.get("distance")
    return bool(match.get("matched") and isinstance(distance, int) and distance == 0)


def transitions_are_continuous(
    previous: ObservedTransition,
    current: ObservedTransition,
) -> bool:
    """앞 전이의 도착 화면과 다음 전이의 시작 화면이 같은지 판정한다."""

    return _same_screen(previous.after, current.before)


def continuous_transition_groups(
    transitions: list[ObservedTransition],
) -> list[list[ObservedTransition]]:
    """전이를 같은 도착·시작 화면이 이어지는 경로로 묶는다."""

    if not transitions:
        return []
    groups: list[list[ObservedTransition]] = []
    current_group = [transitions[0]]
    for transition in transitions[1:]:
        if transitions_are_continuous(current_group[-1], transition):
            current_group.append(transition)
            continue
        groups.append(current_group)
        current_group = [transition]
    groups.append(current_group)
    return groups


def retained_candidate_path(
    candidate: RecipeCandidate,
    review: RecipeCandidateReview,
) -> tuple[list[ObservedTransition], list[PrunedTransition]]:
    """비평가가 남긴 전이가 하나의 연속 경로일 때 원본 그대로 반환한다."""

    reviewable, pruned = reviewable_candidate_transitions(candidate)
    kept_seqs = {
        verdict.seq for verdict in review.transition_verdicts if verdict.keep
    }
    retained: list[ObservedTransition] = []
    for transition in reviewable:
        if transition.seq in kept_seqs:
            retained.append(transition)
        else:
            pruned.append(
                PrunedTransition(
                    seq=transition.seq,
                    actions=[action.action for action in transition.actions],
                    reason="critic_pruned",
                )
            )
    if len(continuous_transition_groups(retained)) != 1:
        return [], pruned
    return retained, pruned


__all__ = [
    "PrunedTransition",
    "continuous_transition_groups",
    "is_preparation_transition",
    "retained_candidate_path",
    "reviewable_candidate_transitions",
    "transition_rejection_reason",
    "transitions_are_continuous",
]

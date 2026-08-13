"""비평가가 남긴 자율탐색 전이를 활성 경험 경로로 승격한다."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent.recipe.store import RecipeStore
from agent.recipe.task_category import normalize_task_category
from agent.runtime.site_context import normalize_page_role
from agent.runtime.target_matching import screen_context_signature_match
from agent.runtime.worker_actions import is_supported_recipe_action_group
from shared.schema.feedback_schema import RecipeCandidate, RecipeCandidateReview
from shared.schema.recipe_schema import (
    ExperiencePath,
    ExperienceTransition,
    PhysicalActionName,
    ScreenCheckpoint,
)
from shared.schema.skill_schema import RECIPE_INPUT_NAMES, RecipeSkillMetadata


class PrunedTransition(BaseModel):
    """승격 대상에서 제외된 전이와 이유."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    actions: list[PhysicalActionName]
    reason: str


class CandidatePromotionResult(BaseModel):
    """후보 하나를 활성 경험 경로로 승격한 결과."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    promoted: bool
    saved_count: int
    promoted_action_count: int
    promoted_transition_count: int
    promoted_path_count: int
    pruned_transitions: list[PrunedTransition] = Field(default_factory=list)


def _has_verifiable_result(transition: ExperienceTransition) -> bool:
    before = transition.before
    after = transition.after
    if after.has_anchor() or after.has_context_phash():
        return True
    before_role = normalize_page_role(before.page_role)
    after_role = normalize_page_role(after.page_role)
    if before_role and after_role and before_role != after_role:
        return True
    return bool(after.url_template and after.url_template != before.url_template)


def is_preparation_transition(
    transition: ExperienceTransition,
    following: ExperienceTransition | None,
) -> bool:
    """화면은 유지됐지만 바로 다음 성공 행동에 입력을 제공한 전이인지 확인한다."""

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
        and transition.actions[0].is_supported_replay_action()
        and following
        and following_evidence
        and transition.after.same_observation_as(following.before)
        and following_evidence.source == "autonomous"
        and following_evidence.result_status == "success"
        and following_evidence.status == "ready"
        and is_supported_recipe_action_group(following.actions)
        and all(action.is_supported_replay_action() for action in following.actions)
        and not any(
            action.risk_level.strip().casefold() == "sensitive"
            for action in following.actions
        )
        and _has_verifiable_result(following)
    )


def transition_rejection_reason(
    transition: ExperienceTransition,
    following: ExperienceTransition | None = None,
) -> str:
    """기록된 전이를 그대로 재생할 수 없는 첫 번째 이유를 반환한다."""

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
    if not all(action.is_supported_replay_action() for action in transition.actions):
        return "unsupported_replay_action"
    if not _has_verifiable_result(transition):
        return "after_state_unverifiable"
    return ""


def reviewable_candidate_transitions(
    candidate: RecipeCandidate,
) -> tuple[list[ExperienceTransition], list[PrunedTransition]]:
    """실행 계약이 완성된 자율탐색 전이만 비평가 입력으로 고른다."""

    reviewable: list[ExperienceTransition] = []
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
    left_role = normalize_page_role(left.page_role)
    right_role = normalize_page_role(right.page_role)
    if left_role != right_role:
        return False
    if left.url_template != right.url_template:
        return False
    match = screen_context_signature_match(
        dict(left.screen_context_signature),
        dict(right.screen_context_signature),
    )
    distance = match.get("distance")
    return bool(
        match.get("matched")
        and isinstance(distance, int)
        and distance == 0
    )


def _retained_path(
    transitions: list[ExperienceTransition],
) -> ExperiencePath | None:
    """남은 전이가 하나의 연속된 성공 경로일 때만 활성 경로로 만든다."""

    if not transitions:
        return None
    for previous, current in zip(transitions, transitions[1:]):
        if not _same_screen(previous.after, current.before):
            return None
    return ExperiencePath(
        transitions=[
            transition.model_copy(deep=True, update={"seq": index})
            for index, transition in enumerate(transitions)
        ]
    )


def _candidate_skill_metadata(
    candidate: RecipeCandidate,
    transitions: list[ExperienceTransition],
) -> RecipeSkillMetadata:
    slots = {
        name
        for transition in transitions
        for action in transition.actions
        for name in action.slot_refs
        if name in RECIPE_INPUT_NAMES
    }
    return RecipeSkillMetadata(
        task_category=normalize_task_category(
            candidate.collection_intent.task_category
        ),
        inputs=[{"name": name} for name in sorted(slots)],
    )


def apply_candidate_promotion(
    candidate: RecipeCandidate,
    review: RecipeCandidateReview,
    db_path=None,
) -> CandidatePromotionResult:
    """비평가가 유지한 전이를 변경하지 않고 활성 경로로 저장한다."""

    reviewable, pruned = reviewable_candidate_transitions(candidate)
    kept_seqs = {
        verdict.seq for verdict in review.transition_verdicts if verdict.keep
    }
    retained: list[ExperienceTransition] = []
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

    path = _retained_path(retained)
    paths = [path] if path else []
    if retained and path is None:
        disconnected = retained[-1]
        pruned.append(
            PrunedTransition(
                seq=disconnected.seq,
                actions=[action.action for action in disconnected.actions],
                reason="path_disconnected_after_pruning",
            )
        )
    metadata = _candidate_skill_metadata(candidate, retained)
    saved_count = RecipeStore(db_path).replace_recipe_paths(
        candidate.site,
        candidate.goal,
        paths,
        metadata=metadata,
        source_run_id=candidate.run_id,
    )
    return CandidatePromotionResult(
        promoted=saved_count > 0,
        saved_count=saved_count,
        promoted_action_count=(
            sum(len(transition.actions) for transition in retained) if path else 0
        ),
        promoted_transition_count=len(retained) if path else 0,
        promoted_path_count=saved_count,
        pruned_transitions=pruned,
    )


__all__ = [
    "CandidatePromotionResult",
    "PrunedTransition",
    "apply_candidate_promotion",
    "is_preparation_transition",
    "reviewable_candidate_transitions",
    "transition_rejection_reason",
]

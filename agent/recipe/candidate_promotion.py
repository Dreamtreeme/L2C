"""자율탐색 후보에서 Critic이 남긴 행동만 활성 경험 경로로 승격한다."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent.recipe.path_builder import build_experience_paths
from agent.recipe.promotion_policy import (
    StepEvidenceVerdict,
    evaluate_candidate_step_evidence,
)
from agent.recipe.store import RecipeStore
from agent.recipe.task_category import normalize_task_category
from agent.runtime.site_context import normalize_page_role
from agent.runtime.worker_actions import (
    RECIPE_COMMIT_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
    is_supported_recipe_action_group,
)
from shared.schema.feedback_schema import RecipeCandidate, RecipeCandidateReview
from shared.schema.recipe_schema import PhysicalAction, PhysicalActionName
from shared.schema.skill_schema import (
    RECIPE_INPUT_NAMES,
    RecipeSkillMetadata,
)


class PromotionSkip(BaseModel):
    """승격 경로에서 제거한 행동과 그 이유."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    action: PhysicalActionName
    reason: str
    blocking_reasons: list[str] = Field(default_factory=list)


class CandidatePromotionResult(BaseModel):
    """후보 하나를 활성 경험 경로로 승격한 결과."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    promoted: bool
    saved_count: int
    promoted_action_count: int
    promoted_transition_count: int
    promoted_path_count: int
    skipped_steps: list[PromotionSkip] = Field(default_factory=list)


def _kept_step_seqs(review: RecipeCandidateReview) -> set[int]:
    """Critic이 유지한다고 판정한 기존 단계 번호만 반환한다."""

    return {verdict.seq for verdict in review.step_verdicts if verdict.keep}


def _page_roles_from_evidence(candidate: RecipeCandidate) -> dict[int, str]:
    """행동 직전 체크포인트에서 단계별 화면 역할을 읽는다."""

    roles: dict[int, str] = {}
    for transition in candidate.transitions:
        role = normalize_page_role(transition.before.page_role)
        for action in transition.actions:
            if role:
                roles[action.source_seq] = role
    return roles


def _candidate_skill_metadata(
    candidate: RecipeCandidate,
) -> RecipeSkillMetadata:
    """재생에 필요한 입력 슬롯을 자율탐색 행동에서 직접 만든다."""

    slots = {
        name
        for step in candidate.steps
        for name in step.slot_refs
        if name in RECIPE_INPUT_NAMES
    }
    return RecipeSkillMetadata(
        task_category=normalize_task_category(
            candidate.collection_intent.task_category
        ),
        inputs=[{"name": name} for name in sorted(slots)],
    )


def _skip(
    skipped: list[PromotionSkip],
    step: PhysicalAction,
    reason: str,
    *,
    blocking_reasons: list[str] | None = None,
) -> None:
    skipped.append(
        PromotionSkip(
            seq=step.source_seq,
            action=step.action,
            reason=reason,
            blocking_reasons=blocking_reasons or [],
        )
    )


def _parameter_contract_valid(
    step: PhysicalAction,
    declared_inputs: set[str],
) -> bool:
    if step.action != "type_in_marker":
        return False
    slot_name = step.parameter_slot()
    return bool(
        len(step.slot_refs) == 1
        and slot_name == step.slot_refs[0]
        and slot_name in declared_inputs
    )


def _declared_input_names(metadata: RecipeSkillMetadata) -> set[str]:
    return {item.name for item in metadata.inputs}


def _base_promotion_rejection(
    *,
    step: PhysicalAction,
    kept_seqs: set[int],
    evidence: StepEvidenceVerdict | None,
) -> tuple[str, list[str]] | None:
    if step.action not in REVIEWABLE_REPLAY_ACTIONS:
        return "unsupported_action", []
    if step.replay_mode not in {"fixed", "parameterized"}:
        return "not_proposed_for_replay", []
    if step.source_seq not in kept_seqs:
        return "critic_pruned", []
    if evidence and evidence.eligible:
        return None
    reasons = list(evidence.blocking_reasons if evidence else [])
    return reasons[0] if reasons else "action_evidence_missing", reasons


def _target_promotion_rejection(
    step: PhysicalAction,
    declared_inputs: set[str],
) -> str:
    if not step.has_replay_target():
        return "roi_signature_missing"
    if step.replay_mode == "parameterized" and not _parameter_contract_valid(
        step,
        declared_inputs,
    ):
        return "parameter_slot_contract_missing"
    if step.replay_mode == "parameterized" and step.action != "type_in_marker":
        return "unsupported_parameterized_action"
    return ""


def _commit_promotion_rejection(step: PhysicalAction) -> str:
    if step.replay_mode != "fixed":
        return "commit_action_not_fixed"
    if not step.param.key:
        return "key_missing"
    return ""


def _remove_ungrouped_commit_actions(
    steps: list[PhysicalAction],
    skipped: list[PromotionSkip],
) -> list[PhysicalAction]:
    grouped: list[PhysicalAction] = []
    for step in steps:
        if step.action not in RECIPE_COMMIT_ACTIONS:
            grouped.append(step)
            continue
        if grouped and is_supported_recipe_action_group([grouped[-1], step]):
            grouped.append(step)
            continue
        _skip(skipped, step, "commit_action_without_target_input")
    return grouped


def _promotable_steps(
    candidate: RecipeCandidate,
    review: RecipeCandidateReview,
    metadata: RecipeSkillMetadata,
) -> tuple[list[PhysicalAction], list[PromotionSkip]]:
    """원본 행동 중 Critic·실행 근거·재생 계약을 모두 통과한 것만 남긴다."""

    kept_seqs = _kept_step_seqs(review)
    evidence_verdicts = evaluate_candidate_step_evidence(candidate)
    page_roles = _page_roles_from_evidence(candidate)
    declared_inputs = _declared_input_names(metadata)
    promoted: list[PhysicalAction] = []
    skipped: list[PromotionSkip] = []

    for step in candidate.steps:
        rejection = _base_promotion_rejection(
            step=step,
            kept_seqs=kept_seqs,
            evidence=evidence_verdicts.get(step.source_seq),
        )
        if rejection:
            reason, blocking_reasons = rejection
            _skip(
                skipped,
                step,
                reason,
                blocking_reasons=blocking_reasons,
            )
            continue

        if not normalize_page_role(page_roles.get(step.source_seq, "")):
            _skip(skipped, step, "page_role_missing")
            continue

        if step.action in TARGET_REPLAY_ACTIONS:
            reason = _target_promotion_rejection(step, declared_inputs)
            if reason:
                _skip(skipped, step, reason)
                continue
            promoted.append(step.model_copy(deep=True))
            continue

        if step.action in RECIPE_COMMIT_ACTIONS:
            reason = _commit_promotion_rejection(step)
            if reason:
                _skip(skipped, step, reason)
                continue
            promoted.append(step.model_copy(deep=True))

    return _remove_ungrouped_commit_actions(promoted, skipped), skipped


def apply_candidate_promotion(
    candidate: RecipeCandidate,
    review: RecipeCandidateReview,
    db_path=None,
) -> CandidatePromotionResult:
    """Critic이 남긴 자율탐색 행동만 원래 순서의 경로로 저장한다."""

    metadata = _candidate_skill_metadata(candidate)
    replay_steps, skipped_steps = _promotable_steps(
        candidate,
        review,
        metadata,
    )
    recipe_paths, path_issues = build_experience_paths(candidate, replay_steps)
    skipped_steps.extend(PromotionSkip(**issue) for issue in path_issues)
    saved_count = RecipeStore(db_path).replace_recipe_paths(
        candidate.site,
        candidate.goal,
        recipe_paths,
        metadata=metadata,
        source_run_id=candidate.run_id,
    )
    return CandidatePromotionResult(
        promoted=saved_count > 0,
        saved_count=saved_count,
        promoted_action_count=sum(
            len(transition.actions)
            for path in recipe_paths
            for transition in path.transitions
        ),
        promoted_transition_count=sum(
            len(path.transitions) for path in recipe_paths
        ),
        promoted_path_count=saved_count,
        skipped_steps=skipped_steps,
    )


__all__ = [
    "CandidatePromotionResult",
    "PromotionSkip",
    "apply_candidate_promotion",
]

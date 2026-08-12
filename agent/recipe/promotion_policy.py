"""활성 경험 경로로 승격할 행동의 실행 근거를 검사한다."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.runtime.worker_actions import (
    RECIPE_COMMIT_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
    is_supported_recipe_action_group,
)
from shared.schema.feedback_schema import RecipeCandidate
from shared.schema.recipe_schema import (
    ActionResultStatus,
    ExperienceTransition,
    PhysicalAction,
    PhysicalActionName,
    TransitionStatus,
)

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


class StepEvidenceVerdict(BaseModel):
    """행동 하나가 재생 후보가 될 수 있는지에 대한 실행 근거 판정."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    action: PhysicalActionName
    eligible: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    result_status: ActionResultStatus = ""
    transition_source: str = ""
    transition_status: TransitionStatus = ""
    execution_group_seqs: list[int] = Field(default_factory=list)
    effect_verified_by_seq: int | None = None
    evidence_mode: Literal[
        "",
        "deferred_group_effect",
        "group_commit_effect",
    ] = ""


def _transition_after_observation_id(
    transition: ExperienceTransition | None,
) -> str:
    return str(transition.after.observation_id or "") if transition else ""


def _action_evidence_verdict(
    action: PhysicalAction,
    transition: ExperienceTransition | None,
) -> StepEvidenceVerdict:
    reasons: list[str] = []
    if action.risk_level.strip().casefold() == "sensitive":
        reasons.append("sensitive_action")
    evidence = transition.evidence if transition else None
    if evidence is None:
        reasons.append("action_evidence_missing")
        if action.action in RECIPE_COMMIT_ACTIONS:
            reasons.append("transition_evidence_missing")
        return StepEvidenceVerdict(
            seq=action.source_seq,
            action=action.action,
            eligible=False,
            blocking_reasons=list(dict.fromkeys(reasons)),
        )

    if evidence.result_status in _BLOCKING_RESULT_STATUSES:
        reasons.append(f"action_{evidence.result_status}")
    managed_reason = _CODE_MANAGED_TRANSITION_SOURCES.get(evidence.source)
    if managed_reason:
        reasons.append(managed_reason)
    if evidence.reason in _BLOCKING_TRANSITION_REASONS:
        reasons.append(evidence.reason)
    if action.action in RECIPE_COMMIT_ACTIONS and evidence.status != "ready":
        reasons.append("transition_not_ready")
    return StepEvidenceVerdict(
        seq=action.source_seq,
        action=action.action,
        eligible=not reasons,
        blocking_reasons=list(dict.fromkeys(reasons)),
        result_status=evidence.result_status,
        transition_source=evidence.source,
        transition_status=evidence.status,
    )


def _apply_verified_action_groups(
    candidate: RecipeCandidate,
    verdicts: dict[int, StepEvidenceVerdict],
) -> None:
    """마지막 행동이 효과를 만든 입력·실행 묶음의 선행 행동을 허용한다."""

    reviewable = sorted(
        (
            action
            for action in candidate.steps
            if action.action in REVIEWABLE_REPLAY_ACTIONS
        ),
        key=lambda action: action.source_seq,
    )
    for first, second in zip(reviewable, reviewable[1:]):
        if not is_supported_recipe_action_group([first, second]):
            continue
        first_verdict = verdicts.get(first.source_seq)
        second_verdict = verdicts.get(second.source_seq)
        if not first_verdict or not second_verdict or not second_verdict.eligible:
            continue
        first_transition = candidate.transition_for_action(first.source_seq)
        second_transition = candidate.transition_for_action(second.source_seq)
        if (
            not first_transition
            or not second_transition
            or _transition_after_observation_id(first_transition)
            != str(second_transition.before.observation_id or "")
        ):
            continue
        remaining = [
            reason
            for reason in first_verdict.blocking_reasons
            if reason not in _DEFERRED_GROUP_EFFECT_REASONS
        ]
        if remaining:
            continue
        group_seqs = [first.source_seq, second.source_seq]
        first_verdict.eligible = True
        first_verdict.blocking_reasons = []
        first_verdict.execution_group_seqs = group_seqs
        first_verdict.effect_verified_by_seq = second.source_seq
        first_verdict.evidence_mode = "deferred_group_effect"
        second_verdict.execution_group_seqs = group_seqs
        second_verdict.effect_verified_by_seq = second.source_seq
        second_verdict.evidence_mode = "group_commit_effect"


def evaluate_candidate_step_evidence(
    candidate: RecipeCandidate,
) -> dict[int, StepEvidenceVerdict]:
    """각 재생 후보 행동의 실행 결과와 도착 화면 근거를 검사한다."""

    verdicts = {
        action.source_seq: _action_evidence_verdict(
            action,
            candidate.transition_for_action(action.source_seq),
        )
        for action in candidate.steps
        if action.action in REVIEWABLE_REPLAY_ACTIONS
    }
    _apply_verified_action_groups(candidate, verdicts)
    return verdicts


def compact_step_evidence_verdicts(candidate: RecipeCandidate) -> list[dict[str, object]]:
    verdicts = evaluate_candidate_step_evidence(candidate)
    return [
        verdicts[seq].model_dump(mode="json")
        for seq in sorted(verdicts)
    ]


__all__ = [
    "compact_step_evidence_verdicts",
    "evaluate_candidate_step_evidence",
    "StepEvidenceVerdict",
]

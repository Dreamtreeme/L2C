"""자율탐색 후보에서 Critic이 남긴 단계만 활성 Reflex 경로로 승격한다."""

from __future__ import annotations

from typing import Any

from agent.recipe.candidate_store import RecipeCandidateStore
from agent.recipe.path_builder import build_recipe_path
from agent.recipe.promotion_policy import evaluate_candidate_step_evidence
from agent.recipe.store import RecipeStore
from agent.recipe.task_category import normalize_task_category
from agent.runtime.site_context import infer_site_page_role, normalize_page_role
from agent.runtime.worker_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
)
from shared.schema.feedback_schema import RecipeCandidate
from shared.schema.skill_schema import RECIPE_INPUT_NAMES, RecipeSkillMetadata


def _step_seq(step: dict[str, Any]) -> int | None:
    try:
        return int(step.get("seq"))
    except (TypeError, ValueError):
        return None


def _kept_step_seqs(review: dict[str, Any]) -> set[int]:
    """Critic이 유지한다고 판정한 기존 단계 번호만 반환한다."""

    kept: set[int] = set()
    for verdict in review.get("step_verdicts") or []:
        if not isinstance(verdict, dict) or not verdict.get("keep"):
            continue
        seq = _step_seq(verdict)
        if seq is not None:
            kept.add(seq)
    return kept


def _page_roles_from_evidence(candidate: RecipeCandidate) -> dict[int, str]:
    """행동 직전 실제 관찰 화면에서 단계별 화면 역할을 복원한다."""

    roles: dict[int, str] = {}
    for episode in candidate.submission.feedback_episodes:
        args = episode.proposal.args
        before = episode.observation.before
        marker_texts = list(before.get("marker_texts") or [])
        role = normalize_page_role(
            args.get("page_role")
            or infer_site_page_role(
                str(before.get("url") or ""),
                marker_texts,
            )
        )
        if role:
            roles[episode.seq] = role
    return roles


def _candidate_skill_metadata(
    candidate: RecipeCandidate,
) -> RecipeSkillMetadata:
    """재생에 필요한 입력 슬롯을 자율탐색 행동에서 직접 만든다."""

    slots: set[str] = set()
    for step in candidate.steps:
        for raw_name in step.slot_refs:
            name = str(raw_name or "").strip()
            if name in RECIPE_INPUT_NAMES:
                slots.add(name)
    return RecipeSkillMetadata(
        task_category=normalize_task_category(
            candidate.submission.collection_intent.task_category
        ),
        inputs=[{"name": name} for name in sorted(slots)],
    )


def _skip(
    skipped: list[dict[str, Any]],
    step: dict[str, Any],
    reason: str,
    *,
    blocking_reasons: list[str] | None = None,
) -> None:
    item = {
        "seq": step.get("seq"),
        "action": step.get("action"),
        "reason": reason,
    }
    if blocking_reasons:
        item["blocking_reasons"] = blocking_reasons
    skipped.append(item)


def _parameter_contract_valid(
    step: dict[str, Any],
    declared_inputs: set[str],
) -> bool:
    if step.get("action") != "type_in_marker":
        return False
    slot_refs = [
        str(item).strip() for item in (step.get("slot_refs") or []) if str(item).strip()
    ]
    param = step.get("param") if isinstance(step.get("param"), dict) else {}
    param_slot = str(param.get("slot_name") or param.get("slot") or "").strip()
    return bool(
        len(slot_refs) == 1
        and param_slot
        and param_slot == slot_refs[0]
        and param_slot in declared_inputs
    )


def _declared_input_names(metadata: RecipeSkillMetadata) -> set[str]:
    return {item.name.strip() for item in metadata.inputs if item.name.strip()}


def _base_promotion_rejection(
    *,
    seq: int | None,
    action: str,
    mode: str,
    kept_seqs: set[int],
    evidence_verdicts: dict[int, dict[str, Any]],
) -> tuple[str, list[str]] | None:
    if seq is None:
        return "seq_missing", []
    if action not in REVIEWABLE_REPLAY_ACTIONS:
        return "unsupported_action", []
    if mode not in {"fixed", "parameterized"}:
        return "not_proposed_for_replay", []
    if seq not in kept_seqs:
        return "critic_pruned", []
    evidence = evidence_verdicts.get(seq)
    if evidence and evidence.get("eligible"):
        return None
    reasons = list((evidence or {}).get("blocking_reasons") or [])
    return reasons[0] if reasons else "action_evidence_missing", reasons


def _target_promotion_rejection(
    step: dict[str, Any],
    mode: str,
    declared_inputs: set[str],
) -> str:
    if not step.get("roi_signature"):
        return "roi_signature_missing"
    if mode == "parameterized" and not _parameter_contract_valid(
        step,
        declared_inputs,
    ):
        return "parameter_slot_contract_missing"
    if mode == "parameterized" and step.get("action") != "type_in_marker":
        return "unsupported_parameterized_action"
    return ""


def _contextual_promotion_rejection(
    step: dict[str, Any],
    mode: str,
) -> str:
    if mode != "fixed":
        return "contextual_action_not_fixed"
    if not step.get("screen_context_signature"):
        return "screen_context_signature_missing"
    param = step.get("param") if isinstance(step.get("param"), dict) else {}
    if step.get("action") == "press_key" and not param.get("key"):
        return "key_missing"
    if step.get("action") == "switch_tab" and not param.get("direction"):
        return "tab_direction_missing"
    return ""


def _promotable_steps(
    candidate: RecipeCandidate,
    review: dict[str, Any],
    metadata: RecipeSkillMetadata,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """원본 단계 중 세 개의 독립 게이트를 모두 통과한 것만 남긴다."""

    kept_seqs = _kept_step_seqs(review)
    evidence_verdicts = evaluate_candidate_step_evidence(candidate)
    page_roles = _page_roles_from_evidence(candidate)
    declared_inputs = _declared_input_names(metadata)
    promoted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for raw_step in candidate.steps:
        step = raw_step.model_dump(mode="json")
        seq = _step_seq(step)
        action = str(step.get("action") or "")
        mode = str(step.get("replay_mode") or "reasoning")
        rejection = _base_promotion_rejection(
            seq=seq,
            action=action,
            mode=mode,
            kept_seqs=kept_seqs,
            evidence_verdicts=evidence_verdicts,
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

        page_role = normalize_page_role(
            step.get("page_role") or page_roles.get(seq, "")
        )
        if not page_role:
            _skip(skipped, step, "page_role_missing")
            continue
        step["page_role"] = page_role

        if action in TARGET_REPLAY_ACTIONS:
            reason = _target_promotion_rejection(step, mode, declared_inputs)
            if reason:
                _skip(skipped, step, reason)
                continue
            promoted.append(step)
            continue

        if action in CONTEXTUAL_REPLAY_ACTIONS:
            reason = _contextual_promotion_rejection(step, mode)
            if reason:
                _skip(skipped, step, reason)
                continue
            promoted.append(step)

    return promoted, skipped


def apply_candidate_promotion(
    candidate: RecipeCandidate,
    review: dict[str, Any],
    db_path=None,
) -> dict[str, Any]:
    """Critic이 남긴 자율탐색 단계만 원래 순서의 경로로 저장한다."""

    metadata = _candidate_skill_metadata(candidate)
    replay_steps, skipped_steps = _promotable_steps(
        candidate,
        review,
        metadata,
    )
    recipe_path, path_issues = build_recipe_path(
        candidate,
        replay_steps,
    )
    skipped_steps.extend(path_issues)
    recipe_paths = [recipe_path] if recipe_path else []
    saved_count = RecipeStore(db_path).replace_recipe_paths(
        candidate.site,
        candidate.goal,
        recipe_paths,
        metadata=metadata,
        source_run_id=candidate.run_id,
    )
    return {
        "enabled": True,
        "promoted": saved_count > 0,
        "saved_count": saved_count,
        "promoted_action_count": sum(
            len(transition.get("actions") or [])
            for path in recipe_paths
            for transition in path.get("transitions", []) or []
        ),
        "promoted_transition_count": sum(
            len(path.get("transitions", []) or []) for path in recipe_paths
        ),
        "promoted_path_count": saved_count,
        "skipped_steps": skipped_steps,
    }


def reapply_reviewed_candidate_promotion(
    run_id: str,
    db_path=None,
) -> dict[str, Any]:
    """저장된 가지치기 판정을 현재 결정론 정책으로 다시 적용한다."""

    candidate = RecipeCandidateStore(db_path).get_candidate(run_id)
    if not candidate:
        return {
            "run_id": run_id,
            "promoted": False,
            "reason": "candidate_not_found",
        }
    validation = dict(candidate.validation)
    review = dict(validation.get("review", {}) or {})
    if review.get("decision") != "accept" or not any(
        isinstance(item, dict) and item.get("keep")
        for item in review.get("step_verdicts") or []
    ):
        return {
            "run_id": run_id,
            "promoted": False,
            "reason": "stored_review_not_promotable",
        }
    promotion = apply_candidate_promotion(
        candidate,
        review,
        db_path=db_path,
    )
    return {"run_id": run_id, **promotion}


__all__ = [
    "apply_candidate_promotion",
    "reapply_reviewed_candidate_promotion",
]

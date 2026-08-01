"""자율탐색 후보에서 Critic이 남긴 단계만 활성 Reflex 경로로 승격한다."""

from __future__ import annotations

from typing import Any

from agent.recipe.page_context import normalize_page_role
from agent.recipe.path_builder import build_recipe_path
from agent.recipe.promotion_policy import evaluate_candidate_step_evidence
from agent.recipe.replay_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
)
from agent.recipe.task_category import task_category_from_candidate
from agent.runtime.site_context import infer_site_page_role
from agent.utils.model_dump import dump_model
from shared.schema.skill_schema import RecipeSkillMetadata


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


def _page_roles_from_evidence(candidate: dict[str, Any]) -> dict[int, str]:
    """행동 직전 실제 관찰 화면에서 단계별 화면 역할을 복원한다."""

    payload = dict(candidate.get("payload", {}) or {})
    roles: dict[int, str] = {}
    for episode in payload.get("feedback_episodes") or []:
        if not isinstance(episode, dict):
            continue
        seq = _step_seq(episode)
        if seq is None:
            continue
        proposal = (
            episode.get("proposal")
            if isinstance(episode.get("proposal"), dict)
            else {}
        )
        args = (
            proposal.get("args")
            if isinstance(proposal.get("args"), dict)
            else {}
        )
        observation = (
            episode.get("observation")
            if isinstance(episode.get("observation"), dict)
            else {}
        )
        before = (
            observation.get("before")
            if isinstance(observation.get("before"), dict)
            else {}
        )
        marker_texts = (
            before.get("marker_texts")
            if isinstance(before.get("marker_texts"), list)
            else []
        )
        role = normalize_page_role(
            args.get("page_role")
            or infer_site_page_role(
                str(before.get("url") or ""),
                marker_texts,
            )
        )
        if role:
            roles[seq] = role
    return roles


def _candidate_skill_metadata(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """재생에 필요한 입력 슬롯을 자율탐색 행동에서 직접 만든다."""

    slots: dict[str, dict[str, Any]] = {}
    for step in candidate.get("steps") or []:
        if not isinstance(step, dict):
            continue
        param = step.get("param") if isinstance(step.get("param"), dict) else {}
        for raw_name in step.get("slot_refs") or []:
            name = str(raw_name or "").strip()
            if not name or name in slots:
                continue
            slots[name] = {
                "name": name,
                "description": str(step.get("intent") or "").strip(),
                "observed_value": param.get("text"),
                "required": True,
                "source": "recorded_step",
            }
    return dump_model(
        RecipeSkillMetadata(
            when_to_use=str(candidate.get("goal") or ""),
            site=str(candidate.get("site") or ""),
            task_category=task_category_from_candidate(candidate),
            inputs=list(slots.values()),
        )
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
        str(item).strip()
        for item in (step.get("slot_refs") or [])
        if str(item).strip()
    ]
    param = (
        step.get("param")
        if isinstance(step.get("param"), dict)
        else {}
    )
    param_slot = str(
        param.get("slot_name") or param.get("slot") or ""
    ).strip()
    return bool(
        len(slot_refs) == 1
        and param_slot
        and param_slot == slot_refs[0]
        and param_slot in declared_inputs
    )


def _promotable_steps(
    candidate: dict[str, Any],
    review: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """원본 단계 중 세 개의 독립 게이트를 모두 통과한 것만 남긴다."""

    kept_seqs = _kept_step_seqs(review)
    evidence_verdicts = evaluate_candidate_step_evidence(candidate)
    page_roles = _page_roles_from_evidence(candidate)
    declared_inputs = {
        str(item.get("name") or "").strip()
        for item in metadata.get("inputs") or []
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    promoted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for raw_step in candidate.get("steps") or []:
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        seq = _step_seq(step)
        action = str(step.get("action") or "")
        mode = str(step.get("replay_mode") or "reasoning")

        if seq is None:
            _skip(skipped, step, "seq_missing")
            continue
        if action not in REVIEWABLE_REPLAY_ACTIONS:
            _skip(skipped, step, "unsupported_action")
            continue
        if mode not in {"fixed", "parameterized"}:
            _skip(skipped, step, "not_proposed_for_replay")
            continue
        if seq not in kept_seqs:
            _skip(skipped, step, "critic_pruned")
            continue

        evidence = evidence_verdicts.get(seq)
        if not evidence or not evidence.get("eligible"):
            reasons = list((evidence or {}).get("blocking_reasons") or [])
            _skip(
                skipped,
                step,
                reasons[0] if reasons else "action_evidence_missing",
                blocking_reasons=reasons,
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
            if not step.get("roi_signature"):
                _skip(skipped, step, "roi_signature_missing")
                continue
            if mode == "parameterized" and not _parameter_contract_valid(
                step,
                declared_inputs,
            ):
                _skip(
                    skipped,
                    step,
                    "parameter_slot_contract_missing",
                )
                continue
            if mode == "parameterized" and action != "type_in_marker":
                _skip(skipped, step, "unsupported_parameterized_action")
                continue
            promoted.append(step)
            continue

        if action in CONTEXTUAL_REPLAY_ACTIONS:
            if mode != "fixed":
                _skip(skipped, step, "contextual_action_not_fixed")
                continue
            if not step.get("screen_context_signature"):
                _skip(
                    skipped,
                    step,
                    "screen_context_signature_missing",
                )
                continue
            param = (
                step.get("param")
                if isinstance(step.get("param"), dict)
                else {}
            )
            if action == "press_key" and not param.get("key"):
                _skip(skipped, step, "key_missing")
                continue
            if action == "switch_tab" and not param.get("direction"):
                _skip(skipped, step, "tab_direction_missing")
                continue
            promoted.append(step)

    return promoted, skipped


def apply_candidate_promotion(
    candidate: dict[str, Any],
    review: dict[str, Any],
    db_path=None,
) -> dict[str, Any]:
    """Critic이 남긴 자율탐색 단계만 원래 순서의 경로로 저장한다."""

    from agent.recipe.store import RecipeStore

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
        str(candidate.get("site") or ""),
        str(candidate.get("goal") or ""),
        recipe_paths,
        metadata=metadata,
        candidate_id=str(candidate.get("candidate_id") or ""),
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
            len(path.get("transitions", []) or [])
            for path in recipe_paths
        ),
        "promoted_path_count": saved_count,
        "skipped_steps": skipped_steps,
    }


def reapply_reviewed_candidate_promotion(
    candidate_id: str,
    db_path=None,
) -> dict[str, Any]:
    """저장된 가지치기 판정을 현재 결정론 정책으로 다시 적용한다."""

    from agent.recipe.candidate_store import RecipeCandidateStore

    candidate = RecipeCandidateStore(db_path).get_candidate(candidate_id)
    if not candidate:
        return {
            "candidate_id": candidate_id,
            "promoted": False,
            "reason": "candidate_not_found",
        }
    validation = dict(candidate.get("validation", {}) or {})
    review = dict(validation.get("review", {}) or {})
    if (
        review.get("decision") != "accept"
        or not any(
            isinstance(item, dict) and item.get("keep")
            for item in review.get("step_verdicts") or []
        )
    ):
        return {
            "candidate_id": candidate_id,
            "promoted": False,
            "reason": "stored_review_not_promotable",
        }
    promotion = apply_candidate_promotion(
        candidate,
        review,
        db_path=db_path,
    )
    return {"candidate_id": candidate_id, **promotion}


__all__ = [
    "apply_candidate_promotion",
    "reapply_reviewed_candidate_promotion",
]

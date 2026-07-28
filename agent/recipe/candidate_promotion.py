"""Critic 검토 결과를 활성 Reflex 레시피로 변환한다."""

from __future__ import annotations

from typing import Any

from agent.recipe.page_context import normalize_page_role
from agent.recipe.promotion_policy import (
    evaluate_candidate_step_evidence,
    transition_observation_supports_contract_review,
)
from agent.recipe.replay_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
)
from agent.recipe.task_category import (
    normalize_task_category,
    task_category_from_candidate,
)
from agent.runtime.site_context import infer_site_page_role


_FOLLOWUP_PARAMETER_KEYS = {
    "press_key": {"key"},
    "go_back": set(),
    "close_current_tab": set(),
    "switch_tab": {"direction"},
}


def _step_intent_map(review: dict[str, Any]) -> dict[int, dict[str, Any]]:
    metadata = dict(review.get("skill_metadata") or {})
    out: dict[int, dict[str, Any]] = {}
    for item in metadata.get("step_intents") or []:
        if not isinstance(item, dict):
            continue
        try:
            out[int(item.get("seq"))] = dict(item)
        except (TypeError, ValueError):
            continue
    return out


def _transition_contract_map(review: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for item in review.get("transition_contracts") or []:
        if not isinstance(item, dict):
            continue
        try:
            seq = int(item.get("seq"))
        except (TypeError, ValueError):
            continue
        contract = item.get("contract")
        if isinstance(contract, dict):
            out[seq] = dict(contract)
    return out


def ensure_review_task_category(
    review: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    task_category = task_category_from_candidate(candidate)
    if not task_category:
        return review
    out = dict(review)
    metadata = dict(out.get("skill_metadata") or {})
    if not normalize_task_category(metadata.get("task_category")):
        metadata["task_category"] = task_category
        out["skill_metadata"] = metadata
    return out


def _annotated_step(
    step: dict[str, Any],
    intent: dict[str, Any],
    contract: dict[str, Any] | None,
    page_role: str = "",
) -> dict[str, Any]:
    out = dict(step)
    for key in ["intent", "target_role", "component", "expected_after", "fixed", "slot_refs"]:
        value = intent.get(key)
        if value not in (None, "", []):
            out[key] = value
    normalized_page_role = normalize_page_role(
        page_role
        or out.get("page_role")
        or intent.get("page_role")
    )
    if normalized_page_role:
        out["page_role"] = normalized_page_role
    replay_mode = intent.get("replay_mode") or out.get("replay_mode") or "reasoning"
    out["replay_mode"] = replay_mode
    if contract:
        out["transition_contract"] = contract
    return out


def _page_role_map_from_candidate(candidate: dict[str, Any]) -> dict[int, str]:
    payload = dict(candidate.get("payload", {}) or {})
    out: dict[int, str] = {}
    for episode in payload.get("feedback_episodes") or []:
        if not isinstance(episode, dict):
            continue
        try:
            seq = int(episode.get("seq"))
        except (TypeError, ValueError):
            continue
        proposal = episode.get("proposal") if isinstance(episode.get("proposal"), dict) else {}
        args = proposal.get("args") if isinstance(proposal.get("args"), dict) else {}
        observation = (
            episode.get("observation")
            if isinstance(episode.get("observation"), dict)
            else {}
        )
        before = observation.get("before") if isinstance(observation.get("before"), dict) else {}
        result = (observation.get("result") or {}) if isinstance(observation, dict) else {}
        result_args = result.get("args") if isinstance(result.get("args"), dict) else {}
        marker_texts = (
            before.get("marker_texts")
            if isinstance(before.get("marker_texts"), list)
            else []
        )
        page_role = normalize_page_role(
            infer_site_page_role(str(before.get("url") or ""), marker_texts)
            or args.get("page_role")
            or result_args.get("page_role")
        )
        if page_role:
            out[seq] = page_role
    return out


def _feedback_episode_before(
    candidate: dict[str, Any],
    seq: int,
) -> dict[str, Any] | None:
    """현재 단계 바로 전에 실제 실행된 행동 에피소드 하나를 찾는다."""

    payload = dict(candidate.get("payload", {}) or {})
    previous: list[tuple[int, dict[str, Any]]] = []
    for episode in payload.get("feedback_episodes", []) or []:
        if not isinstance(episode, dict):
            continue
        try:
            episode_seq = int(episode.get("seq"))
        except (TypeError, ValueError):
            continue
        if episode_seq < seq:
            previous.append((episode_seq, episode))
    if not previous:
        return None
    return max(previous, key=lambda item: item[0])[1]


def _transition_ready_for_seq(
    candidate: dict[str, Any],
    seq: int,
) -> bool:
    payload = dict(candidate.get("payload", {}) or {})
    for item in payload.get("transition_records", []) or []:
        if (
            not isinstance(item, dict)
            or not str(item.get("action_seq", "")).lstrip("-").isdigit()
            or int(item["action_seq"]) != seq
        ):
            continue
        if item.get("status") == "ready":
            return True
        if transition_observation_supports_contract_review(item):
            return True
    return False


def _followup_trigger(
    candidate: dict[str, Any],
    seq: int,
) -> dict[str, str] | None:
    """직전 행동이 성공했다는 증거가 있을 때만 후속 전략 트리거를 만든다."""

    episode = _feedback_episode_before(candidate, seq)
    if not episode:
        return None
    try:
        trigger_seq = int(episode.get("seq"))
    except (TypeError, ValueError):
        return None
    proposal = (
        episode.get("proposal")
        if isinstance(episode.get("proposal"), dict)
        else {}
    )
    observation = (
        episode.get("observation")
        if isinstance(episode.get("observation"), dict)
        else {}
    )
    result = (
        observation.get("result")
        if isinstance(observation.get("result"), dict)
        else {}
    )
    feedback = (
        episode.get("feedback")
        if isinstance(episode.get("feedback"), dict)
        else {}
    )
    action = str(proposal.get("action") or result.get("action") or "")
    if not action:
        return None
    if str(feedback.get("label") or "") in {
        "wrong_target",
        "no_effect",
        "loop_risk",
        "error",
    }:
        return None
    if str(result.get("status") or "") in {"error", "skipped"}:
        return None
    if action in {
        "click_marker",
        "type_in_marker",
        "scroll",
        "press_key",
        "go_back",
        "close_current_tab",
        "switch_tab",
    } and not _transition_ready_for_seq(candidate, trigger_seq):
        return None

    args = (
        proposal.get("args")
        if isinstance(proposal.get("args"), dict)
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
    page_role = normalize_page_role(
        args.get("page_role")
        or infer_site_page_role(
            str(before.get("url") or ""),
            marker_texts,
        )
    )
    return {
        "action": action,
        "component": str(
            proposal.get("component_candidate")
            or args.get("target_component")
            or args.get("component_candidate")
            or ""
        ),
        "page_role": page_role,
    }


def _followup_page_role(
    step: dict[str, Any],
    inferred_page_role: str = "",
) -> str:
    """모델 선언과 실제 관찰 중 후속 행동 당시 문맥을 가장 잘 보존한다."""

    return normalize_page_role(
        step.get("declared_page_role")
        or step.get("observed_page_role")
        or step.get("page_role")
        or inferred_page_role
    )


def _followup_strategy(
    candidate: dict[str, Any],
    step: dict[str, Any],
    *,
    page_role: str,
    contract: dict[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        seq = int(step.get("seq"))
    except (TypeError, ValueError):
        return None
    action = str(step.get("action") or "")
    allowed_keys = _FOLLOWUP_PARAMETER_KEYS.get(action)
    if allowed_keys is None:
        return None
    trigger = _followup_trigger(candidate, seq)
    if trigger is None:
        return None
    param = {
        key: value
        for key, value in dict(step.get("param") or {}).items()
        if key in allowed_keys and value not in (None, "")
    }
    if action == "press_key" and not param.get("key"):
        return None
    if action == "switch_tab" and not param.get("direction"):
        return None
    return {
        "site": str(candidate.get("site") or ""),
        "task_category": task_category_from_candidate(candidate),
        "trigger": trigger,
        "page_role": page_role,
        "url_template": str(step.get("url_template") or ""),
        "action": action,
        "param": param,
        "expected_after": str(step.get("expected_after") or ""),
        "transition_contract": dict(contract or {}),
    }


def _source_followup_strategies(
    candidate: dict[str, Any],
    source_steps: list[dict[str, Any]],
    page_roles: dict[int, str],
) -> list[dict[str, Any]]:
    """후보가 소유한 후속 전략 키를 계산해 교체 시 이전 값을 정리한다."""

    out: list[dict[str, Any]] = []
    for step in source_steps:
        if step.get("action") not in CONTEXTUAL_REPLAY_ACTIONS:
            continue
        try:
            seq = int(step.get("seq"))
        except (TypeError, ValueError):
            continue
        strategy = _followup_strategy(
            candidate,
            step,
            page_role=_followup_page_role(
                step,
                page_roles.get(seq, ""),
            ),
            contract=None,
        )
        if strategy:
            out.append(strategy)
    return out


def _promotable_replay_steps(
    source_steps: list[dict[str, Any]],
    review: dict[str, Any],
    page_roles: dict[int, str] | None = None,
    evidence_verdicts: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Critic이 승인한 단계 중 broad replay에 안전한 단계만 활성 레시피로 만든다."""
    intents = _step_intent_map(review)
    contracts = _transition_contract_map(review)
    replay_steps: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    page_roles = dict(page_roles or {})
    evidence_verdicts = dict(evidence_verdicts or {})
    metadata = dict(review.get("skill_metadata") or {})
    declared_inputs = {
        str(item.get("name") or "").strip()
        for item in (metadata.get("inputs") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }

    for raw_step in source_steps or []:
        if not isinstance(raw_step, dict):
            continue
        try:
            seq = int(raw_step.get("seq"))
        except (TypeError, ValueError):
            skipped.append({"seq": raw_step.get("seq"), "reason": "seq_missing"})
            continue
        intent = intents.get(seq, {})
        step = _annotated_step(
            raw_step,
            intent,
            contracts.get(seq),
            page_role=page_roles.get(seq, ""),
        )
        action = str(step.get("action") or "")
        replay_mode = str(step.get("replay_mode") or "reasoning")
        verdict = evidence_verdicts.get(seq, {})
        if action in TARGET_REPLAY_ACTIONS and verdict and not verdict.get("eligible", False):
            reasons = list(verdict.get("blocking_reasons") or [])
            skipped.append(
                {
                    "seq": seq,
                    "action": action,
                    "reason": reasons[0] if reasons else "deterministic_validation_failed",
                    "blocking_reasons": reasons,
                }
            )
            continue
        if replay_mode not in {"fixed", "parameterized"}:
            skipped.append({"seq": seq, "action": action, "reason": "reasoning_step"})
            continue
        if action in TARGET_REPLAY_ACTIONS:
            if not normalize_page_role(step.get("page_role")):
                skipped.append({"seq": seq, "action": action, "reason": "page_role_missing"})
                continue
            if not step.get("roi_signature"):
                skipped.append({"seq": seq, "action": action, "reason": "roi_signature_missing"})
                continue
            if action == "type_in_marker" and replay_mode == "parameterized":
                slot_refs = [
                    str(item).strip()
                    for item in (step.get("slot_refs") or [])
                    if str(item).strip()
                ]
                param = step.get("param") if isinstance(step.get("param"), dict) else {}
                param_slot = str(param.get("slot_name") or param.get("slot") or "").strip()
                if (
                    len(slot_refs) != 1
                    or not param_slot
                    or param_slot != slot_refs[0]
                    or param_slot not in declared_inputs
                ):
                    skipped.append(
                        {
                            "seq": seq,
                            "action": action,
                            "reason": "parameter_slot_contract_missing",
                        }
                    )
                    continue
            replay_steps.append(step)
            continue
        if action in CONTEXTUAL_REPLAY_ACTIONS:
            continue
        skipped.append({"seq": seq, "action": action, "reason": "non_target_action"})
    return replay_steps, skipped


def _promotable_followup_strategies(
    candidate: dict[str, Any],
    source_steps: list[dict[str, Any]],
    review: dict[str, Any],
    page_roles: dict[int, str] | None = None,
    evidence_verdicts: dict[int, dict[str, Any]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """좌표 없는 행동은 직전 성공 문맥과 전환 계약이 모두 있을 때만 승격한다."""

    intents = _step_intent_map(review)
    contracts = _transition_contract_map(review)
    page_roles = dict(page_roles or {})
    evidence_verdicts = dict(evidence_verdicts or {})
    source_strategies = _source_followup_strategies(
        candidate,
        source_steps,
        page_roles,
    )
    replay_strategies: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for raw_step in source_steps:
        if raw_step.get("action") not in CONTEXTUAL_REPLAY_ACTIONS:
            continue
        try:
            seq = int(raw_step.get("seq"))
        except (TypeError, ValueError):
            skipped.append(
                {
                    "seq": raw_step.get("seq"),
                    "action": raw_step.get("action"),
                    "reason": "seq_missing",
                }
            )
            continue
        action = str(raw_step.get("action") or "")
        verdict = evidence_verdicts.get(seq, {})
        if not verdict.get("eligible", False):
            reasons = list(verdict.get("blocking_reasons") or [])
            skipped.append(
                {
                    "seq": seq,
                    "action": action,
                    "reason": (
                        reasons[0]
                        if reasons
                        else "deterministic_validation_failed"
                    ),
                    "blocking_reasons": reasons,
                }
            )
            continue
        intent = intents.get(seq, {})
        step = _annotated_step(
            raw_step,
            intent,
            contracts.get(seq),
            page_role=_followup_page_role(
                raw_step,
                page_roles.get(seq, ""),
            ),
        )
        if str(step.get("replay_mode") or "reasoning") != "fixed":
            skipped.append(
                {
                    "seq": seq,
                    "action": action,
                    "reason": "reasoning_step",
                }
            )
            continue
        contract = contracts.get(seq)
        if not contract:
            skipped.append(
                {
                    "seq": seq,
                    "action": action,
                    "reason": "transition_contract_missing",
                }
            )
            continue
        page_role = _followup_page_role(
            step,
            page_roles.get(seq, ""),
        )
        if not page_role:
            skipped.append(
                {
                    "seq": seq,
                    "action": action,
                    "reason": "page_role_missing",
                }
            )
            continue
        strategy = _followup_strategy(
            candidate,
            step,
            page_role=page_role,
            contract=contract,
        )
        if strategy is None:
            skipped.append(
                {
                    "seq": seq,
                    "action": action,
                    "reason": "successful_trigger_missing",
                }
            )
            continue
        replay_strategies.append(strategy)

    return source_strategies, replay_strategies, skipped


def apply_candidate_promotion(
    candidate: dict[str, Any],
    review: dict[str, Any],
    db_path=None,
) -> dict[str, Any]:
    from agent.recipe.store import RecipeStore

    source_steps = [
        dict(step)
        for step in candidate.get("steps", []) or []
        if isinstance(step, dict)
    ]
    page_roles = _page_role_map_from_candidate(candidate)
    evidence_verdicts = evaluate_candidate_step_evidence(candidate)
    replay_steps, skipped_steps = _promotable_replay_steps(
        source_steps,
        review,
        page_roles=page_roles,
        evidence_verdicts=evidence_verdicts,
    )
    (
        source_followups,
        replay_followups,
        skipped_followups,
    ) = _promotable_followup_strategies(
        candidate,
        source_steps,
        review,
        page_roles=page_roles,
        evidence_verdicts=evidence_verdicts,
    )
    store = RecipeStore(db_path)
    site = candidate.get("site", "") or ""
    roi_saved_count = store.replace_recipe_steps(
        site,
        candidate.get("goal", "") or "",
        source_steps,
        replay_steps,
        metadata=dict(review.get("skill_metadata") or {}),
    )
    followup_saved_count = store.replace_followup_strategies(
        site,
        source_followups,
        replay_followups,
    )
    saved_count = roi_saved_count + followup_saved_count
    return {
        "enabled": True,
        "promoted": saved_count > 0,
        "saved_count": saved_count,
        "promoted_step_count": len(replay_steps) + len(replay_followups),
        "promoted_roi_step_count": len(replay_steps),
        "promoted_followup_count": len(replay_followups),
        "skipped_steps": [*skipped_steps, *skipped_followups],
    }


def reapply_reviewed_candidate_promotion(candidate_id: str, db_path=None) -> dict[str, Any]:
    """저장된 Critic 판정을 현재 결정론 정책으로 다시 적용한다."""

    from agent.recipe.candidate_store import RecipeCandidateStore

    candidate = RecipeCandidateStore(db_path).get_candidate(candidate_id)
    if not candidate:
        return {"candidate_id": candidate_id, "promoted": False, "reason": "candidate_not_found"}
    validation = dict(candidate.get("validation", {}) or {})
    review = dict(validation.get("review", {}) or {})
    if review.get("decision") != "accept" or not review.get("promote_to_active_recipe"):
        return {
            "candidate_id": candidate_id,
            "promoted": False,
            "reason": "stored_review_not_promotable",
        }
    promotion = apply_candidate_promotion(candidate, review, db_path=db_path)
    return {"candidate_id": candidate_id, **promotion}


__all__ = [
    "apply_candidate_promotion",
    "ensure_review_task_category",
    "reapply_reviewed_candidate_promotion",
]

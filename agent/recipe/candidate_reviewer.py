"""반사 레시피 후보 비평가 게이트(RecipeCandidateReview).

이 모듈은 후보 증거(candidate evidence)를 포장하고 비평가(Critic)의 판정을
적용한다. 대상 품질, 일반화 가능성, 재사용 가능성 같은 의미 판단은 코드에서
직접 하지 않는다.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from agent.recipe.task_category import normalize_task_category
from shared.schema.feedback_schema import RecipeCandidateReview


CriticFn = Callable[[dict[str, Any]], dict[str, Any] | RecipeCandidateReview]


def _dump_model(model) -> dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _coerce_review(raw: dict[str, Any] | RecipeCandidateReview) -> dict[str, Any]:
    if isinstance(raw, RecipeCandidateReview):
        return _dump_model(raw)
    return _dump_model(RecipeCandidateReview(**(raw or {})))


def _fallback_review(reason: str) -> dict[str, Any]:
    return _dump_model(
        RecipeCandidateReview(
            decision="revise",
            reasons=[reason],
            feedback_to_worker="Candidate review could not be completed. Re-submit with clearer worker evidence.",
            promote_to_active_recipe=False,
            confidence=0.0,
        )
    )


def _candidate_task_category(candidate: dict[str, Any]) -> str:
    worker_submission = dict(candidate.get("payload", {}) or {})
    evidence = worker_submission.get("skill_metadata_evidence") if isinstance(worker_submission.get("skill_metadata_evidence"), dict) else {}
    return normalize_task_category(worker_submission.get("task_category") or evidence.get("task_category") or "")


def build_candidate_review_payload(candidate: dict[str, Any], allow_promotion: bool = False) -> dict[str, Any]:
    """후보 증거(candidate evidence)를 의미 판단 없이 비평가(Critic)에게 전달한다."""
    worker_submission = dict(candidate.get("payload", {}) or {})
    task_category = _candidate_task_category(candidate)
    promotion_policy = (
        "Active recipe promotion is enabled for this review only if the candidate is reusable. "
        "Set promote_to_active_recipe=true only when fixed/parameterized steps are safe to replay. "
        "The code will still activate only ROI-verifiable click/type target actions."
        if allow_promotion
        else "Active recipe promotion is disabled in this review, so promote_to_active_recipe is only a recommendation signal."
    )
    return {
        "candidate_id": candidate.get("candidate_id", "") or "",
        "status": candidate.get("status", "") or "",
        "site": candidate.get("site", "") or "",
        "task_category": task_category,
        "goal": candidate.get("goal", "") or "",
        "keyword": candidate.get("keyword", "") or "",
        "steps": list(candidate.get("steps", []) or []),
        "transition_observations": list(worker_submission.get("transition_observations", []) or []),
        "skill_metadata_evidence": dict(worker_submission.get("skill_metadata_evidence", {}) or {}),
        "worker_submission": worker_submission,
        "commander_review": dict(candidate.get("review", {}) or {}),
        "promotion_enabled": allow_promotion,
        "promotion_policy": promotion_policy,
        "review_task": (
            "Decide whether this candidate is reusable Reflex recipe evidence. "
            f"{promotion_policy} "
            "Use the worker evidence, skill metadata evidence, and commander review. If accepted, fill "
            "skill_metadata with task_category, when_to_use, inputs, step_intents, verification, and fallback conditions. "
            "Preserve the supplied task_category instead of inventing a different category unless the evidence clearly contradicts it. "
            "For every recorded step, set step_intents.replay_mode to fixed, parameterized, or reasoning. "
            "Use fixed only when the same UI operation is valid across runs. Use parameterized when only a named "
            "runtime slot changes. Use reasoning for choices that depend on the current screen, current result set, "
            "visited items, or remaining target count. A job-card title selected from search results must be "
            "reasoning unless its current runtime title is supplied through an explicit slot; never replay a job "
            "title observed in the exploration run as if it were a stable control. "
            "Also assign OCR-verifiable transition_contracts to the recorded action seq values. Build contracts "
            "only from observed transition evidence: common cues identify the completed page, outcomes distinguish "
            "known normal branches such as results_found/results_empty, and loading cues only describe observed "
            "intermediate screens. Do not invent an unseen outcome. "
            "Return revise/reject with feedback when the candidate needs another worker run or should not be reused."
        ),
    }


def _llm_review_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = os.getenv("VISION_RECIPE_CRITIC_MODEL", os.getenv("VISION_WORKER_REVIEW_MODEL", "gemini-3.5-flash"))
    promotion_policy = str(payload.get("promotion_policy") or "")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0).with_structured_output(RecipeCandidateReview)
    messages = [
        SystemMessage(
            content=(
                "You are the Reflex Recipe Critic. The script layer only packaged evidence and did not judge "
                "semantic quality. Review the candidate and return only RecipeCandidateReview. "
                f"{promotion_policy} "
                "When accepting, write concise skill_metadata that preserves the supplied task_category and explains when to use the recipe, which inputs "
                "are variable, which steps are fixed, parameterized, or require reasoning, and how replay success "
                "or fallback should be verified. Every step_intent must include replay_mode. Current search-result "
                "card selection and target-count-dependent navigation require reasoning unless backed by an explicit "
                "runtime slot or condition. "
                "Compile transition_observations into transition_contracts keyed by recorded action seq. Use only "
                "OCR-verifiable cues and only outcomes supported by observations."
            )
        ),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
    ]
    return _coerce_review(llm.invoke(messages))


def review_candidate(
    candidate: dict[str, Any],
    critic: CriticFn | None = None,
    allow_promotion: bool = False,
) -> dict[str, Any]:
    payload = build_candidate_review_payload(candidate, allow_promotion=allow_promotion)
    try:
        raw = critic(payload) if critic else _llm_review_candidate(payload)
        return _coerce_review(raw)
    except Exception as exc:
        return _fallback_review(f"critic_review_failed: {str(exc)[:200]}")


def _status_for_review(review: dict[str, Any]) -> str:
    decision = review.get("decision")
    if decision == "accept":
        return "accepted"
    if decision == "reject":
        return "rejected"
    return "revise"


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


def _ensure_review_task_category(review: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    task_category = _candidate_task_category(candidate)
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
) -> dict[str, Any]:
    out = dict(step)
    for key in ["intent", "target_role", "component", "expected_after", "fixed", "slot_refs"]:
        value = intent.get(key)
        if value not in (None, "", []):
            out[key] = value
    replay_mode = intent.get("replay_mode") or out.get("replay_mode") or "reasoning"
    out["replay_mode"] = replay_mode
    if contract:
        out["transition_contract"] = contract
    return out


def _promotable_replay_steps(
    source_steps: list[dict[str, Any]],
    review: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Critic이 승인한 단계 중 broad replay에 안전한 단계만 활성 레시피로 만든다."""
    intents = _step_intent_map(review)
    contracts = _transition_contract_map(review)
    replay_steps: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    target_actions = {"click_marker", "type_in_marker"}

    for raw_step in source_steps or []:
        if not isinstance(raw_step, dict):
            continue
        try:
            seq = int(raw_step.get("seq"))
        except (TypeError, ValueError):
            skipped.append({"seq": raw_step.get("seq"), "reason": "seq_missing"})
            continue
        intent = intents.get(seq, {})
        step = _annotated_step(raw_step, intent, contracts.get(seq))
        action = str(step.get("action") or "")
        replay_mode = str(step.get("replay_mode") or "reasoning")
        if replay_mode not in {"fixed", "parameterized"}:
            skipped.append({"seq": seq, "action": action, "reason": "reasoning_step"})
            continue
        if action in target_actions:
            if not step.get("roi_signature"):
                skipped.append({"seq": seq, "action": action, "reason": "roi_signature_missing"})
                continue
            replay_steps.append(step)
            continue
        skipped.append({"seq": seq, "action": action, "reason": "non_target_action"})
    return replay_steps, skipped


def _apply_active_promotion(
    candidate: dict[str, Any],
    review: dict[str, Any],
    db_path=None,
) -> dict[str, Any]:
    from agent.recipe.store import RecipeStore

    source_steps = [dict(step) for step in candidate.get("steps", []) or [] if isinstance(step, dict)]
    replay_steps, skipped_steps = _promotable_replay_steps(source_steps, review)
    saved_count = 0
    if replay_steps:
        saved_count = RecipeStore(db_path).replace_recipe_steps(
            candidate.get("site", "") or "",
            candidate.get("goal", "") or "",
            source_steps,
            replay_steps,
            metadata=dict(review.get("skill_metadata") or {}),
        )
    return {
        "enabled": True,
        "promoted": saved_count > 0,
        "saved_count": saved_count,
        "promoted_step_count": len(replay_steps),
        "skipped_steps": skipped_steps,
    }


def review_and_apply_candidate(
    candidate_id: str,
    db_path=None,
    critic: CriticFn | None = None,
    mode: str = "review",
) -> dict[str, Any]:
    from agent.recipe.candidate_store import RecipeCandidateStore

    candidate_store = RecipeCandidateStore(db_path)
    candidate = candidate_store.get_candidate(candidate_id)
    if not candidate:
        return _fallback_review(f"candidate_not_found: {candidate_id}")

    normalized_mode = _process_mode(mode)
    allow_promotion = normalized_mode == "promote"
    review = _ensure_review_task_category(
        review_candidate(candidate, critic=critic, allow_promotion=allow_promotion),
        candidate,
    )
    promotion = {"enabled": allow_promotion, "promoted": False, "saved_count": 0, "promoted_step_count": 0, "skipped_steps": []}
    if allow_promotion and review.get("decision") == "accept" and review.get("promote_to_active_recipe"):
        promotion = _apply_active_promotion(candidate, review, db_path=db_path)

    validation = {
        "review": review,
        "promotion": promotion,
    }
    candidate_store.update_status(candidate_id, _status_for_review(review), validation=validation)
    out = dict(review)
    out["candidate_id"] = candidate_id
    out["promotion"] = promotion
    return out


def _process_mode(mode: str | None) -> str:
    normalized = (mode or "review").strip().lower()
    return normalized if normalized in {"review", "promote"} else "review"


def process_recipe_candidates(
    limit: int = 5,
    mode: str = "review",
    status: str = "pending_replay",
    db_path=None,
    critic: CriticFn | None = None,
) -> dict[str, Any]:
    """저장된 후보(recipe_candidates)를 비평가 게이트(Critic gate)에 넘긴다.

    스크립트 계층은 상태(status)로 후보 행만 고르고, 품질 판단은 하지 않는다.
    """
    from agent.recipe.candidate_store import RecipeCandidateStore

    normalized_mode = _process_mode(mode)
    safe_limit = max(0, int(limit or 0))
    candidates = RecipeCandidateStore(db_path).list_recent(limit=safe_limit, status=status or None)
    results = [
        review_and_apply_candidate(
            candidate["candidate_id"],
            db_path=db_path,
            critic=critic,
            mode=normalized_mode,
        )
        for candidate in candidates
    ]
    return {
        "mode": normalized_mode,
        "status": status,
        "requested_limit": safe_limit,
        "candidate_count": len(candidates),
        "processed_count": len(results),
        "results": results,
    }

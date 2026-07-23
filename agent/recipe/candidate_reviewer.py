"""반사 레시피 후보 비평가 게이트(RecipeCandidateReview).

이 모듈은 후보 증거(candidate evidence)를 포장하고 비평가(Critic)의 판정을
적용한다. 대상 품질, 일반화 가능성, 재사용 가능성 같은 의미 판단은 코드에서
직접 하지 않는다.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from agent.config import get_settings
from agent.recipe.page_context import infer_page_role_from_url_and_texts, normalize_page_role
from agent.recipe.promotion_policy import compact_step_evidence_verdicts, evaluate_candidate_step_evidence
from agent.recipe.task_category import normalize_task_category
from agent.utils.model_dump import dump_model
from shared.schema.feedback_schema import RecipeCandidateReview


CriticFn = Callable[[dict[str, Any]], dict[str, Any] | RecipeCandidateReview]


def _critic_evidence_text_limit() -> int:
    return get_settings().recipe.critic_evidence_text_limit


def _target_action_seqs(steps: list[dict[str, Any]]) -> set[int]:
    target_seqs: set[int] = set()
    for step in steps or []:
        if not isinstance(step, dict) or step.get("action") not in {"click_marker", "type_in_marker"}:
            continue
        try:
            target_seqs.add(int(step.get("seq")))
        except (TypeError, ValueError):
            continue
    return target_seqs


def _compact_transition_observations(
    worker_submission: dict[str, Any],
    target_seqs: set[int],
) -> list[dict[str, Any]]:
    """승격 가능한 대상 행동의 전환 결과만 제한된 OCR 증거와 함께 남긴다."""

    text_limit = _critic_evidence_text_limit()
    evidence: list[dict[str, Any]] = []
    keys = (
        "action_seq",
        "action",
        "expected_after",
        "source",
        "attempt",
        "elapsed_sec",
        "status",
        "outcome",
        "reason",
        "phash_distance",
        "ocr_skipped",
        "marker_count",
    )
    for observation in worker_submission.get("transition_observations", []) or []:
        if not isinstance(observation, dict):
            continue
        try:
            seq = int(observation.get("action_seq"))
        except (TypeError, ValueError):
            continue
        if seq not in target_seqs:
            continue
        item = {
            key: observation.get(key)
            for key in keys
            if observation.get(key) not in (None, "", [], {})
        }
        item["marker_texts"] = list(observation.get("marker_texts", []) or [])[:text_limit]
        evidence.append(item)
    return evidence


def _compact_worker_execution(worker_submission: dict[str, Any]) -> dict[str, Any]:
    """Critic에 필요한 실행 결과만 남겨 반복된 전체 상태를 제거한다."""

    keys = (
        "run_status",
        "is_finished",
        "hit_recursion_limit",
        "collected_count",
        "target_count",
        "persisted_count",
        "worker_notes",
    )
    execution = {
        key: worker_submission.get(key)
        for key in keys
        if worker_submission.get(key) not in (None, "", [], {})
    }
    execution["extracted_summary"] = dict(worker_submission.get("extracted_summary", {}) or {})
    return execution


def _compact_feedback_evidence(
    worker_submission: dict[str, Any],
    target_seqs: set[int],
) -> list[dict[str, Any]]:
    """행동별 이전 화면과 실행 결과를 Critic이 비교할 수 있게 축약한다."""

    text_limit = _critic_evidence_text_limit()
    evidence: list[dict[str, Any]] = []
    for episode in worker_submission.get("feedback_episodes", []) or []:
        if not isinstance(episode, dict):
            continue
        try:
            seq = int(episode.get("seq"))
        except (TypeError, ValueError):
            continue
        if seq not in target_seqs:
            continue
        proposal = episode.get("proposal") if isinstance(episode.get("proposal"), dict) else {}
        observation = episode.get("observation") if isinstance(episode.get("observation"), dict) else {}
        before = observation.get("before") if isinstance(observation.get("before"), dict) else {}
        after = observation.get("after") if isinstance(observation.get("after"), dict) else {}
        result = observation.get("result") if isinstance(observation.get("result"), dict) else {}
        feedback = episode.get("feedback") if isinstance(episode.get("feedback"), dict) else {}
        item = {
            "seq": seq,
            "action": proposal.get("action") or result.get("action") or "",
            "expected_after": proposal.get("expected_after") or "",
            "before_url": before.get("url") or "",
            "after_url": after.get("url") or "",
            "screen_changed": bool(after.get("screen_changed", False)),
            "before_marker_texts": list(before.get("marker_texts", []) or [])[:text_limit],
            "result_status": result.get("status") or "",
            "result_reason": result.get("reason") or "",
            "feedback_label": feedback.get("label") or "",
            "feedback_reason": feedback.get("reason") or "",
        }
        evidence.append({key: value for key, value in item.items() if value not in (None, "", [], {})})
    return evidence


def _coerce_review(raw: dict[str, Any] | RecipeCandidateReview) -> dict[str, Any]:
    if isinstance(raw, RecipeCandidateReview):
        return dump_model(raw)
    return dump_model(RecipeCandidateReview(**(raw or {})))


def _fallback_review(reason: str) -> dict[str, Any]:
    return dump_model(
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
    steps = [step for step in candidate.get("steps", []) or [] if isinstance(step, dict)]
    target_seqs = _target_action_seqs(steps)
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
        "steps": steps,
        "transition_observations": _compact_transition_observations(worker_submission, target_seqs),
        "feedback_evidence": _compact_feedback_evidence(worker_submission, target_seqs),
        "deterministic_step_validation": compact_step_evidence_verdicts(candidate),
        "skill_metadata_evidence": dict(worker_submission.get("skill_metadata_evidence", {}) or {}),
        "worker_execution": _compact_worker_execution(worker_submission),
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
            "Judge each recorded action independently; an overall successful run does not prove that every click was correct. "
            "Treat deterministic_step_validation entries with eligible=false as blocked by code; never classify those steps as reusable. "
            "Compare each step with feedback_evidence and the transition_observation having the same seq/action_seq. "
            "If a click was a no-op, opened an unrelated page, or has no post-action evidence supporting expected_after, "
            "mark that step as reasoning instead of fixed/parameterized. "
            "Preserve or assign page_role for every fixed/parameterized replay step; active replay requires page_role plus ROI. "
            "Use fixed only when the same UI operation is valid across runs. Use parameterized when only a named "
            "runtime slot changes. Use reasoning for choices that depend on the current screen, current result set, "
            "visited items, or remaining target count. A job-card title selected from search results must be "
            "reasoning unless its current runtime title is supplied through an explicit slot; never replay a job "
            "title observed in the exploration run as if it were a stable control. "
            "Choosing which filter category to open and deciding when to apply filters depend on the current request "
            "and selected filter state, so keep those steps as reasoning unless an explicit runtime slot or condition "
            "fully determines the choice. A parameterized filter input must name a required runtime slot; never fall "
            "back to the exploration run's filter text when that slot is absent. "
            "Also assign OCR-verifiable transition_contracts to the recorded action seq values. Build contracts "
            "only from observed transition evidence: common cues identify the completed page, outcomes distinguish "
            "known normal branches such as results_found/results_empty, and loading cues only describe observed "
            "intermediate screens. Generic header text visible both before and after an action is not a valid ready cue. "
            "Do not invent an unseen outcome. "
            "Return revise/reject with feedback when the candidate needs another worker run or should not be reused."
        ),
    }


def _llm_review_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.application.model_policy import commander_model_name

    model_name = commander_model_name(
        "VISION_RECIPE_CRITIC_MODEL",
        "VISION_WORKER_REVIEW_MODEL",
    )
    request_timeout = get_settings().recipe.critic_timeout_sec
    promotion_policy = str(payload.get("promotion_policy") or "")
    from agent.application.model_clients import get_structured_google_model

    llm = get_structured_google_model(
        model_name,
        RecipeCandidateReview,
        temperature=0.0,
        request_timeout=request_timeout,
        retries=1,
    )
    messages = [
        SystemMessage(
            content=(
                "You are the Reflex Recipe Critic. The script layer only packaged evidence and did not judge "
                "semantic quality. Review the candidate and return only RecipeCandidateReview. "
                f"{promotion_policy} "
                "When accepting, write concise skill_metadata that preserves the supplied task_category and explains when to use the recipe, which inputs "
                "are variable, which steps are fixed, parameterized, or require reasoning, and how replay success "
                "or fallback should be verified. Every step_intent must include replay_mode, and reusable click/type steps must preserve page_role. Current search-result "
                "card selection and target-count-dependent navigation require reasoning unless backed by an explicit "
                "runtime slot or condition. "
                "Choosing a filter category and applying filters are current-request/current-state decisions unless "
                "an explicit runtime slot or condition fully determines them. Parameterized filter text requires a "
                "named runtime input and must not reuse the exploration value when that input is missing. "
                "Judge every action independently. Overall task success does not make a wrong click reusable. Compare "
                "deterministic_step_validation first. A step with eligible=false is blocked and must be reasoning. "
                "feedback_evidence with the transition_observation for the same seq/action_seq; classify no-op clicks, "
                "unrelated navigation, and actions without evidence for expected_after as reasoning. "
                "Compile transition_observations into transition_contracts keyed by recorded action seq. Use only "
                "OCR-verifiable cues that distinguish the post-action screen from feedback_evidence.before_marker_texts. "
                "Generic headers visible before and after the action are not ready cues. Use only outcomes supported by observations."
            )
        ),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
    ]
    from agent.application.run_context import invoke_with_metrics

    return _coerce_review(invoke_with_metrics(llm, messages, "recipe_critic"))


def review_candidate(
    candidate: dict[str, Any],
    critic: CriticFn | None = None,
    allow_promotion: bool = False,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    payload = build_candidate_review_payload(candidate, allow_promotion=allow_promotion)
    try:
        raw = critic(payload) if critic else _llm_review_candidate(payload)
        return _coerce_review(raw)
    except Exception as exc:
        if raise_on_error:
            raise
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
    page_role: str = "",
) -> dict[str, Any]:
    out = dict(step)
    for key in ["intent", "target_role", "component", "expected_after", "fixed", "slot_refs"]:
        value = intent.get(key)
        if value not in (None, "", []):
            out[key] = value
    normalized_page_role = normalize_page_role(page_role or out.get("page_role") or intent.get("page_role"))
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
        observation = episode.get("observation") if isinstance(episode.get("observation"), dict) else {}
        before = observation.get("before") if isinstance(observation.get("before"), dict) else {}
        result = (observation.get("result") or {}) if isinstance(observation, dict) else {}
        result_args = result.get("args") if isinstance(result.get("args"), dict) else {}
        marker_texts = before.get("marker_texts") if isinstance(before.get("marker_texts"), list) else []
        page_role = normalize_page_role(
            infer_page_role_from_url_and_texts(str(before.get("url") or ""), marker_texts)
            or args.get("page_role")
            or result_args.get("page_role")
        )
        if page_role:
            out[seq] = page_role
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
    target_actions = {"click_marker", "type_in_marker"}
    page_roles = dict(page_roles or {})
    evidence_verdicts = dict(evidence_verdicts or {})

    for raw_step in source_steps or []:
        if not isinstance(raw_step, dict):
            continue
        try:
            seq = int(raw_step.get("seq"))
        except (TypeError, ValueError):
            skipped.append({"seq": raw_step.get("seq"), "reason": "seq_missing"})
            continue
        intent = intents.get(seq, {})
        step = _annotated_step(raw_step, intent, contracts.get(seq), page_role=page_roles.get(seq, ""))
        action = str(step.get("action") or "")
        replay_mode = str(step.get("replay_mode") or "reasoning")
        verdict = evidence_verdicts.get(seq, {})
        if action in target_actions and verdict and not verdict.get("eligible", False):
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
        if action in target_actions:
            if not normalize_page_role(step.get("page_role")):
                skipped.append({"seq": seq, "action": action, "reason": "page_role_missing"})
                continue
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
    replay_steps, skipped_steps = _promotable_replay_steps(
        source_steps,
        review,
        page_roles=_page_role_map_from_candidate(candidate),
        evidence_verdicts=evaluate_candidate_step_evidence(candidate),
    )
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


def reapply_reviewed_candidate_promotion(candidate_id: str, db_path=None) -> dict[str, Any]:
    """저장된 Critic 판정을 현재 결정론 정책으로 다시 적용한다."""

    from agent.recipe.candidate_store import RecipeCandidateStore

    candidate = RecipeCandidateStore(db_path).get_candidate(candidate_id)
    if not candidate:
        return {"candidate_id": candidate_id, "promoted": False, "reason": "candidate_not_found"}
    validation = dict(candidate.get("validation", {}) or {})
    review = dict(validation.get("review", {}) or {})
    if review.get("decision") != "accept" or not review.get("promote_to_active_recipe"):
        return {"candidate_id": candidate_id, "promoted": False, "reason": "stored_review_not_promotable"}
    promotion = _apply_active_promotion(candidate, review, db_path=db_path)
    return {"candidate_id": candidate_id, **promotion}


def review_and_apply_candidate(
    candidate_id: str,
    db_path=None,
    critic: CriticFn | None = None,
    mode: str = "review",
    raise_on_critic_error: bool = False,
) -> dict[str, Any]:
    from agent.recipe.candidate_store import RecipeCandidateStore

    candidate_store = RecipeCandidateStore(db_path)
    candidate = candidate_store.get_candidate(candidate_id)
    if not candidate:
        return _fallback_review(f"candidate_not_found: {candidate_id}")

    normalized_mode = _process_mode(mode)
    allow_promotion = normalized_mode == "promote"
    review = _ensure_review_task_category(
        review_candidate(
            candidate,
            critic=critic,
            allow_promotion=allow_promotion,
            raise_on_error=raise_on_critic_error,
        ),
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

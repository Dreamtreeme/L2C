"""반사 레시피 후보 비평가 게이트(RecipeCandidateReview).

이 모듈은 후보 증거(candidate evidence)를 포장하고 비평가(Critic)의 판정을
적용한다. 대상 품질, 일반화 가능성, 재사용 가능성 같은 의미 판단은 코드에서
직접 하지 않는다.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from agent.config import get_settings
from agent.recipe.candidate_promotion import (
    apply_candidate_promotion,
    ensure_review_task_category,
)
from agent.recipe.promotion_policy import compact_step_evidence_verdicts
from agent.recipe.replay_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
)
from agent.recipe.task_category import task_category_from_candidate
from agent.utils.model_dump import dump_model
from shared.schema.feedback_schema import RecipeCandidateReview


CriticFn = Callable[[dict[str, Any]], dict[str, Any] | RecipeCandidateReview]


def _critic_evidence_text_limit() -> int:
    return get_settings().recipe.critic_evidence_text_limit


def _reviewable_action_specs(
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for step in steps or []:
        if (
            not isinstance(step, dict)
            or step.get("action") not in REVIEWABLE_REPLAY_ACTIONS
        ):
            continue
        try:
            seq = int(step.get("seq"))
        except (TypeError, ValueError):
            continue
        specs.append({"seq": seq, "action": str(step.get("action") or "")})
    return specs


def _reviewable_action_seqs(steps: list[dict[str, Any]]) -> set[int]:
    return {item["seq"] for item in _reviewable_action_specs(steps)}


def _feedback_evidence_seqs(
    worker_submission: dict[str, Any],
    steps: list[dict[str, Any]],
    reviewable_seqs: set[int],
) -> set[int]:
    """후속 행동에는 바로 앞 행동의 성공 문맥도 Critic 증거로 포함한다."""

    contextual_seqs = {
        int(step["seq"])
        for step in steps
        if (
            isinstance(step, dict)
            and step.get("action") in CONTEXTUAL_REPLAY_ACTIONS
            and str(step.get("seq", "")).lstrip("-").isdigit()
        )
    }
    episode_seqs = sorted(
        {
            int(episode["seq"])
            for episode in worker_submission.get("feedback_episodes", []) or []
            if (
                isinstance(episode, dict)
                and str(episode.get("seq", "")).lstrip("-").isdigit()
            )
        }
    )
    out = set(reviewable_seqs)
    for seq in contextual_seqs:
        previous = [item for item in episode_seqs if item < seq]
        if previous:
            out.add(previous[-1])
    return out


def _compact_transition_records(
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
    for observation in worker_submission.get("transition_records", []) or []:
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


def _promotion_policy(allow_promotion: bool) -> str:
    if allow_promotion:
        return (
            "Active recipe promotion is evaluated per step, not for the trajectory as one indivisible path. "
            "Set promote_to_active_recipe=true when at least one fixed/parameterized step is independently "
            "safe to replay, even if the same successful trajectory also contains mistakes, recovery actions, or "
            "reasoning-only steps. Classify every unsafe step as reasoning; the code will activate only the safe, "
            "ROI-verifiable click/type steps and transition-verified contextual follow-up actions. "
            "Set promote_to_active_recipe=false only when no step is safe "
            "for active replay. Do not maximize the number of promoted steps. A screen change alone is not success. "
            "A step is reusable only when its expected result was achieved and it belongs to the causal path that "
            "completed the task. If later actions abandoned that branch, reopened the start page, or used an "
            "alternative route to recover, classify the abandoned branch steps as reasoning."
        )
    return (
        "Active recipe promotion is disabled in this review, so promote_to_active_recipe is only "
        "a recommendation signal."
    )


def _serialize_candidate_review_payload(payload: dict[str, Any]) -> str:
    """구조를 유지하면서 전송에 불필요한 JSON 공백을 제거한다."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_candidate_review_payload(candidate: dict[str, Any], allow_promotion: bool = False) -> dict[str, Any]:
    """후보 증거(candidate evidence)를 의미 판단 없이 비평가(Critic)에게 전달한다."""
    worker_submission = dict(candidate.get("payload", {}) or {})
    steps = [step for step in candidate.get("steps", []) or [] if isinstance(step, dict)]
    reviewable_seqs = _reviewable_action_seqs(steps)
    evidence_seqs = _feedback_evidence_seqs(
        worker_submission,
        steps,
        reviewable_seqs,
    )
    required_step_intents = _reviewable_action_specs(steps)
    task_category = task_category_from_candidate(candidate)
    return {
        "candidate_id": candidate.get("candidate_id", "") or "",
        "status": candidate.get("status", "") or "",
        "site": candidate.get("site", "") or "",
        "task_category": task_category,
        "goal": candidate.get("goal", "") or "",
        "keyword": candidate.get("keyword", "") or "",
        "steps": steps,
        "transition_records": _compact_transition_records(
            worker_submission,
            reviewable_seqs,
        ),
        "feedback_evidence": _compact_feedback_evidence(
            worker_submission,
            evidence_seqs,
        ),
        "deterministic_step_validation": compact_step_evidence_verdicts(candidate),
        "required_step_intents": required_step_intents,
        "skill_metadata_evidence": dict(worker_submission.get("skill_metadata_evidence", {}) or {}),
        "worker_execution": _compact_worker_execution(worker_submission),
        "commander_review": dict(candidate.get("review", {}) or {}),
        "promotion_enabled": allow_promotion,
    }


def _llm_review_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.application.model_policy import commander_model_name

    model_name = commander_model_name(
        "VISION_RECIPE_CRITIC_MODEL",
        "VISION_WORKER_REVIEW_MODEL",
    )
    request_timeout = get_settings().recipe.critic_timeout_sec
    promotion_policy = _promotion_policy(bool(payload.get("promotion_enabled")))
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
                "A press_key, go_back, close_current_tab, or switch_tab step may be fixed only as a contextual "
                "follow-up to the immediately preceding successful action. It must have either a ready transition "
                "record or a contract-missing record with deterministic visual-change evidence, and the review must "
                "supply a transition contract. Never promote a global rule such as always press Enter or always go back. "
                "Scroll remains reasoning because its distance and stopping point depend on the current screen. "
                "required_step_intents is an output-completeness contract, not a replay recommendation. Return "
                "exactly one skill_metadata.step_intents item for every listed seq/action. Never omit a non-reusable "
                "step; classify it as replay_mode=reasoning. If critic_correction is present, fix every listed "
                "contract error and return a new complete review. "
                "Choosing a filter category and applying filters are current-request/current-state decisions unless "
                "an explicit runtime slot or condition fully determines them. Parameterized filter text requires a "
                "named runtime input and must not reuse the exploration value when that input is missing. "
                "Judge every action independently. Overall task success does not make a wrong click reusable. Compare "
                "deterministic_step_validation first. A step with eligible=false is blocked and must be reasoning. "
                "feedback_evidence with the transition_observation for the same seq/action_seq; classify no-op clicks, "
                "unrelated navigation, and actions without evidence for expected_after as reasoning. "
                "Compile transition_records into transition_contracts keyed by recorded action seq for every "
                "fixed/parameterized click, type, key, back, or tab action. Use only "
                "OCR-verifiable cues that distinguish the post-action screen from feedback_evidence.before_marker_texts. "
                "Generic headers visible before and after the action are not ready cues. Use only outcomes supported by observations."
            )
        ),
        HumanMessage(content=_serialize_candidate_review_payload(payload)),
    ]
    from agent.application.run_context import invoke_with_metrics

    return _coerce_review(invoke_with_metrics(llm, messages, "recipe_critic"))


def _step_intent_contract_errors(
    payload: dict[str, Any],
    review: dict[str, Any],
) -> list[str]:
    """활성 승격 응답이 모든 대상 행동을 명시적으로 판정했는지 검사한다."""

    if (
        not payload.get("promotion_enabled")
        or review.get("decision") != "accept"
        or not review.get("promote_to_active_recipe")
    ):
        return []

    items = list((review.get("skill_metadata") or {}).get("step_intents") or [])
    by_seq: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            seq = int(item.get("seq"))
        except (TypeError, ValueError):
            continue
        by_seq.setdefault(seq, []).append(item)

    errors: list[str] = []
    for required in payload.get("required_step_intents") or []:
        seq = int(required["seq"])
        expected_action = str(required.get("action") or "")
        matches = by_seq.get(seq, [])
        if not matches:
            errors.append(f"missing seq={seq} action={expected_action}")
            continue
        if len(matches) != 1:
            errors.append(f"duplicate seq={seq} count={len(matches)}")
            continue
        actual_action = str(matches[0].get("action") or "")
        if actual_action != expected_action:
            errors.append(
                f"action mismatch seq={seq} expected={expected_action} actual={actual_action}"
            )
    return errors


def review_candidate(
    candidate: dict[str, Any],
    critic: CriticFn | None = None,
    allow_promotion: bool = False,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    payload = build_candidate_review_payload(candidate, allow_promotion=allow_promotion)
    try:
        invoke_critic = critic or _llm_review_candidate
        review = _coerce_review(invoke_critic(payload))
        errors = _step_intent_contract_errors(payload, review)
        if not errors:
            return review

        corrected_payload = dict(payload)
        corrected_payload["critic_correction"] = {
            "kind": "step_intent_contract",
            "errors": errors,
            "instruction": (
                "Return exactly one step_intent for every required seq/action. "
                "Use replay_mode=reasoning for steps that must not be replayed."
            ),
        }
        corrected_review = _coerce_review(invoke_critic(corrected_payload))
        corrected_errors = _step_intent_contract_errors(corrected_payload, corrected_review)
        if corrected_errors:
            return _fallback_review(
                "critic_step_intent_contract_failed: " + "; ".join(corrected_errors[:8])
            )
        return corrected_review
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
    review = ensure_review_task_category(
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
        promotion = apply_candidate_promotion(
            candidate,
            review,
            db_path=db_path,
        )

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

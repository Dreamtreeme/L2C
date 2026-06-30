"""반사 레시피 후보 비평가 게이트(RecipeCandidateReview).

이 모듈은 후보 증거(candidate evidence)를 포장하고 비평가(Critic)의 판정을
적용한다. 대상 품질, 일반화 가능성, 재사용 가능성 같은 의미 판단은 코드에서
직접 하지 않는다.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

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


def build_candidate_review_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    """후보 증거(candidate evidence)를 의미 판단 없이 비평가(Critic)에게 전달한다."""
    worker_submission = dict(candidate.get("payload", {}) or {})
    return {
        "candidate_id": candidate.get("candidate_id", "") or "",
        "status": candidate.get("status", "") or "",
        "site": candidate.get("site", "") or "",
        "goal": candidate.get("goal", "") or "",
        "keyword": candidate.get("keyword", "") or "",
        "steps": list(candidate.get("steps", []) or []),
        "transition_observations": list(worker_submission.get("transition_observations", []) or []),
        "skill_metadata_evidence": dict(worker_submission.get("skill_metadata_evidence", {}) or {}),
        "worker_submission": worker_submission,
        "commander_review": dict(candidate.get("review", {}) or {}),
        "review_task": (
            "Decide whether this candidate is reusable Reflex recipe evidence. "
            "Active recipe promotion is disabled in code, so promote_to_active_recipe is only a review signal. "
            "Use the worker evidence, skill metadata evidence, and commander review. If accepted, fill "
            "skill_metadata with when_to_use, inputs, step_intents, verification, and fallback conditions. "
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
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0).with_structured_output(RecipeCandidateReview)
    messages = [
        SystemMessage(
            content=(
                "You are the Reflex Recipe Critic. The script layer only packaged evidence and did not judge "
                "semantic quality. Review the candidate and return only RecipeCandidateReview. Active recipe "
                "promotion is disabled in code, so use promote_to_active_recipe only as a recommendation signal. "
                "When accepting, write concise skill_metadata that explains when to use the recipe, which inputs "
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


def review_candidate(candidate: dict[str, Any], critic: CriticFn | None = None) -> dict[str, Any]:
    payload = build_candidate_review_payload(candidate)
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


def review_and_apply_candidate(
    candidate_id: str,
    db_path=None,
    critic: CriticFn | None = None,
) -> dict[str, Any]:
    from agent.recipe.candidate_store import RecipeCandidateStore

    candidate_store = RecipeCandidateStore(db_path)
    candidate = candidate_store.get_candidate(candidate_id)
    if not candidate:
        return _fallback_review(f"candidate_not_found: {candidate_id}")

    review = review_candidate(candidate, critic=critic)

    validation = {
        "review": review,
    }
    candidate_store.update_status(candidate_id, _status_for_review(review), validation=validation)
    out = dict(review)
    out["candidate_id"] = candidate_id
    return out


def _process_mode(mode: str | None) -> str:
    return "review"


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

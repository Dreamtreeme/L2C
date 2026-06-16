"""LLM Critic gate for pending Reflex recipe candidates.

This module only packages candidate evidence and applies the Critic verdict. It
intentionally avoids script-level semantic checks such as target quality,
generalizability, or whether a UI action is reusable.
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
    """Package candidate evidence for Critic LLM without judging semantic quality."""
    return {
        "candidate_id": candidate.get("candidate_id", "") or "",
        "status": candidate.get("status", "") or "",
        "site": candidate.get("site", "") or "",
        "goal": candidate.get("goal", "") or "",
        "keyword": candidate.get("keyword", "") or "",
        "steps": list(candidate.get("steps", []) or []),
        "worker_submission": dict(candidate.get("payload", {}) or {}),
        "commander_review": dict(candidate.get("review", {}) or {}),
        "review_task": (
            "Decide whether this candidate should be promoted to the active Reflex recipe table. "
            "Use the worker evidence and commander review. Return revise/reject with feedback when the candidate "
            "needs another worker run or should not be reused."
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
                "semantic quality. Review the candidate and return only RecipeCandidateReview. Set "
                "promote_to_active_recipe=true only when this candidate should become an active reusable recipe."
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
    allow_promote: bool = True,
) -> dict[str, Any]:
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    candidate_store = RecipeCandidateStore(db_path)
    candidate = candidate_store.get_candidate(candidate_id)
    if not candidate:
        return _fallback_review(f"candidate_not_found: {candidate_id}")

    review = review_candidate(candidate, critic=critic)
    promoted_count = 0
    if allow_promote and review.get("decision") == "accept" and review.get("promote_to_active_recipe"):
        promoted_count = RecipeStore(db_path).commit_recipe(
            candidate.get("site", "") or "unknown",
            candidate.get("goal", "") or "",
            list(candidate.get("steps", []) or []),
        )

    validation = {
        "review": review,
        "promoted_count": promoted_count,
        "allow_promote": allow_promote,
    }
    candidate_store.update_status(candidate_id, _status_for_review(review), validation=validation)
    out = dict(review)
    out["candidate_id"] = candidate_id
    out["promoted_count"] = promoted_count
    return out


def _process_mode(mode: str | None) -> str:
    return "promote" if str(mode or "").strip().lower() == "promote" else "review"


def process_recipe_candidates(
    limit: int = 5,
    mode: str = "review",
    status: str = "pending_replay",
    db_path=None,
    critic: CriticFn | None = None,
) -> dict[str, Any]:
    """Run the LLM Critic gate for stored candidates.

    The script layer only selects candidate rows by status and forwards each row
    to the Critic gate. It does not judge recipe quality itself.
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
            allow_promote=(normalized_mode == "promote"),
        )
        for candidate in candidates
    ]
    return {
        "mode": normalized_mode,
        "status": status,
        "requested_limit": safe_limit,
        "candidate_count": len(candidates),
        "processed_count": len(results),
        "promoted_count": sum(int(result.get("promoted_count") or 0) for result in results),
        "results": results,
    }

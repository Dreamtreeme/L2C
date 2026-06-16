"""Tools for reviewing stored Reflex recipe learning candidates."""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def review_recipe_candidates(mode: str = "review", limit: int = 5, status: str = "pending_replay") -> str:
    """
    Run the Critic LLM gate for stored Reflex recipe candidates.

    mode='review' records Critic decisions without active recipe promotion.
    mode='promote' allows Critic-approved candidates to be written to active recipes.
    """
    try:
        from agent.recipe.candidate_reviewer import process_recipe_candidates

        payload = process_recipe_candidates(limit=limit, mode=mode, status=status)
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("[review_recipe_candidates] Failed to process candidates: %s", e, exc_info=True)
        return json.dumps({"error": str(e), "mode": mode, "status": status}, ensure_ascii=False)
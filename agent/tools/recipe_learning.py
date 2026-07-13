"""저장된 Reflex 레시피 후보를 검토하는 도구."""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def review_recipe_candidates(mode: str = "review", limit: int = 5, status: str = "pending_replay") -> str:
    """
    저장된 Reflex 레시피 후보를 Critic LLM 게이트에 넘긴다.

    mode="review"는 판정만 기록하고, mode="promote"는 Critic이 승인한 ROI 재생 단계만
    활성 recipes 테이블에 저장한다.
    """
    try:
        from agent.recipe.candidate_reviewer import process_recipe_candidates

        payload = process_recipe_candidates(limit=limit, mode=mode, status=status)
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("[review_recipe_candidates] Failed to process candidates: %s", e, exc_info=True)
        return json.dumps({"error": str(e), "mode": "review", "status": status}, ensure_ascii=False)

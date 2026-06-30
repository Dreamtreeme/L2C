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

    Critic 판정은 점검용으로만 기록하고, 활성 레시피 승격은 수행하지 않는다.
    """
    try:
        from agent.recipe.candidate_reviewer import process_recipe_candidates

        # 활성 레시피 자동 승격은 제거했으므로 입력 mode와 무관하게 검토로만 처리한다.
        payload = process_recipe_candidates(limit=limit, mode="review", status=status)
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("[review_recipe_candidates] Failed to process candidates: %s", e, exc_info=True)
        return json.dumps({"error": str(e), "mode": "review", "status": status}, ensure_ascii=False)

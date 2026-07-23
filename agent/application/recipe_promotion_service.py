"""Reflex 후보 승격을 요청 경로와 분리해 영속 대기열에 등록한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.config import get_settings
from agent.recipe.candidate_store import RecipeCandidateStore
from agent.utils.logger import logger


def auto_promotion_enabled() -> bool:
    return get_settings().recipe.auto_promote


def schedule_recipe_candidate_promotion(
    candidate_id: str,
    db_path: str | Path | None = None,
) -> bool:
    """후보를 DB 대기열에 기록하고 사용자 요청은 기다리지 않는다."""

    resolved = str(candidate_id or "").strip()
    if not resolved or not auto_promotion_enabled():
        return False
    queued = RecipeCandidateStore(db_path).enqueue_review(resolved)
    if queued:
        logger.info("Recipe candidate promotion queued", candidate_id=resolved)
    return queued


def get_recipe_candidate_promotion_status(
    candidate_id: str,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """대기·처리·완료 상태를 기다리지 않고 조회한다."""

    resolved = str(candidate_id or "").strip()
    candidate = RecipeCandidateStore(db_path).get_candidate(resolved) if resolved else None
    if not candidate:
        return {
            "candidate_id": resolved,
            "status": "not_found",
            "review_attempts": 0,
            "review_error": "",
        }
    validation = dict(candidate.get("validation") or {})
    return {
        "candidate_id": resolved,
        "status": str(candidate.get("status") or ""),
        "review_attempts": int(candidate.get("review_attempts") or 0),
        "review_error": str(candidate.get("review_error") or ""),
        "promotion": dict(validation.get("promotion") or {}),
    }


__all__ = [
    "auto_promotion_enabled",
    "get_recipe_candidate_promotion_status",
    "schedule_recipe_candidate_promotion",
]

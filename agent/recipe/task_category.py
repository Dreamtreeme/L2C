"""Reflex 레시피 작업 카테고리 정규화 유틸리티."""

from __future__ import annotations

from typing import Any

from agent.recipe.text_utils import normalize_text


DEFAULT_SEARCH_TASK_CATEGORY = "검색"


def normalize_task_category(value: Any) -> str:
    """LLM이 정한 작업 카테고리를 비교 가능한 문자열로만 정규화한다."""

    return normalize_text(value).casefold()


def task_category_matches(requested: Any, recorded: Any) -> bool:
    """요청 카테고리가 있으면 저장된 카테고리와 정확히 맞아야 한다."""

    requested_category = normalize_task_category(requested)
    if not requested_category:
        return True
    recorded_category = normalize_task_category(recorded)
    if not recorded_category:
        return False
    return requested_category == recorded_category

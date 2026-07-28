"""Reflex replay에 쓰는 화면 역할(page role) 유틸리티."""

from __future__ import annotations

from typing import Any

from agent.recipe.text_utils import normalize_text


_ALIASES = {
    "detail": "job_detail",
    "posting_detail": "job_detail",
    "jobdetail": "job_detail",
    "detail_tab": "job_detail",
    "side_panel_detail": "job_detail",
    "list": "search",
    "results": "search",
    "search_results": "search",
}


def normalize_page_role(value: Any) -> str:
    """LLM/코드가 남긴 page_role 값을 비교 가능한 이름으로 정규화한다."""

    role = normalize_text(value).casefold().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(role, role)


def page_role_matches(recorded: Any, current: Any) -> bool:
    """저장된 page_role과 현재 화면 page_role이 명확히 같을 때만 재생을 허용한다."""

    recorded_role = normalize_page_role(recorded)
    current_role = normalize_page_role(current)
    if not recorded_role or recorded_role == "unknown":
        return False
    if not current_role or current_role == "unknown":
        return False
    return recorded_role == current_role

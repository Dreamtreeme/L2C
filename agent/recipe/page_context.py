"""Reflex replay에 쓰는 화면 역할(page role) 유틸리티."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from agent.recipe.text_utils import normalize_text


_ALIASES = {
    "detail": "job_detail",
    "posting_detail": "job_detail",
    "jobdetail": "job_detail",
    "list": "search",
    "results": "search",
    "search_results": "search",
}


def normalize_page_role(value: Any) -> str:
    """LLM/코드가 남긴 page_role 값을 비교 가능한 이름으로 정규화한다."""

    role = normalize_text(value).casefold().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(role, role)


def page_role_family(value: Any) -> str:
    """같은 문서 위에 표시되는 홈과 검색 오버레이를 하나의 화면 계열로 묶는다."""

    role = normalize_page_role(value)
    if role in {"home", "search_overlay"}:
        return "home_overlay"
    return role


def page_role_matches(recorded: Any, current: Any) -> bool:
    """저장된 page_role과 현재 화면 page_role이 명확히 같을 때만 재생을 허용한다."""

    recorded_role = normalize_page_role(recorded)
    current_role = normalize_page_role(current)
    if not recorded_role or recorded_role == "unknown":
        return False
    if not current_role or current_role == "unknown":
        return False
    return page_role_family(recorded_role) == page_role_family(current_role)


def infer_page_role_from_url_and_texts(current_url: str, marker_texts: list[Any] | None = None) -> str:
    """URL과 OCR 텍스트만으로 현재 화면 역할을 보수적으로 추정한다."""

    parsed = urlparse(current_url or "")
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    if "/wd/" in path or "/job/" in path:
        return "job_detail"
    if "search" in path or "query=" in query:
        return "search"

    collapsed_text = "".join(normalize_text(text) for text in marker_texts or [])
    collapsed_text = re.sub(r"\s+", "", collapsed_text).casefold()
    if any(term in collapsed_text for term in ("검색어를입력", "인기검색어", "추천검색어")):
        return "search_overlay"

    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host and (parsed.path or "/") in {"", "/"}:
        return "home"
    return ""

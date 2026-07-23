"""지휘자와 수집 작업자가 공유하는 구조화된 수집 요청 계약."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agent.utils.model_dump import dump_model


class CollectionCountMode(str, Enum):
    """수집 개수를 해석하는 방식."""

    UNSPECIFIED = "unspecified"
    EXPLICIT = "explicit"
    VISIBLE_ALL = "visible_all"


class CollectionPurpose(str, Enum):
    """수집한 공고를 사용하는 목적."""

    LOOKUP = "lookup"
    COLLECT = "collect"
    COMPARE = "compare"
    TREND = "trend"


class JobSearchFilters(BaseModel):
    """사이트에서 적용하거나 공고 내용으로 검증할 검색 조건."""

    posted_date_expression: str = Field(default="", description="오늘, 지난달처럼 사용자가 말한 기간 표현")
    posted_from: str = Field(default="", description="명시적으로 확인된 시작일(YYYY-MM-DD)")
    posted_to: str = Field(default="", description="명시적으로 확인된 종료일(YYYY-MM-DD)")
    experience: str = Field(default="", description="경력 조건")
    location: str = Field(default="", description="근무 지역 조건")
    employment_type: str = Field(default="", description="고용 형태 조건")


class CollectionIntent(BaseModel):
    """사용자 요청에서 추출해 실행 계층 전체가 공유하는 수집 의도."""

    original_query: str = Field(default="", description="조건을 제거하지 않은 원래 사용자 요청")
    site: str = Field(default="", description="수집 대상 사이트 slug")
    search_keyword: str = Field(default="", description="사이트 검색창에 입력할 검색어")
    count_mode: CollectionCountMode = CollectionCountMode.UNSPECIFIED
    target_count: int = Field(default=0, ge=0, le=100)
    filters: JobSearchFilters = Field(default_factory=JobSearchFilters)
    freshness_required: bool = Field(default=False, description="최신 공고 확인이 필요한지 여부")
    require_job_content: bool = Field(
        default=True,
        description="주요업무 또는 자격요건이 확인된 공고만 완전 수집으로 인정할지 여부",
    )
    purpose: CollectionPurpose = CollectionPurpose.COLLECT
    analysis_goal: str = Field(default="", description="비교·트렌드 등 수집 이후 분석 목적")


def normalize_collection_intent(
    value: CollectionIntent | dict[str, Any] | None = None,
    *,
    original_query: str = "",
    site: str = "",
    search_keyword: str = "",
    target_count: int = 0,
) -> CollectionIntent:
    """구조화 요청과 기존 평면 인자를 하나의 일관된 계약으로 합친다."""

    if isinstance(value, CollectionIntent):
        data = dump_model(value)
    elif isinstance(value, dict):
        data = dict(value)
    else:
        data = {}

    data["original_query"] = str(data.get("original_query") or original_query or "").strip()
    data["site"] = str(data.get("site") or site or "").strip()
    data["search_keyword"] = str(data.get("search_keyword") or search_keyword or "").strip()
    try:
        normalized_count = max(0, min(100, int(data.get("target_count") or target_count or 0)))
    except (TypeError, ValueError):
        normalized_count = 0

    raw_mode = data.get("count_mode") or CollectionCountMode.UNSPECIFIED.value
    mode = raw_mode.value if isinstance(raw_mode, CollectionCountMode) else str(raw_mode)
    if mode not in {item.value for item in CollectionCountMode}:
        mode = CollectionCountMode.UNSPECIFIED.value
    if normalized_count > 0:
        mode = CollectionCountMode.EXPLICIT.value
    elif mode == CollectionCountMode.EXPLICIT.value:
        mode = CollectionCountMode.UNSPECIFIED.value
    if mode == CollectionCountMode.VISIBLE_ALL.value:
        normalized_count = 0

    data["count_mode"] = mode
    data["target_count"] = normalized_count
    return CollectionIntent(**data)


__all__ = [
    "CollectionCountMode",
    "CollectionIntent",
    "CollectionPurpose",
    "JobSearchFilters",
    "normalize_collection_intent",
]

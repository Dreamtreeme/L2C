"""지휘자와 수집 작업자가 공유하는 구조화된 수집 요청 계약."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from shared.schema.jd_schema import JobField


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

    model_config = ConfigDict(extra="forbid")

    posted_from: str = Field(
        default="", description="명시적으로 확인된 시작일(YYYY-MM-DD)"
    )
    posted_to: str = Field(
        default="", description="명시적으로 확인된 종료일(YYYY-MM-DD)"
    )
    experience: str = Field(default="", description="경력 조건")
    location: str = Field(default="", description="근무 지역 조건")
    employment_type: str = Field(default="", description="고용 형태 조건")


class CollectionIntent(BaseModel):
    """사용자 요청에서 추출해 실행 계층 전체가 공유하는 수집 의도."""

    model_config = ConfigDict(extra="forbid")

    original_query: str = Field(
        default="", description="조건을 제거하지 않은 원래 사용자 요청"
    )
    site: str = Field(default="", description="수집 대상 사이트 slug")
    search_keyword: str = Field(default="", description="사이트 검색창에 입력할 검색어")
    count_mode: CollectionCountMode = CollectionCountMode.VISIBLE_ALL
    target_count: int = Field(default=0, ge=0, le=100)
    filters: JobSearchFilters = Field(default_factory=JobSearchFilters)
    freshness_required: bool = Field(
        default=False, description="최신 공고 확인이 필요한지 여부"
    )
    purpose: CollectionPurpose = CollectionPurpose.COLLECT
    task_category: str = Field(
        default="검색",
        min_length=1,
        description="Reflex 레시피 조회에 사용하는 작업 분류",
    )
    required_fields: list[JobField] = Field(
        default_factory=list,
        description="공통 핵심 필드와 답변 근거 요구를 합쳐 반드시 확인할 공고 필드",
    )

    @field_validator("required_fields")
    @classmethod
    def unique_required_fields(
        cls,
        values: list[JobField],
    ) -> list[JobField]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def align_count_mode(self) -> "CollectionIntent":
        if self.target_count > 0:
            self.count_mode = CollectionCountMode.EXPLICIT
        elif self.count_mode in {
            CollectionCountMode.UNSPECIFIED,
            CollectionCountMode.EXPLICIT,
        }:
            self.count_mode = CollectionCountMode.VISIBLE_ALL
        return self


class CollectionResult(BaseModel):
    """수집 서비스가 지휘자와 E2E 도구에 반환하는 단일 결과 계약."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "partial", "failed"]
    message: str = ""
    error_code: str = ""
    site: str = ""
    site_name: str = ""
    search_keyword: str = ""
    task_category: str = ""
    target_count: int = 0
    collected_count: int = 0
    resolved_count: int = 0
    persisted_count: int = 0
    created_count: int = 0
    updated_count: int = 0
    rejected_count: int = 0
    persisted_items: list[dict[str, Any]] = Field(default_factory=list)
    observed_job_ids: list[int] = Field(default_factory=list)
    document_ids: list[int] = Field(default_factory=list)
    scope_exhausted: bool = False
    worker_finished: bool = False
    hit_recursion_limit: bool = False
    worker_run_id: str = ""


__all__ = [
    "CollectionCountMode",
    "CollectionIntent",
    "CollectionPurpose",
    "CollectionResult",
    "JobSearchFilters",
]

"""LLM과 백엔드가 공유하는 실행·근거 계약의 단일 기준."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EVIDENCE_FIELDS = (
    "company_name",
    "position",
    "job_category",
    "experience_text",
    "employment_type",
    "location",
    "posted_at",
    "tech_stack",
    "main_tasks",
    "requirements",
    "preferred",
    "benefits",
    "raw_ocr_text",
)

EvidenceField = Literal[
    "company_name",
    "position",
    "job_category",
    "experience_text",
    "employment_type",
    "location",
    "posted_at",
    "tech_stack",
    "main_tasks",
    "requirements",
    "preferred",
    "benefits",
    "raw_ocr_text",
]


class CollectionToolArguments(BaseModel):
    """지휘자가 비전 수집 작업자에게 전달할 수 있는 인자."""

    model_config = ConfigDict(extra="forbid")

    query: str = ""
    site: str = ""
    original_query: str = ""
    count_mode: Literal["unspecified", "explicit", "visible_all"] = "unspecified"
    target_count: int = Field(default=0, ge=0, le=100)
    posted_from: str = ""
    posted_to: str = ""
    experience: str = ""
    location: str = ""
    employment_type: str = ""
    freshness_required: bool = False
    purpose: Literal["lookup", "collect", "compare", "trend"] = "collect"
    analysis_goal: str = ""
    task_category: str = "검색"


class EvidenceDocument(BaseModel):
    """답변 모델에 전달하는 공고 근거 문서."""

    id: int
    url: str = ""
    company_name: str = ""
    position: str = ""
    job_category: str = ""
    experience_text: str = ""
    employment_type: str = ""
    location: str = ""
    posted_at: str = ""
    posted_at_text: str = ""
    tech_stack: str = ""
    main_tasks: str = ""
    requirements: str = ""
    preferred: str = ""
    benefits: str = ""
    raw_ocr_text: str = ""


__all__ = [
    "CollectionToolArguments",
    "EVIDENCE_FIELDS",
    "EvidenceDocument",
    "EvidenceField",
]

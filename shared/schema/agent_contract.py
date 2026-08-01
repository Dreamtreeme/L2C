"""LLM과 백엔드가 공유하는 실행·근거 계약의 단일 기준."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


JOB_COLLECTION_FIELDS = (
    "company_name",
    "position",
    "url",
    "job_category",
    "experience_level",
    "experience_text",
    "education",
    "employment_type",
    "location",
    "posted_at",
    "posted_at_text",
    "deadline",
    "tech_stack",
    "main_tasks",
    "requirements",
    "preferred",
    "benefits",
    "salary",
    "raw_ocr_text",
)

JobCollectionField = Literal[
    "company_name",
    "position",
    "url",
    "job_category",
    "experience_level",
    "experience_text",
    "education",
    "employment_type",
    "location",
    "posted_at",
    "posted_at_text",
    "deadline",
    "tech_stack",
    "main_tasks",
    "requirements",
    "preferred",
    "benefits",
    "salary",
    "raw_ocr_text",
]

JOB_COLLECTION_FIELD_LABELS: dict[str, str] = {
    "company_name": "회사명",
    "position": "직무명",
    "url": "공고 URL",
    "job_category": "직군",
    "experience_level": "경력 수준",
    "experience_text": "경력 조건 원문",
    "education": "학력",
    "employment_type": "고용 형태",
    "location": "근무지",
    "posted_at": "게시일",
    "posted_at_text": "게시일 원문",
    "deadline": "마감일",
    "tech_stack": "기술 스택",
    "main_tasks": "주요 업무",
    "requirements": "자격 요건",
    "preferred": "우대 사항",
    "benefits": "혜택 및 복지",
    "salary": "급여",
    "raw_ocr_text": "OCR 원문",
}

DEFAULT_JOB_COLLECTION_FIELDS: tuple[JobCollectionField, ...] = (
    "company_name",
    "position",
    "url",
    "main_tasks",
    "requirements",
)


class JobCollectionContract(BaseModel):
    """상세 판독부터 저장 검증까지 공유하는 공고 필드 계약."""

    model_config = ConfigDict(extra="forbid")

    required_fields: list[JobCollectionField] = Field(default_factory=list)

    @field_validator("required_fields")
    @classmethod
    def unique_required_fields(
        cls,
        values: list[JobCollectionField],
    ) -> list[JobCollectionField]:
        return list(dict.fromkeys(values))


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
    "DEFAULT_JOB_COLLECTION_FIELDS",
    "EVIDENCE_FIELDS",
    "EvidenceDocument",
    "EvidenceField",
    "JOB_COLLECTION_FIELDS",
    "JOB_COLLECTION_FIELD_LABELS",
    "JobCollectionContract",
    "JobCollectionField",
]

"""수집, 저장, 답변에서 공유하는 채용공고 데이터 계약."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JobField(StrEnum):
    """사용자 요청과 수집 근거에서 참조할 수 있는 공고 필드."""

    COMPANY_NAME = "company_name"
    POSITION = "position"
    URL = "url"
    JOB_CATEGORY = "job_category"
    EXPERIENCE_TEXT = "experience_text"
    EDUCATION = "education"
    EMPLOYMENT_TYPE = "employment_type"
    LOCATION = "location"
    POSTED_AT = "posted_at"
    POSTED_AT_TEXT = "posted_at_text"
    DEADLINE = "deadline"
    TECH_STACK = "tech_stack"
    MAIN_TASKS = "main_tasks"
    REQUIREMENTS = "requirements"
    PREFERRED = "preferred"
    BENEFITS = "benefits"
    SALARY = "salary"
    RAW_OCR_TEXT = "raw_ocr_text"


class JobReviewStatus(StrEnum):
    """누적한 상세 근거를 검토한 결과."""

    COMPLETE = "complete"
    NEEDS_MORE = "needs_more"
    SOURCE_INCOMPLETE = "source_incomplete"
    INVALID_TARGET = "invalid_target"


JOB_FIELDS: tuple[str, ...] = tuple(field.value for field in JobField)
JOB_IDENTITY_FIELDS: tuple[JobField, ...] = (
    JobField.COMPANY_NAME,
    JobField.POSITION,
    JobField.URL,
)
JOB_DETAIL_FIELDS: tuple[JobField, ...] = (
    JobField.TECH_STACK,
    JobField.MAIN_TASKS,
    JobField.REQUIREMENTS,
    JobField.PREFERRED,
    JobField.BENEFITS,
)


class JobPosting(BaseModel):
    """LLM 추출부터 SQLite 저장까지 유지하는 공고 사실."""

    model_config = ConfigDict(extra="forbid")

    company_name: str | None = Field(None, description="회사명")
    position: str | None = Field(None, description="포지션/직무명")
    url: str | None = Field(None, description="공고 상세 URL")
    job_category: str | None = Field(None, description="직군/카테고리")
    education: str | None = Field(None, description="학력 요건")
    employment_type: str | None = Field(None, description="고용 형태")
    location: str | None = Field(None, description="근무 위치")
    posted_at: str | None = Field(
        None,
        description="공고 게시일(YYYY-MM-DD, 화면에서 확인된 경우만)",
    )
    posted_at_text: str | None = Field(None, description="화면에 표시된 게시일 원문")
    deadline: str | None = Field(None, description="마감일")
    tech_stack: list[str] = Field(default_factory=list, description="기술스택 목록")
    main_tasks: list[str] = Field(default_factory=list, description="주요 업무")
    requirements: list[str] = Field(default_factory=list, description="자격 요건")
    preferred: list[str] = Field(default_factory=list, description="우대 사항")
    benefits: list[str] = Field(default_factory=list, description="복지 및 혜택")
    salary: str | None = Field(None, description="연봉 정보")
    source_platform: str | None = Field(None, description="수집 출처 플랫폼")
    raw_ocr_text: str | None = Field(None, description="누적한 원천 OCR 텍스트")
    content_hash: str | None = Field(None, description="공고 중복 후보 그룹 해시")
    evidence_hash: str | None = Field(None, description="출처 증거 무결성 해시")
    experience_min: int | None = Field(None, description="최소 경력 년수")
    experience_max: int | None = Field(None, description="최대 경력 년수")
    experience_text: str | None = Field(None, description="경력 정보 원문")

    @field_validator("posted_at", mode="before")
    @classmethod
    def keep_iso_posted_at(cls, value: object) -> str | None:
        """날짜 비교에 사용할 수 있는 ISO 게시일만 보존한다."""

        if value in (None, ""):
            return None
        text = str(value).strip()
        try:
            date.fromisoformat(text)
        except ValueError:
            return None
        return text


class JobCollectionEvidence(BaseModel):
    """공고 사실과 분리해 보존하는 화면 판독 및 출처 근거."""

    model_config = ConfigDict(extra="forbid")

    required_fields: list[JobField] = Field(default_factory=list)
    field_evidence: dict[JobField, str] = Field(default_factory=dict)
    screenshot_path: str = ""
    ocr_text_path: str = ""
    source_card_key: str = ""

    @field_validator("required_fields")
    @classmethod
    def unique_fields(cls, values: list[JobField]) -> list[JobField]:
        return list(dict.fromkeys(values))


class JobDraft(BaseModel):
    """작업자 그래프가 공고 검토 노드에 전달하는 누적 화면 근거."""

    model_config = ConfigDict(extra="forbid")

    url: str
    detail_key: str = ""
    raw_ocr_text: str
    required_fields: list[JobField] = Field(default_factory=list)
    screenshot_path: str = ""
    source_card_key: str = ""
    screen_count: int = Field(default=0, ge=0)
    last_action: str = ""
    transition_status: str = ""
    transition_reason: str = ""
    review_model_tier: Literal["lightweight", "primary"] = "lightweight"

    @field_validator("url", "raw_ocr_text")
    @classmethod
    def require_draft_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("공고 초안의 URL과 OCR 원문은 비어 있을 수 없습니다.")
        return normalized

    @field_validator("required_fields")
    @classmethod
    def unique_required_fields(cls, values: list[JobField]) -> list[JobField]:
        return list(dict.fromkeys(values))

    def fingerprint(self) -> str:
        """같은 상세 근거와 화면 종료 근거를 중복 검토하지 않도록 식별한다."""

        payload = "\x1f".join(
            (
                self.url,
                self.detail_key,
                self.raw_ocr_text,
                ",".join(field.value for field in self.required_fields),
                self.last_action,
                self.transition_status,
                self.transition_reason,
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()


class JobReview(BaseModel):
    """공고 초안을 같은 수집 계약으로 검토한 단일 판정."""

    model_config = ConfigDict(extra="forbid")

    detail_key: str = ""
    url: str
    status: JobReviewStatus
    posting: JobPosting = Field(default_factory=JobPosting)
    missing_fields: list[JobField] = Field(default_factory=list)
    field_evidence: dict[JobField, str] = Field(default_factory=dict)
    draft_fingerprint: str = ""
    model_tier: Literal["lightweight", "primary"] = "lightweight"
    reason: str = ""
    issues: list[str] = Field(default_factory=list)

    @field_validator("missing_fields")
    @classmethod
    def unique_missing_fields(cls, values: list[JobField]) -> list[JobField]:
        return list(dict.fromkeys(values))


class JobCapture(BaseModel):
    """비전 작업자가 상세 화면에서 확보한 정제 전 원문과 근거."""

    model_config = ConfigDict(extra="forbid")

    url: str
    raw_ocr_text: str
    evidence: JobCollectionEvidence = Field(default_factory=JobCollectionEvidence)

    @field_validator("url", "raw_ocr_text")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("상세 URL과 OCR 원문은 비어 있을 수 없습니다.")
        return normalized


class CollectedJob(BaseModel):
    """후처리를 마치고 저장할 수 있는 공고 한 건."""

    model_config = ConfigDict(extra="forbid")

    posting: JobPosting
    evidence: JobCollectionEvidence = Field(default_factory=JobCollectionEvidence)

    @model_validator(mode="after")
    def require_resolved_collection_fields(self) -> "CollectedJob":
        """식별 필드와 실행 계약 필드가 확인된 공고만 수집 완료로 인정한다."""

        required = dict.fromkeys(
            (*JOB_IDENTITY_FIELDS, *self.evidence.required_fields)
        )
        missing = [
            field.value
            for field in required
            if getattr(self.posting, field.value) in (None, "", [], {})
        ]
        if missing:
            raise ValueError("수집 완료 필드 누락: " + ", ".join(missing))
        return self


class StoredJob(JobPosting):
    """SQLite 식별자가 포함된 답변용 공고."""

    id: int


__all__ = [
    "CollectedJob",
    "JOB_DETAIL_FIELDS",
    "JOB_FIELDS",
    "JOB_IDENTITY_FIELDS",
    "JobCollectionEvidence",
    "JobCapture",
    "JobDraft",
    "JobField",
    "JobPosting",
    "JobReview",
    "JobReviewStatus",
    "StoredJob",
]

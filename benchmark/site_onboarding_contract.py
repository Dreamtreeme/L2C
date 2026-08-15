"""신규 사이트 적용 공수 실험의 입력과 증거 계약."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.schema.agent_contract import DEFAULT_JOB_COLLECTION_FIELDS
from shared.schema.jd_schema import JobField


class SiteOnboardingTask(BaseModel):
    """네 실험 세션에 공통으로 적용하는 수집 계약."""

    model_config = ConfigDict(extra="forbid")

    development_query: str = "백엔드 개발자"
    target_count: int = Field(default=2, ge=1, le=10)
    required_fields: list[JobField] = Field(
        default_factory=lambda: list(DEFAULT_JOB_COLLECTION_FIELDS)
    )
    acceptance_queries: list[str] = Field(
        default_factory=lambda: [
            "프론트엔드 개발자",
            "데이터 엔지니어",
            "안드로이드 개발자",
        ]
    )
    reserve_queries: list[str] = Field(
        default_factory=lambda: ["QA 엔지니어", "DevOps 엔지니어"]
    )
    time_limit_minutes: int = Field(default=90, ge=1, le=240)

    @field_validator("required_fields", "acceptance_queries", "reserve_queries")
    @classmethod
    def unique_values(cls, values: list) -> list:
        return list(dict.fromkeys(values))


class AcceptanceRun(BaseModel):
    """미사용 검색어 한 건의 실행·품질 증거."""

    model_config = ConfigDict(extra="forbid")

    query: str
    summary_path: str
    review_path: str = ""
    passed: bool
    runtime_sec: float = Field(ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0)


class FoundationPreparation(BaseModel):
    """사이트별 비교 시간에서 제외하는 Classic 공통 기반 준비 기록."""

    model_config = ConfigDict(extra="forbid")

    started_at: datetime
    finished_at: datetime
    changed_loc: int = Field(ge=0)
    modified_files: list[str] = Field(default_factory=list)
    acceptance_path: str
    notes: str = ""

    @property
    def duration_sec(self) -> float:
        return max(0.0, (self.finished_at - self.started_at).total_seconds())


class SiteAdaptationRecord(BaseModel):
    """사이트와 접근 방식 하나에 대한 Codex 작업 기록."""

    model_config = ConfigDict(extra="forbid")

    site: str
    homepage: str
    approach: Literal["classic", "vision"]
    baseline_sha: str
    result_sha: str = ""
    task_id: str = ""
    codex_model: str
    prompt_sha256: str
    started_at: datetime
    first_success_at: datetime | None = None
    finished_at: datetime | None = None
    status: Literal["running", "completed", "failed", "invalid"] = "running"
    human_interventions: list[str] = Field(default_factory=list)
    fix_iterations: list[str] = Field(default_factory=list)
    site_specific_changed_loc: int = Field(default=0, ge=0)
    common_runtime_changed_loc: int = Field(default=0, ge=0)
    modified_product_files: list[str] = Field(default_factory=list)
    locator_count: int = Field(default=0, ge=0)
    profile_line_count: int = Field(default=0, ge=0)
    acceptance_runs: list[AcceptanceRun] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def prompt_to_first_success_sec(self) -> float | None:
        if self.first_success_at is None:
            return None
        return max(0.0, (self.first_success_at - self.started_at).total_seconds())

    @property
    def prompt_to_acceptance_sec(self) -> float | None:
        if self.finished_at is None:
            return None
        return max(0.0, (self.finished_at - self.started_at).total_seconds())


class SiteAdaptationManifest(BaseModel):
    """공통 기준선과 네 격리 세션을 묶는 비교 입력."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 2
    baseline_sha: str
    prompt_sha256: str
    task_contract: SiteOnboardingTask = Field(default_factory=SiteOnboardingTask)
    foundation: FoundationPreparation
    records: list[SiteAdaptationRecord]


__all__ = [
    "AcceptanceRun",
    "FoundationPreparation",
    "SiteAdaptationManifest",
    "SiteAdaptationRecord",
    "SiteOnboardingTask",
]

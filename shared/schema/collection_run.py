"""비전 작업자 실행과 저장 단계 사이에서 공유하는 수집 결과 계약."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from shared.schema.feedback_schema import WorkerSubmission
from shared.schema.jd_schema import CollectedJob, JobCapture


class CollectionBatch(BaseModel):
    """비전 작업자가 검토한 결과와 실행 근거 묶음."""

    model_config = ConfigDict(extra="forbid")

    submission: WorkerSubmission
    job_captures: list[JobCapture] = Field(default_factory=list)
    collected_jobs: list[CollectedJob] = Field(default_factory=list)
    rejected_items: list[dict[str, Any]] = Field(default_factory=list)
    site_name: str


class PersistenceReport(BaseModel):
    """DB 저장과 검색 사전 연결이 끝난 공고의 결과."""

    model_config = ConfigDict(extra="forbid")

    stored_items: list[dict[str, Any]] = Field(default_factory=list)
    persisted_items: list[dict[str, Any]] = Field(default_factory=list)
    rejected_items: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def stored_count(self) -> int:
        return len(self.stored_items)

    @property
    def persisted_count(self) -> int:
        return len(self.persisted_items)

    @property
    def created_count(self) -> int:
        return sum(item.get("operation") == "created" for item in self.persisted_items)

    @property
    def updated_count(self) -> int:
        return sum(item.get("operation") == "updated" for item in self.persisted_items)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_items)

    def reject(self, item: dict[str, Any]) -> None:
        self.rejected_items.append(item)


class RecipeLearningResult(BaseModel):
    """자율탐색 기록을 레시피 후보로 넘긴 결과."""

    model_config = ConfigDict(extra="forbid")

    status: str
    reason: str = ""
    error: str = ""


class CollectionExperienceResult(BaseModel):
    """작업자 실행 기록과 레시피 후보를 별도로 저장한 결과."""

    model_config = ConfigDict(extra="forbid")

    recipe_learning: RecipeLearningResult


__all__ = [
    "CollectionBatch",
    "CollectionExperienceResult",
    "PersistenceReport",
    "RecipeLearningResult",
]

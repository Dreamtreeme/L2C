"""비전 작업자 실행과 저장 단계 사이에서 공유하는 수집 결과 계약."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from shared.schema.feedback_schema import WorkerSubmission
from shared.schema.jd_schema import CollectedJob


class CollectionBatch(BaseModel):
    """비전 작업자가 수집했지만 아직 DB에 저장하지 않은 한 실행 결과."""

    model_config = ConfigDict(extra="forbid")

    submission: WorkerSubmission
    collected_jobs: list[CollectedJob] = Field(default_factory=list)
    site_slug: str
    site_name: str


class PersistedCollection(BaseModel):
    """공고와 작업자 제출물을 저장한 결과."""

    model_config = ConfigDict(extra="forbid")

    submission: WorkerSubmission
    submission_id: str
    persistence: dict[str, Any] = Field(default_factory=dict)
    recipe_learning: dict[str, Any] = Field(default_factory=dict)


__all__ = ["CollectionBatch", "PersistedCollection"]

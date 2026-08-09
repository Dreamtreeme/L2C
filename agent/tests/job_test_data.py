"""정규 공고 계약을 사용하는 테스트 데이터 생성 도우미."""

from __future__ import annotations

from typing import Any

from shared.db.database import Database
from shared.schema.jd_schema import JobCollectionEvidence, JobPosting


def insert_job(
    db: Database,
    url: str,
    data: dict[str, Any],
    *,
    evidence: JobCollectionEvidence | None = None,
) -> int:
    return db.upsert(
        JobPosting.model_validate({"url": url, **data}),
        evidence=evidence,
    )


__all__ = ["insert_job"]

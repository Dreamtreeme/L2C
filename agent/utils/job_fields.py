"""공고 스키마 필드 조회와 수집 범위 계산."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.schema.jd_schema import (
    JOB_FIELDS,
    JOB_IDENTITY_FIELDS as SCHEMA_IDENTITY_FIELDS,
    JobField,
    JobPosting,
)
from shared.schema.collection_intent import CollectionIntent


def _has_job_field_value(value: Any) -> bool:
    """빈 값과 빈 컨테이너를 수집된 필드로 보지 않는다."""

    if isinstance(value, Mapping):
        return any(_has_job_field_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_job_field_value(item) for item in value)
    return bool(str(value or "").strip())


def job_field_value(job: JobPosting, field: str) -> Any:
    """공고 모델에서 비어 있지 않은 필드 값을 읽는다."""

    value = getattr(job, field, None)
    return value if _has_job_field_value(value) else None


def missing_job_fields(
    job: JobPosting,
    fields: list[str] | tuple[str, ...],
    *,
    unavailable_fields: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """공고에 없고 화면에서 미제공으로 확인되지 않은 필드를 반환한다."""

    unavailable = set(unavailable_fields)
    return [
        field
        for field in fields
        if field not in unavailable and job_field_value(job, field) is None
    ]


def normalize_job_collection_fields(values: Any) -> list[str]:
    """도구·설정 경계에서 받은 필드 목록을 공고 스키마 키로 제한한다."""

    if not isinstance(values, (list, tuple, set)):
        return []
    allowed = set(JOB_FIELDS)
    return list(
        dict.fromkeys(
            field for value in values if (field := str(value or "").strip()) in allowed
        )
    )


def required_job_fields(
    collection_intent: CollectionIntent,
    *,
    profile_fields: list[str] | tuple[str, ...] = (),
) -> list[JobField]:
    """식별 필드, 사이트 필드와 요청 필드를 하나의 수집 범위로 합친다."""

    fields = list(
        dict.fromkeys(
            field
            for group in (
                [field.value for field in SCHEMA_IDENTITY_FIELDS],
                profile_fields,
                [item.value for item in collection_intent.required_fields],
            )
            for field in normalize_job_collection_fields(group)
        )
    )
    return [JobField(field) for field in fields]


__all__ = [
    "job_field_value",
    "missing_job_fields",
    "normalize_job_collection_fields",
    "required_job_fields",
]

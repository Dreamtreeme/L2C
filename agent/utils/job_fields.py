"""공고 스키마 필드 조회와 수집 범위 계산."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.schema.jd_schema import (
    JOB_DETAIL_FIELDS as SCHEMA_DETAIL_FIELDS,
    JOB_FIELDS,
    JOB_IDENTITY_FIELDS as SCHEMA_IDENTITY_FIELDS,
)


IDENTITY_JOB_FIELDS: tuple[str, ...] = tuple(
    field.value for field in SCHEMA_IDENTITY_FIELDS
)
DETAIL_JOB_FIELDS: tuple[str, ...] = tuple(
    field.value for field in SCHEMA_DETAIL_FIELDS
)


def has_job_field_value(value: Any) -> bool:
    """빈 값과 빈 컨테이너를 수집된 필드로 보지 않는다."""

    if isinstance(value, Mapping):
        return any(has_job_field_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(has_job_field_value(item) for item in value)
    return bool(str(value or "").strip())


def job_field_value(job: Any, field: str) -> Any:
    """공고 모델 또는 직렬화된 공고에서 필드 값을 읽는다."""

    value = job.get(field) if isinstance(job, Mapping) else getattr(job, field, None)
    return value if has_job_field_value(value) else None


def missing_job_fields(
    job: Any,
    fields: list[str] | tuple[str, ...],
    *,
    unavailable_fields: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """공고에 없고 화면에서 미제공으로 확인되지도 않은 필드를 반환한다."""

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
            field
            for value in values
            if (field := str(value or "").strip()) in allowed
        )
    )


def required_job_fields(
    collection_intent: Mapping[str, Any] | None = None,
    *,
    profile_fields: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """식별 필드, 사이트 필드와 요청 필드를 하나의 수집 범위로 합친다."""

    intent = collection_intent if isinstance(collection_intent, Mapping) else {}
    return list(
        dict.fromkeys(
            field
            for group in (
                IDENTITY_JOB_FIELDS,
                profile_fields,
                intent.get("required_fields") or [],
            )
            for field in normalize_job_collection_fields(group)
        )
    )


__all__ = [
    "DETAIL_JOB_FIELDS",
    "IDENTITY_JOB_FIELDS",
    "has_job_field_value",
    "job_field_value",
    "missing_job_fields",
    "normalize_job_collection_fields",
    "required_job_fields",
]

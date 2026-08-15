"""공고 스키마 필드 조회와 수집 범위 계산."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.schema.jd_schema import (
    JOB_FIELDS,
    JobField,
    JobPosting,
)
from shared.schema.collection_intent import CollectionIntent
from shared.schema.agent_contract import (
    DEFAULT_JOB_COLLECTION_FIELDS,
    JOB_COLLECTION_FIELD_LABELS,
)


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
) -> list[str]:
    """공고 스키마에서 값이 비어 있는 필드를 반환한다."""

    return [
        field
        for field in fields
        if job_field_value(job, field) is None
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
) -> list[JobField]:
    """공통 핵심 필드와 사용자 요청 필드를 하나의 수집 범위로 합친다."""

    fields = list(
        dict.fromkeys(
            field
            for group in (
                [field.value for field in DEFAULT_JOB_COLLECTION_FIELDS],
                [item.value for item in collection_intent.required_fields],
            )
            for field in normalize_job_collection_fields(group)
        )
    )
    return [JobField(field) for field in fields]


def required_fields_from_intent(collection_intent: CollectionIntent) -> list[str]:
    """작업자 상태에 확정된 필수 필드 키를 읽는다."""

    return [field.value for field in collection_intent.required_fields]


def field_contract_items(fields: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    """프롬프트에 표시할 표준 필드 키와 한글 이름을 만든다."""

    return [
        {
            "field": field,
            "label": JOB_COLLECTION_FIELD_LABELS.get(field, field),
        }
        for field in normalize_job_collection_fields(fields)
    ]


__all__ = [
    "field_contract_items",
    "job_field_value",
    "missing_job_fields",
    "normalize_job_collection_fields",
    "required_fields_from_intent",
    "required_job_fields",
]

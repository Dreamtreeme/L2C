from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.schema.agent_contract import (
    JOB_COLLECTION_FIELDS,
    JobCollectionField,
)

_PERSISTENCE_FIELD_ALIASES: dict[str, list[str]] = {
    "company_name": ["company_name", "company", "companyName", "\ud68c\uc0ac\uba85", "\uae30\uc5c5\uba85", "\ud68c\uc0ac"],
    "position": ["position", "job_title", "jobTitle", "title", "role", "\uc9c1\ubb34\uba85", "\uacf5\uace0\uba85", "\ud3ec\uc9c0\uc158"],
    "url": ["url", "URL", "job_url", "jobUrl", "posting_url", "\uacf5\uace0url", "\uc0c1\uc138url"],
    "job_category": ["job_category", "jobCategory", "\uc9c1\uad70", "\uc9c1\ubb34 \uce74\ud14c\uace0\ub9ac"],
    "experience_level": ["experience_level", "experienceLevel", "\uacbd\ub825 \uc218\uc900"],
    "experience_text": ["experience_text", "experienceText", "\uacbd\ub825", "\uacbd\ub825\uc870\uac74"],
    "education": ["education", "\ud559\ub825", "\ud559\ub825\uc870\uac74"],
    "main_tasks": ["main_tasks", "mainTasks", "\uc8fc\uc694\uc5c5\ubb34", "\uc8fc\uc694 \uc5c5\ubb34", "\ub2f4\ub2f9\uc5c5\ubb34"],
    "requirements": ["requirements", "qualification", "qualifications", "\uc790\uaca9\uc694\uac74", "\uc790\uaca9 \uc694\uac74", "\uc790\uaca9\uc870\uac74"],
    "preferred": ["preferred", "preferences", "preferred_qualifications", "\uc6b0\ub300\uc0ac\ud56d", "\uc6b0\ub300 \uc0ac\ud56d"],
    "benefits": ["benefits", "welfare", "\ud61c\ud0dd", "\ubcf5\uc9c0", "\ud61c\ud0dd\ubc0f\ubcf5\uc9c0"],
    "tech_stack": ["tech_stack", "techStack", "skills", "\uae30\uc220\uc2a4\ud0dd", "\uae30\uc220 \uc2a4\ud0dd"],
    "location": ["location", "\uadfc\ubb34\uc9c0", "\uc9c0\uc5ed"],
    "employment_type": ["employment_type", "employmentType", "\uace0\uc6a9\ud615\ud0dc"],
    "posted_at": ["posted_at", "postedAt", "published_at", "publishedAt", "\uac8c\uc2dc\uc77c", "\ub4f1\ub85d\uc77c", "\uacf5\uace0\ub4f1\ub85d\uc77c"],
    "posted_at_text": ["posted_at_text", "postedAtText", "\uac8c\uc2dc\uc77c\uc6d0\ubb38", "\ub4f1\ub85d\uc77c\uc6d0\ubb38"],
    "deadline": ["deadline", "\ub9c8\uac10\uc77c", "\uc811\uc218\ub9c8\uac10"],
    "salary": ["salary", "\uc5f0\ubd09", "\uae09\uc5ec"],
    "raw_ocr_text": ["raw_ocr_text", "rawOcrText", "\uc6d0\ubb38", "ocr\uc6d0\ubb38"],
}

IDENTITY_JOB_FIELDS: tuple[JobCollectionField, ...] = (
    "company_name",
    "position",
    "url",
)
DETAIL_JOB_FIELDS: tuple[JobCollectionField, ...] = (
    "tech_stack",
    "main_tasks",
    "requirements",
    "preferred",
    "benefits",
)


def _first_present(job: dict[str, Any], aliases: list[str]) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in job.items()}
    for alias in aliases:
        if alias in job and job.get(alias) not in (None, "", [], {}):
            return job.get(alias)
        value = lowered.get(alias.lower())
        if value not in (None, "", [], {}):
            return value
    return None


def has_job_field_value(value: Any) -> bool:
    """빈 컨테이너와 공백 문자열을 수집된 값으로 보지 않는다."""

    if isinstance(value, Mapping):
        return any(has_job_field_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(has_job_field_value(item) for item in value)
    return bool(str(value or "").strip())


def job_field_value(job: Any, field: str) -> Any:
    """내부 표준 필드 또는 JobPosting 속성에서 값을 읽는다."""

    if isinstance(job, Mapping):
        value = job.get(field)
        return value if has_job_field_value(value) else None
    value = getattr(job, field, None)
    return value if has_job_field_value(value) else None


def job_field_presence(
    job: Any,
    fields: list[str] | tuple[str, ...],
) -> dict[str, bool]:
    """요청한 표준 필드별 값 존재 여부를 반환한다."""

    return {
        field: has_job_field_value(job_field_value(job, field))
        for field in fields
    }


def missing_job_fields(
    job: Any,
    fields: list[str] | tuple[str, ...],
    *,
    unavailable_fields: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """명시적으로 미제공인 필드를 제외하고 실제 누락 필드를 반환한다."""

    unavailable = set(unavailable_fields)
    presence = job_field_presence(job, fields)
    return [
        field
        for field in fields
        if field not in unavailable and not presence.get(field, False)
    ]


def normalize_job_collection_fields(
    values: Any,
) -> list[JobCollectionField]:
    """외부 상태의 필드 목록을 지원되는 표준 키로만 정규화한다."""

    if not isinstance(values, (list, tuple, set)):
        return []
    allowed = set(JOB_COLLECTION_FIELDS)
    normalized: list[JobCollectionField] = []
    seen: set[str] = set()
    for value in values:
        field = str(value or "").strip()
        if field not in allowed or field in seen:
            continue
        seen.add(field)
        normalized.append(field)  # type: ignore[arg-type]
    return normalized


def required_job_fields(
    collection_intent: Mapping[str, Any] | None = None,
    *,
    profile_fields: list[str] | tuple[str, ...] = (),
) -> list[JobCollectionField]:
    """수집 요청 전체에서 사용할 필수 필드 계약을 한 번 계산한다."""

    intent = collection_intent if isinstance(collection_intent, Mapping) else {}
    groups: list[Any] = [
        IDENTITY_JOB_FIELDS,
        profile_fields,
        intent.get("required_fields") or [],
    ]
    merged: list[JobCollectionField] = []
    for group in groups:
        for field in normalize_job_collection_fields(group):
            if field not in merged:
                merged.append(field)
    return merged


def deterministic_job_for_persistence(job: dict[str, Any]) -> dict[str, Any]:
    """외부 수집 JSON의 필드 별칭을 DB 표준 필드로 변환한다."""

    normalized = dict(job)
    for field, aliases in _PERSISTENCE_FIELD_ALIASES.items():
        if normalized.get(field) in (None, "", [], {}):
            value = _first_present(job, aliases)
            if value not in (None, "", [], {}):
                normalized[field] = value
    return normalized

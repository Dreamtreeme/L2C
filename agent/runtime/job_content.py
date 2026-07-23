"""채용공고 구조화 결과에 실제 직무 본문이 있는지 판정한다."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


JOB_CONTENT_ALIASES: dict[str, tuple[str, ...]] = {
    "main_tasks": ("main_tasks", "주요업무", "responsibilities"),
    "requirements": ("requirements", "자격요건", "qualifications"),
}


def _field_value(job: Any, aliases: tuple[str, ...]) -> Any:
    for name in aliases:
        if isinstance(job, Mapping) and name in job:
            return job[name]
        if not isinstance(job, Mapping) and hasattr(job, name):
            return getattr(job, name)
    return None


def _has_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_value(item) for item in value)
    return bool(str(value or "").strip())


def job_content_presence(job: Any) -> dict[str, bool]:
    """주요업무와 자격요건의 구조화 필드별 존재 여부를 반환한다."""

    return {
        field: _has_value(_field_value(job, aliases))
        for field, aliases in JOB_CONTENT_ALIASES.items()
    }


def has_meaningful_job_content(job: Any) -> bool:
    """주요업무 또는 자격요건 중 하나라도 수집됐는지 반환한다."""

    return any(job_content_presence(job).values())


__all__ = [
    "JOB_CONTENT_ALIASES",
    "has_meaningful_job_content",
    "job_content_presence",
]

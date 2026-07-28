"""수집 상태에서 공고 목록을 읽는 공통 계약."""

from __future__ import annotations

from typing import Any


JOB_LIST_KEYS = ("공고목록", "jobs", "job_list")
JOB_IDENTITY_KEYS = (
    "company_name",
    "회사명",
    "position",
    "직무명",
    "job_title",
    "url",
    "source_url",
)


def job_list_value(data: dict) -> Any:
    """지원하는 공고 목록 키 중 실제로 존재하는 값을 반환한다."""

    for key in JOB_LIST_KEYS:
        if key in data:
            return data.get(key)
    return None


def job_items(data: Any) -> list[dict[str, Any]]:
    """공고 목록 계약에 해당하는 항목만 반환한다."""

    if not isinstance(data, dict) or not data:
        return []

    def is_job_item(item: Any) -> bool:
        return (
            isinstance(item, dict)
            and bool(item)
            and any(item.get(key) not in (None, "") for key in JOB_IDENTITY_KEYS)
        )

    value = job_list_value(data)
    if isinstance(value, list):
        return [item for item in value if is_job_item(item)]
    if is_job_item(value):
        return [value]
    return [data] if value is None and is_job_item(data) else []


def job_count(data: Any) -> int:
    return len(job_items(data))


__all__ = [
    "JOB_IDENTITY_KEYS",
    "JOB_LIST_KEYS",
    "job_count",
    "job_items",
    "job_list_value",
]

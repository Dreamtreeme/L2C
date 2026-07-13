"""수집 상태에서 공고 목록을 읽는 공통 계약."""

from __future__ import annotations

from typing import Any


JOB_LIST_KEYS = ("공고목록", "jobs", "job_list")


def job_list_value(data: dict) -> Any:
    """지원하는 공고 목록 키 중 실제로 존재하는 값을 반환한다."""

    for key in JOB_LIST_KEYS:
        if key in data:
            return data.get(key)
    return None


__all__ = ["JOB_LIST_KEYS", "job_list_value"]

"""상세 OCR 전에 이미 수집한 공고 URL인지 확인한다."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent.application.job_lookup_service import find_job_id_by_url
from agent.runtime.job_collection import job_list_value


def duplicate_detail_skip_enabled() -> bool:
    raw = os.getenv("VISION_SKIP_EXISTING_JOB_DETAILS", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _url_key(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _current_run_contains_url(extracted_jd: Any, url: str) -> bool:
    if not isinstance(extracted_jd, dict):
        return False
    jobs = job_list_value(extracted_jd)
    if isinstance(jobs, dict):
        jobs = [jobs]
    if not isinstance(jobs, list):
        jobs = [extracted_jd] if extracted_jd else []
    target = _url_key(url)
    return any(
        isinstance(job, dict)
        and _url_key(job.get("url") or job.get("URL") or job.get("공고url")) == target
        for job in jobs
    )


def existing_job_url_trace(
    url: str,
    extracted_jd: Any,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """현재 실행과 DB에서 동일한 상세 URL을 찾는다."""

    target = _url_key(url)
    if not duplicate_detail_skip_enabled() or not target:
        return {"matched": False, "reason": "duplicate_skip_disabled_or_url_missing"}
    if _current_run_contains_url(extracted_jd, target):
        return {"matched": True, "source": "current_run", "url": target}

    if db_path is None:
        from shared.config import DB_PATH

        db_path = DB_PATH
    job_id = find_job_id_by_url(target, db_path=db_path)
    if job_id is not None:
        return {"matched": True, "source": "database", "job_id": job_id, "url": target}
    return {"matched": False, "reason": "url_not_found", "url": target}


__all__ = ["duplicate_detail_skip_enabled", "existing_job_url_trace"]

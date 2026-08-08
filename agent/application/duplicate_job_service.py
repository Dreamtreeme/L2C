"""현재 실행과 DB에서 이미 수집한 공고인지 확인한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.application.job_lookup_service import (
    find_job_ids_by_card_identities,
    find_job_id_by_url,
)
from agent.runtime.job_collection import job_items


def _url_key(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _current_run_contains_url(extracted_jd: Any, url: str) -> bool:
    if not isinstance(extracted_jd, dict):
        return False
    jobs = job_items(extracted_jd)
    target = _url_key(url)
    return any(
        isinstance(job, dict)
        and _url_key(job.get("url")) == target
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
    if not target:
        return {"matched": False, "reason": "url_missing"}
    if _current_run_contains_url(extracted_jd, target):
        return {"matched": True, "source": "current_run", "url": target}

    if db_path is None:
        from agent.config import get_settings

        db_path = get_settings().paths.db_path
    job_id = find_job_id_by_url(target, db_path=db_path)
    if job_id is not None:
        return {"matched": True, "source": "database", "job_id": job_id, "url": target}
    return {"matched": False, "reason": "url_not_found", "url": target}


def mark_existing_job_cards(
    queue: list[dict[str, Any]],
    current_url: str,
    *,
    db_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """목록에서 회사명과 제목이 정확히 같은 DB 공고를 처리 완료로 표시한다."""

    pending_items = [
        item
        for item in queue or []
        if isinstance(item, dict)
        and str(item.get("status") or "pending") == "pending"
    ]
    matched_job_ids = iter(
        find_job_ids_by_card_identities(
            [(item.get("company"), item.get("title")) for item in pending_items],
            current_url,
            db_path=db_path,
        )
    )
    updated: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for raw in queue or []:
        item = dict(raw)
        if str(item.get("status") or "pending") != "pending":
            updated.append(item)
            continue
        job_id = next(matched_job_ids, None)
        if job_id is None:
            updated.append(item)
            continue
        item.update(
            {
                "status": "skipped",
                "skip_reason": "existing_card_identity",
                "job_id": job_id,
            }
        )
        updated.append(item)
        traces.append(
            {
                "queue_id": str(item.get("queue_id") or ""),
                "company": str(item.get("company") or ""),
                "title": str(item.get("title") or ""),
                "job_id": job_id,
            }
        )
    return updated, traces


__all__ = [
    "existing_job_url_trace",
    "mark_existing_job_cards",
]

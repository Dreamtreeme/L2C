"""작업자가 검토한 공고를 SQLite와 검색 사전에 저장한다."""

from __future__ import annotations

import logging
from pathlib import Path

from agent.application.job_taxonomy_linker import JobTaxonomyLinker
from agent.config import get_settings
from shared.db.database import Database
from shared.schema.collection_run import CollectionBatch, PersistenceReport
from shared.schema.jd_schema import CollectedJob

logger = logging.getLogger(__name__)


def _store_job(
    index: int,
    collected_job: CollectedJob,
    *,
    db: Database,
    taxonomy_linker: JobTaxonomyLinker,
    report: PersistenceReport,
) -> None:
    posting = collected_job.posting
    evidence = collected_job.evidence
    url = str(posting.url)
    try:
        existed = db.exists(url)
        job_id = int(db.upsert(posting, evidence=evidence))
    except Exception as exc:
        logger.error("공고 저장 실패 index=%s: %s", index, exc)
        report.reject(
            {
                "index": index,
                "url": url,
                "issues": [f"persistence_error:{type(exc).__name__}"],
            }
        )
        return

    stored_item = {
        "job_id": job_id,
        "url": url,
        "company_name": posting.company_name or "",
        "position": posting.position or "",
        "operation": "updated" if existed else "created",
        "required_fields": [field.value for field in evidence.required_fields],
        "screenshot_path": evidence.screenshot_path,
    }
    report.stored_items.append(stored_item)
    try:
        taxonomy_linker.link_job(job_id)
    except Exception as exc:
        logger.warning("검색 사전 연결 실패 job_id=%s: %s", job_id, exc)
        report.reject(
            {
                "index": index,
                "job_id": job_id,
                "url": url,
                "stored": True,
                "issues": [f"taxonomy_index_failed:{type(exc).__name__}"],
            }
        )
        return

    report.persisted_items.append(stored_item)


def store_collection_batch(
    collection: CollectionBatch,
    *,
    db_path: str | Path | None = None,
) -> PersistenceReport:
    """검토가 끝난 공고를 UPSERT하고 검색 가능한 공고를 구분해 반환한다."""

    resolved_db_path = Path(db_path or get_settings().paths.db_path)
    report = PersistenceReport(rejected_items=list(collection.rejected_items))
    db = Database(resolved_db_path)
    taxonomy_linker = JobTaxonomyLinker(resolved_db_path)
    for index, collected_job in enumerate(collection.collected_jobs):
        _store_job(
            index,
            collected_job,
            db=db,
            taxonomy_linker=taxonomy_linker,
            report=report,
        )
    return report


__all__ = ["store_collection_batch"]

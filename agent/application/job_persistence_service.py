"""수집 완료 공고를 SQLite와 검색 사전에 저장한다."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.application.search_taxonomy_service import SearchTaxonomyService
from agent.config import get_settings
from agent.runtime.job_identity import url_with_source_card_key
from agent.runtime.site_context import looks_like_job_detail_url
from shared.db.database import Database
from shared.schema.collection_intent import CollectionIntent
from shared.schema.jd_schema import CollectedJob, JobCollectionEvidence, JobPosting

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedJob:
    index: int
    posting: JobPosting
    evidence: JobCollectionEvidence


@dataclass
class _PersistenceReport:
    submitted_count: int
    required_fields: list[str]
    stored_count: int = 0
    persisted_count: int = 0
    taxonomy_index_failed_count: int = 0
    created_count: int = 0
    updated_count: int = 0
    persisted_items: list[dict[str, Any]] = field(default_factory=list)
    rejected_items: list[dict[str, Any]] = field(default_factory=list)

    def reject(self, item: dict[str, Any]) -> None:
        self.rejected_items.append(item)

    def as_dict(self) -> dict[str, Any]:
        return {
            "submitted_count": self.submitted_count,
            "stored_count": self.stored_count,
            "persisted_count": self.persisted_count,
            "taxonomy_index_failed_count": self.taxonomy_index_failed_count,
            "created_count": self.created_count,
            "updated_count": self.updated_count,
            "persisted_items": self.persisted_items,
            "rejected_count": len(self.rejected_items),
            "rejected_items": self.rejected_items,
            "required_fields": self.required_fields,
        }


def _request_filter_issues(
    posting: JobPosting,
    collection_intent: CollectionIntent,
) -> list[str]:
    """수집 계획에 명시된 날짜 범위를 최종 공고에 적용한다."""

    filters = collection_intent.filters
    posted_from = filters.posted_from.strip()
    posted_to = filters.posted_to.strip()
    posted_at = str(posting.posted_at or "").strip()
    if (posted_from or posted_to or collection_intent.freshness_required) and not posted_at:
        return ["requested_evidence_missing:posted_at"]
    issues: list[str] = []
    if posted_from and posted_at < posted_from:
        issues.append("requested_filter_mismatch:posted_at_before_range")
    if posted_to and posted_at > posted_to:
        issues.append("requested_filter_mismatch:posted_at_after_range")
    return issues


def _prepare_job(
    index: int,
    collected_job: CollectedJob,
    collection_intent: CollectionIntent,
) -> tuple[_PreparedJob | None, dict[str, Any] | None]:
    posting = collected_job.posting
    evidence = collected_job.evidence
    url = str(posting.url or "").strip()
    if evidence.source_card_key and not looks_like_job_detail_url(url):
        url = url_with_source_card_key(url, evidence.source_card_key)
        posting = posting.model_copy(update={"url": url})

    issues = _request_filter_issues(posting, collection_intent)
    if issues:
        return None, {
            "index": index,
            "url": url,
            "company_name": posting.company_name or "",
            "position": posting.position or "",
            "issues": issues,
        }
    return _PreparedJob(index=index, posting=posting, evidence=evidence), None


def _store_job(
    prepared: _PreparedJob,
    *,
    db: Database,
    taxonomy_service: SearchTaxonomyService,
    report: _PersistenceReport,
) -> None:
    posting = prepared.posting
    url = str(posting.url)
    try:
        existed = db.exists(url)
        job_id = db.upsert(posting, evidence=prepared.evidence)
        report.stored_count += 1
        try:
            taxonomy_service.link_job(int(job_id))
        except Exception as exc:
            report.taxonomy_index_failed_count += 1
            report.reject(
                {
                    "index": prepared.index,
                    "job_id": int(job_id),
                    "url": url,
                    "company_name": posting.company_name or "",
                    "position": posting.position or "",
                    "issues": [f"taxonomy_index_failed:{type(exc).__name__}"],
                }
            )
            logger.warning(
                "[job_persistence] 검색 사전 연결 실패 job_id=%s: %s",
                job_id,
                exc,
            )
            return

        report.persisted_count += 1
        if existed:
            report.updated_count += 1
        else:
            report.created_count += 1
        report.persisted_items.append(
            {
                "job_id": int(job_id),
                "url": url,
                "company_name": posting.company_name or "",
                "position": posting.position or "",
                "operation": "updated" if existed else "created",
                "required_fields": [
                    field.value for field in prepared.evidence.required_fields
                ],
                "unavailable_fields": [
                    field.value for field in prepared.evidence.unavailable_fields
                ],
            }
        )
    except Exception as exc:
        logger.error(
            "[job_persistence] 공고 저장 실패 index=%s: %s",
            prepared.index,
            exc,
        )
        report.reject(
            {
                "index": prepared.index,
                "url": url,
                "issues": [f"persistence_error:{type(exc).__name__}"],
            }
        )


def persist_collected_jobs_with_report(
    collected_jobs: Sequence[CollectedJob],
    *,
    collection_intent: CollectionIntent,
) -> dict[str, Any]:
    """정규화된 공고를 저장하고 공고별 결과를 반환한다."""

    jobs = list(collected_jobs)
    report = _PersistenceReport(
        submitted_count=len(jobs),
        required_fields=[field.value for field in collection_intent.required_fields],
    )
    db_path = get_settings().paths.db_path
    db = Database(db_path)
    taxonomy_service = SearchTaxonomyService(db_path)

    for index, collected_job in enumerate(jobs):
        prepared, rejected = _prepare_job(index, collected_job, collection_intent)
        if rejected is not None:
            report.reject(rejected)
            continue
        if prepared is not None:
            _store_job(
                prepared,
                db=db,
                taxonomy_service=taxonomy_service,
                report=report,
            )
    return report.as_dict()


__all__ = ["persist_collected_jobs_with_report"]

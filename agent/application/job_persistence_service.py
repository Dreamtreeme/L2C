"""비전 작업자가 수집한 채용공고를 정규화하고 DB에 저장한다."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent.utils.model_dump import dump_model


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedJob:
    index: int
    url: str
    posting: Any
    data: dict[str, Any]
    screenshot_path: str | None
    ocr_text_path: str | None
    required_fields: list[str]
    unavailable_fields: list[str]


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

    def include_required_fields(self, values: list[str]) -> None:
        for value in values:
            if value not in self.required_fields:
                self.required_fields.append(value)

    def reject(self, item: dict[str, Any]) -> None:
        self.rejected_items.append(item)

    def as_dict(self) -> dict[str, Any]:
        return {
            "submitted_count": self.submitted_count,
            "stored_count": self.stored_count,
            "persisted_count": self.persisted_count,
            "taxonomy_index_failed_count": (
                self.taxonomy_index_failed_count
            ),
            "created_count": self.created_count,
            "updated_count": self.updated_count,
            "persisted_items": self.persisted_items,
            "rejected_count": len(self.rejected_items),
            "rejected_items": self.rejected_items,
            "required_fields": self.required_fields,
        }


def normalize_job_for_persistence(job: dict[str, Any]) -> dict[str, Any]:
    """상세 정제가 끝난 수집 결과를 DB 입력 형식으로 정규화한다."""

    from agent.utils.job_fields import deterministic_job_for_persistence

    return deterministic_job_for_persistence(job)


def _job_validation_issues(
    job_posting: Any,
    collection_intent: dict[str, Any],
    *,
    unavailable_fields: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """저장에 필요한 사실과 요청 조건을 검증하되 의미 유사도는 추측하지 않는다."""

    from agent.utils.job_fields import (
        missing_job_fields,
        required_job_fields,
    )

    issues: list[str] = []
    for missing_field in missing_job_fields(
        job_posting,
        required_job_fields(collection_intent),
        unavailable_fields=unavailable_fields,
    ):
        issues.append(f"required_field_missing:{missing_field}")

    filters = (
        collection_intent.get("filters")
        if isinstance(collection_intent, dict)
        else {}
    )
    filters = filters if isinstance(filters, dict) else {}
    posted_from = str(filters.get("posted_from") or "").strip()
    posted_to = str(filters.get("posted_to") or "").strip()
    date_required = bool(
        posted_from
        or posted_to
        or collection_intent.get("freshness_required")
    )
    posted_at = str(
        getattr(job_posting, "posted_at", None) or ""
    ).strip()
    if date_required and not posted_at:
        issues.append("requested_evidence_missing:posted_at")
    if posted_at and posted_from and posted_at < posted_from:
        issues.append("requested_filter_mismatch:posted_at_before_range")
    if posted_at and posted_to and posted_at > posted_to:
        issues.append("requested_filter_mismatch:posted_at_after_range")
    return issues


def _collected_jobs(extracted_jd: dict[str, Any]) -> list[Any]:
    jobs = next(
        (
            extracted_jd[key]
            for key in ("jobs", "공고목록", "job_list")
            if key in extracted_jd
        ),
        None,
    )
    if jobs is None:
        return [extracted_jd] if extracted_jd else []
    return jobs if isinstance(jobs, list) else [jobs]


def _intent_for_job(
    job: dict[str, Any],
    base_intent: dict[str, Any],
) -> dict[str, Any]:
    intent = dict(base_intent)
    if not intent.get("required_fields"):
        intent["required_fields"] = list(
            job.get("_collection_required_fields") or []
        )
    return intent


def _prepare_job(
    index: int,
    job: dict[str, Any],
    collection_intent: dict[str, Any],
) -> tuple[_PreparedJob | None, dict[str, Any] | None]:
    from agent.runtime.job_identity import url_with_source_card_key
    from agent.runtime.site_context import looks_like_job_detail_url
    from agent.utils.job_fields import normalize_job_collection_fields
    from agent.utils.preprocessor import Preprocessor

    page_exhausted = bool(job.get("_collection_page_exhausted"))
    unavailable_fields = (
        normalize_job_collection_fields(
            job.get("_collection_unavailable_fields")
        )
        if page_exhausted
        else []
    )
    normalized_job = normalize_job_for_persistence(job)
    url = str(normalized_job.get("url") or "").strip()
    if not url:
        logger.warning(
            "[job_persistence] Skipping job #%s (%s - %s): URL not collected",
            index,
            normalized_job.get("company_name", ""),
            normalized_job.get("position", ""),
        )
        return None, {
            "index": index,
            "issues": ["required_field_missing:url"],
        }

    try:
        card_key = str(
            normalized_job.get("_source_card_key")
            or job.get("_source_card_key")
            or ""
        ).strip()
        if card_key and not looks_like_job_detail_url(url):
            url = url_with_source_card_key(url, card_key)
        normalized_job["url"] = url
        raw_ocr_text = (
            str(normalized_job.get("raw_ocr_text") or "").strip()
            or None
        )
        posting = Preprocessor.process_raw_jd(
            normalized_job,
            raw_ocr_text=raw_ocr_text,
        )
        issues = _job_validation_issues(
            posting,
            collection_intent,
            unavailable_fields=unavailable_fields,
        )
        if issues:
            logger.warning(
                "[job_persistence] Rejected job #%s before persistence: %s",
                index,
                ", ".join(issues),
            )
            return None, {
                "index": index,
                "url": posting.url or url,
                "company_name": posting.company_name or "",
                "position": posting.position or "",
                "issues": issues,
            }

        persistence_data = dump_model(posting)
        required_fields = list(
            collection_intent.get("required_fields") or []
        )
        persistence_data["_collection_required_fields"] = required_fields
        persistence_data["_collection_unavailable_fields"] = list(
            unavailable_fields
        )
        persistence_data["_collection_page_exhausted"] = page_exhausted
        persistence_data["_collection_field_evidence"] = dict(
            job.get("_collection_field_evidence") or {}
        )
        return _PreparedJob(
            index=index,
            url=url,
            posting=posting,
            data=persistence_data,
            screenshot_path=(
                str(
                    normalized_job.get("_evidence_screenshot_path")
                    or ""
                ).strip()
                or None
            ),
            ocr_text_path=(
                str(
                    normalized_job.get("_evidence_ocr_text_path") or ""
                ).strip()
                or None
            ),
            required_fields=required_fields,
            unavailable_fields=list(unavailable_fields),
        ), None
    except Exception as exc:
        logger.error(
            "[job_persistence] Failed to prepare job #%s: %s",
            index,
            exc,
        )
        return None, {
            "index": index,
            "url": url,
            "issues": [f"persistence_error:{type(exc).__name__}"],
        }


def _store_job(
    prepared: _PreparedJob,
    *,
    db: Any,
    taxonomy_service: Any,
    report: _PersistenceReport,
) -> None:
    try:
        existed = db.exists(prepared.url)
        job_id = db.upsert(
            url=prepared.url,
            data=prepared.data,
            screenshot_path=prepared.screenshot_path,
            ocr_text_path=prepared.ocr_text_path,
        )
        report.stored_count += 1
        try:
            taxonomy_service.link_job(int(job_id))
        except Exception as exc:
            report.taxonomy_index_failed_count += 1
            logger.warning(
                "[job_persistence] Search taxonomy linking failed for job #%s: %s",
                job_id,
                exc,
            )
            report.reject(
                {
                    "index": prepared.index,
                    "job_id": int(job_id),
                    "url": prepared.posting.url or prepared.url,
                    "company_name": prepared.posting.company_name or "",
                    "position": prepared.posting.position or "",
                    "issues": [
                        f"taxonomy_index_failed:{type(exc).__name__}"
                    ],
                }
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
                "url": prepared.posting.url or prepared.url,
                "company_name": prepared.posting.company_name or "",
                "position": prepared.posting.position or "",
                "operation": "updated" if existed else "created",
                "required_fields": prepared.required_fields,
                "unavailable_fields": prepared.unavailable_fields,
            }
        )
        logger.info(
            "[job_persistence] Upserted job #%s: %s - %s",
            prepared.index,
            prepared.posting.company_name,
            prepared.posting.position,
        )
    except Exception as exc:
        logger.error(
            "[job_persistence] Failed to persist job #%s: %s",
            prepared.index,
            exc,
        )
        report.reject(
            {
                "index": prepared.index,
                "url": prepared.url,
                "issues": [f"persistence_error:{type(exc).__name__}"],
            }
        )


def persist_collected_data_with_report(
    extracted_jd: dict,
    collection_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """공고별 검증 결과와 저장 건수를 함께 반환한다."""

    from agent.application.search_taxonomy_service import (
        SearchTaxonomyService,
    )
    from agent.config import get_settings
    from shared.db.database import Database

    jobs = _collected_jobs(extracted_jd)
    base_intent = dict(collection_intent or {})
    report = _PersistenceReport(
        submitted_count=len(jobs),
        required_fields=list(base_intent.get("required_fields") or []),
    )
    db_path = get_settings().paths.db_path
    db = Database(db_path)
    taxonomy_service = SearchTaxonomyService(db_path)

    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or not job:
            report.reject(
                {"index": index, "issues": ["invalid_job_payload"]}
            )
            continue
        intent = _intent_for_job(job, base_intent)
        report.include_required_fields(
            list(intent.get("required_fields") or [])
        )
        prepared, rejected = _prepare_job(index, job, intent)
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


__all__ = [
    "normalize_job_for_persistence",
    "persist_collected_data_with_report",
]

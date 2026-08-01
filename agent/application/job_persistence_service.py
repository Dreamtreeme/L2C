"""비전 작업자가 수집한 채용공고를 정규화하고 DB에 저장한다."""

from __future__ import annotations

import logging
from typing import Any

from agent.runtime.job_collection import job_list_value
from agent.utils.model_dump import dump_model


logger = logging.getLogger(__name__)


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
    for field in missing_job_fields(
        job_posting,
        required_job_fields(collection_intent),
        unavailable_fields=unavailable_fields,
    ):
        issues.append(f"required_field_missing:{field}")

    filters = collection_intent.get("filters") if isinstance(collection_intent, dict) else {}
    filters = filters if isinstance(filters, dict) else {}
    posted_from = str(filters.get("posted_from") or "").strip()
    posted_to = str(filters.get("posted_to") or "").strip()
    date_required = bool(
        posted_from
        or posted_to
        or filters.get("posted_date_expression")
        or collection_intent.get("freshness_required")
    )
    posted_at = str(getattr(job_posting, "posted_at", None) or "").strip()
    if date_required and not posted_at:
        issues.append("requested_evidence_missing:posted_at")
    if posted_at and posted_from and posted_at < posted_from:
        issues.append("requested_filter_mismatch:posted_at_before_range")
    if posted_at and posted_to and posted_at > posted_to:
        issues.append("requested_filter_mismatch:posted_at_after_range")

    requested_fields = {
        "experience": "experience_text",
        "location": "location",
        "employment_type": "employment_type",
    }
    for request_field, job_field in requested_fields.items():
        if filters.get(request_field) and not str(getattr(job_posting, job_field, None) or "").strip():
            issues.append(f"requested_evidence_missing:{job_field}")
    return issues


def persist_collected_data_with_report(
    extracted_jd: dict,
    keyword: str,
    collection_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """공고별 검증 결과와 저장 건수를 함께 반환한다."""

    from agent.utils.preprocessor import Preprocessor
    from agent.application.search_taxonomy_service import SearchTaxonomyService
    from agent.runtime.job_identity import url_with_source_card_key
    from agent.runtime.site_context import looks_like_job_detail_url
    from shared.config import DB_PATH
    from shared.db.database import Database

    db = Database(DB_PATH)
    taxonomy_service = SearchTaxonomyService(DB_PATH)
    jobs = job_list_value(extracted_jd)
    if jobs is not None:
        job_list = jobs if isinstance(jobs, list) else [jobs]
    else:
        job_list = [extracted_jd] if extracted_jd else []

    stored_count = 0
    persisted_count = 0
    taxonomy_index_failed_count = 0
    created_count = 0
    updated_count = 0
    persisted_items: list[dict[str, Any]] = []
    rejected_items: list[dict[str, Any]] = []
    base_collection_intent = dict(collection_intent or {})
    report_required_fields = list(
        base_collection_intent.get("required_fields") or []
    )
    for index, job in enumerate(job_list):
        if not isinstance(job, dict) or not job:
            rejected_items.append({"index": index, "issues": ["invalid_job_payload"]})
            continue

        effective_intent = dict(base_collection_intent)
        if not effective_intent.get("required_fields"):
            effective_intent["required_fields"] = list(
                job.get("_collection_required_fields") or []
            )
        for field in effective_intent.get("required_fields") or []:
            if field not in report_required_fields:
                report_required_fields.append(field)
        page_exhausted = bool(
            job.get("_collection_page_exhausted")
        )
        from agent.utils.job_fields import normalize_job_collection_fields

        unavailable_fields = (
            normalize_job_collection_fields(
                job.get("_collection_unavailable_fields")
            )
            if page_exhausted
            else []
        )
        normalized_job = normalize_job_for_persistence(job)
        url = normalized_job.get("url") or job.get("url") or job.get("URL") or job.get("공고url")
        if not url:
            company_name = job.get("회사명", job.get("company_name", ""))
            position = job.get("직무명", job.get("position", ""))
            logger.warning(
                "[job_persistence] Skipping job #%s (%s - %s): URL not collected",
                index,
                company_name,
                position,
            )
            rejected_items.append({"index": index, "issues": ["required_field_missing:url"]})
            continue

        try:
            card_key = str(
                normalized_job.get("_source_card_key")
                or job.get("_source_card_key")
                or ""
            ).strip()
            if card_key and not looks_like_job_detail_url(str(url)):
                url = url_with_source_card_key(str(url), card_key)
            normalized_job["url"] = str(url).strip()
            raw_ocr_text = str(normalized_job.get("raw_ocr_text") or "").strip() or None
            job_posting = Preprocessor.process_raw_jd(
                normalized_job,
                raw_ocr_text=raw_ocr_text,
            )
            issues = _job_validation_issues(
                job_posting,
                effective_intent,
                unavailable_fields=unavailable_fields,
            )
            if issues:
                rejected_items.append(
                    {
                        "index": index,
                        "url": job_posting.url or str(url),
                        "company_name": job_posting.company_name or "",
                        "position": job_posting.position or "",
                        "issues": issues,
                    }
                )
                logger.warning(
                    "[job_persistence] Rejected job #%s before persistence: %s",
                    index,
                    ", ".join(issues),
                )
                continue
            existed = db.exists(str(url))
            screenshot_path = str(
                normalized_job.get("_evidence_screenshot_path") or ""
            ).strip() or None
            ocr_text_path = str(
                normalized_job.get("_evidence_ocr_text_path") or ""
            ).strip() or None
            persistence_data = dump_model(job_posting)
            persistence_data["_collection_required_fields"] = list(
                effective_intent.get("required_fields") or []
            )
            persistence_data["_collection_unavailable_fields"] = list(
                unavailable_fields
            )
            persistence_data["_collection_page_exhausted"] = page_exhausted
            persistence_data["_collection_field_evidence"] = dict(
                job.get("_collection_field_evidence") or {}
            )
            job_id = db.upsert(
                url=url,
                data=persistence_data,
                screenshot_path=screenshot_path,
                ocr_text_path=ocr_text_path,
            )
            stored_count += 1
            try:
                taxonomy_service.link_job(int(job_id))
            except Exception as exc:
                taxonomy_index_failed_count += 1
                logger.warning(
                    "[job_persistence] Search taxonomy linking failed for job #%s: %s",
                    job_id,
                    exc,
                )
                rejected_items.append(
                    {
                        "index": index,
                        "job_id": int(job_id),
                        "url": job_posting.url or str(url),
                        "company_name": job_posting.company_name or "",
                        "position": job_posting.position or "",
                        "issues": [
                            f"taxonomy_index_failed:{type(exc).__name__}"
                        ],
                    }
                )
                continue
            persisted_count += 1
            if existed:
                updated_count += 1
            else:
                created_count += 1
            persisted_items.append(
                {
                    "job_id": int(job_id),
                    "url": job_posting.url or str(url),
                    "company_name": job_posting.company_name or "",
                    "position": job_posting.position or "",
                    "operation": "updated" if existed else "created",
                    "required_fields": list(
                        effective_intent.get("required_fields") or []
                    ),
                    "unavailable_fields": list(unavailable_fields),
                }
            )
            logger.info(
                "[job_persistence] Upserted job #%s: %s - %s",
                index,
                job_posting.company_name,
                job_posting.position,
            )
        except Exception as exc:
            logger.error("[job_persistence] Failed to persist job #%s: %s", index, exc)
            rejected_items.append(
                {
                    "index": index,
                    "url": str(url),
                    "issues": [f"persistence_error:{type(exc).__name__}"],
                }
            )

    return {
        "submitted_count": len(job_list),
        "stored_count": stored_count,
        "persisted_count": persisted_count,
        "taxonomy_index_failed_count": taxonomy_index_failed_count,
        "created_count": created_count,
        "updated_count": updated_count,
        "persisted_items": persisted_items,
        "rejected_count": len(rejected_items),
        "rejected_items": rejected_items,
        "required_fields": report_required_fields,
    }


__all__ = [
    "normalize_job_for_persistence",
    "persist_collected_data_with_report",
]

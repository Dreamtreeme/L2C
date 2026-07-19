"""비전 작업자가 수집한 채용공고를 정규화하고 DB에 저장한다."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from agent.runtime.job_collection import job_list_value
from agent.utils.model_dump import dump_model


logger = logging.getLogger(__name__)


def normalization_mode() -> str:
    mode = os.getenv("VISION_JD_NORMALIZATION_MODE", "deterministic").strip().lower()
    return mode if mode in {"deterministic", "llm", "off"} else "deterministic"


def normalize_job_for_persistence(job: dict[str, Any], keyword: str = "") -> dict[str, Any]:
    """수집 원본을 설정된 방식으로 DB 입력 스키마에 맞춘다."""

    mode = normalization_mode()
    if mode == "off":
        return dict(job)
    if mode == "deterministic":
        from agent.utils.job_fields import deterministic_job_for_persistence

        return deterministic_job_for_persistence(job)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from agent.application.model_clients import get_structured_google_model
        from shared.schema.jd_schema import JobPosting

        model_name = os.getenv(
            "VISION_JD_NORMALIZATION_MODEL",
            os.getenv("VISION_WORKER_REVIEW_MODEL", "gemini-3.5-flash"),
        )
        llm = get_structured_google_model(model_name, JobPosting, temperature=0.0)
        messages = [
            SystemMessage(
                content=(
                    "Normalize one raw job posting collected by a vision worker into the JobPosting schema. "
                    "Read field names in any language, including Korean. Preserve the original job URL. "
                    "Use empty strings or empty lists for unknown fields; do not invent missing facts. "
                    "Do not compute content_hash or evidence_hash."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {"search_keyword": keyword, "raw_job": job},
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        ]
        from agent.application.run_context import invoke_with_metrics

        normalized = dump_model(
            invoke_with_metrics(llm, messages, "job_normalization")
        )
        raw_url = job.get("url") or job.get("URL") or job.get("공고url")
        if raw_url and not normalized.get("url"):
            normalized["url"] = raw_url
        normalized["_normalization_source"] = "llm"
        return normalized
    except Exception as exc:  # pragma: no cover - 공급자 실패는 원본 저장으로 폴백한다.
        logger.warning("[job_persistence] JD normalization failed; using raw job: %s", exc)
        fallback = dict(job)
        fallback["_normalization_source"] = "llm_failed"
        fallback["_normalization_error"] = str(exc)[:200]
        return fallback


def _job_validation_issues(job_posting: Any, collection_intent: dict[str, Any]) -> list[str]:
    """저장에 필요한 사실과 요청 조건을 검증하되 의미 유사도는 추측하지 않는다."""

    issues: list[str] = []
    for field in ("url", "company_name", "position"):
        if not str(getattr(job_posting, field, None) or "").strip():
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
    from shared.config import DB_PATH
    from shared.db.database import Database

    db = Database(DB_PATH)
    taxonomy_service = SearchTaxonomyService(DB_PATH)
    jobs = job_list_value(extracted_jd)
    if jobs is not None:
        job_list = jobs if isinstance(jobs, list) else [jobs]
    else:
        job_list = [extracted_jd] if extracted_jd else []

    persisted_count = 0
    created_count = 0
    updated_count = 0
    persisted_items: list[dict[str, Any]] = []
    rejected_items: list[dict[str, Any]] = []
    for index, job in enumerate(job_list):
        if not isinstance(job, dict) or not job:
            rejected_items.append({"index": index, "issues": ["invalid_job_payload"]})
            continue

        normalized_job = normalize_job_for_persistence(job, keyword=keyword)
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
            normalized_job["url"] = str(url).strip()
            raw_ocr_text = str(normalized_job.get("raw_ocr_text") or "").strip() or None
            job_posting = Preprocessor.process_raw_jd(
                normalized_job,
                raw_ocr_text=raw_ocr_text,
            )
            issues = _job_validation_issues(job_posting, collection_intent or {})
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
            job_id = db.upsert(
                url=url,
                data=dump_model(job_posting),
                screenshot_path=screenshot_path,
                ocr_text_path=ocr_text_path,
            )
            try:
                taxonomy_service.link_job(int(job_id))
            except Exception as exc:
                logger.warning(
                    "[job_persistence] Search taxonomy linking failed for job #%s: %s",
                    job_id,
                    exc,
                )
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
        "persisted_count": persisted_count,
        "created_count": created_count,
        "updated_count": updated_count,
        "persisted_items": persisted_items,
        "rejected_count": len(rejected_items),
        "rejected_items": rejected_items,
    }


def persist_collected_data(
    extracted_jd: dict,
    keyword: str,
    collection_intent: dict[str, Any] | None = None,
) -> int:
    """기존 호출자용 저장 건수 반환 인터페이스."""

    return int(
        persist_collected_data_with_report(
            extracted_jd,
            keyword,
            collection_intent=collection_intent,
        )["persisted_count"]
    )


__all__ = [
    "normalization_mode",
    "normalize_job_for_persistence",
    "persist_collected_data",
    "persist_collected_data_with_report",
]

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
                    "Do not compute content_hash."
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


def persist_collected_data(extracted_jd: dict, keyword: str) -> int:
    """수집 결과를 전처리한 뒤 채용공고 DB에 UPSERT한다."""

    from agent.utils.preprocessor import Preprocessor
    from shared.config import DB_PATH
    from shared.db.database import Database

    db = Database(DB_PATH)
    jobs = job_list_value(extracted_jd)
    if jobs is not None:
        job_list = jobs if isinstance(jobs, list) else [jobs]
    else:
        job_list = [extracted_jd] if extracted_jd else []

    persisted_count = 0
    for index, job in enumerate(job_list):
        if not isinstance(job, dict) or not job:
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
            continue

        try:
            normalized_job["url"] = str(url).strip()
            job_posting = Preprocessor.process_raw_jd(normalized_job)
            db.upsert(url=url, data=dump_model(job_posting))
            persisted_count += 1
            logger.info(
                "[job_persistence] Upserted job #%s: %s - %s",
                index,
                job_posting.company_name,
                job_posting.position,
            )
        except Exception as exc:
            logger.error("[job_persistence] Failed to persist job #%s: %s", index, exc)

    return persisted_count


__all__ = [
    "normalization_mode",
    "normalize_job_for_persistence",
    "persist_collected_data",
]

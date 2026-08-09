"""비전 작업자가 수집한 OCR 원문을 저장 가능한 공고로 정제한다."""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.application.job_normalization_service import complete_extracted_job
from agent.config import get_settings
from agent.llm.policy import lightweight_model_name
from agent.observability.run_context import (
    ModelRequestTimeout,
    RunCancelled,
    RunDeadlineExceeded,
    invoke_with_metrics,
    raise_if_cancelled,
)
from agent.prompts.detail_extraction import build_detail_extraction_system_prompt
from agent.runtime.job_identity import url_with_source_card_key
from agent.runtime.site_context import looks_like_job_detail_url
from agent.utils.job_fields import missing_job_fields
from agent.utils.logger import logger
from shared.schema.collection_intent import CollectionIntent
from shared.schema.collection_run import CollectionBatch, PostprocessedCollection
from shared.schema.jd_schema import CollectedJob, JobCapture, JobPosting


def detail_extraction_model_spec() -> str:
    return (
        get_settings().models.detail_final_extraction_model
        or lightweight_model_name()
    )


def get_detail_extraction_llm() -> Any:
    """상세 OCR 정제용 구조화 모델을 지연 초기화한다."""

    from agent.llm.clients import get_structured_google_model

    return get_structured_google_model(
        detail_extraction_model_spec(),
        JobPosting,
        temperature=0.0,
        execution_role="detail",
    )


def extract_job_from_capture(capture: JobCapture) -> JobPosting:
    """원문 수집 계약 한 건을 공고 스키마로 구조화한다."""

    evidence = capture.evidence
    messages = [
        SystemMessage(
            content=build_detail_extraction_system_prompt(
                "누적 OCR 본문에서 채용공고 1건을 JobPosting 스키마로 정리하십시오. "
                "OCR에 없는 사실은 만들지 말고 알 수 없는 필드는 비우십시오. "
                "required_fields는 OCR과 field_evidence를 확인해 채우십시오. "
                "카드 목록에서 추정한 회사명이나 직무명은 사실 근거로 사용하지 않습니다. "
                "unavailable_fields는 공고가 제공하지 않는 필드입니다. "
                "현재 상세 URL은 보존하십시오."
            )
        ),
        HumanMessage(
            content=json.dumps(
                {
                    "current_url": capture.url,
                    "required_fields": [
                        field.value for field in evidence.required_fields
                    ],
                    "field_evidence": {
                        field.value: value
                        for field, value in evidence.field_evidence.items()
                    },
                    "unavailable_fields": [
                        field.value for field in evidence.unavailable_fields
                    ],
                    "ocr_text": capture.raw_ocr_text,
                },
                ensure_ascii=False,
                indent=2,
            )
        ),
    ]
    started = time.perf_counter()
    response = invoke_with_metrics(
        get_detail_extraction_llm(),
        messages,
        "detail_extraction",
        stream=True,
    )
    posting = (
        response
        if isinstance(response, JobPosting)
        else JobPosting.model_validate(response)
    )
    logger.info(
        "Detail OCR final extraction completed",
        duration_sec=round(time.perf_counter() - started, 6),
        model=detail_extraction_model_spec(),
        ocr_chars=len(capture.raw_ocr_text),
    )
    return complete_extracted_job(
        posting,
        current_url=capture.url,
        raw_ocr_text=capture.raw_ocr_text,
    )


def _request_filter_issues(
    posting: JobPosting,
    collection_intent: CollectionIntent,
) -> list[str]:
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


def _missing_required_fields(
    capture: JobCapture,
    posting: JobPosting,
) -> list[str]:
    evidence = capture.evidence
    unavailable = [field.value for field in evidence.unavailable_fields]
    required = [field.value for field in evidence.required_fields]
    return missing_job_fields(
        posting,
        required,
        unavailable_fields=unavailable,
    )


def _postprocess_capture(
    capture: JobCapture,
    collection_intent: CollectionIntent,
) -> CollectedJob:
    posting = extract_job_from_capture(capture)
    url = str(posting.url or "").strip()
    if capture.evidence.source_card_key and not looks_like_job_detail_url(url):
        url = url_with_source_card_key(url, capture.evidence.source_card_key)
        posting = posting.model_copy(update={"url": url})

    issues = _request_filter_issues(posting, collection_intent)
    if issues:
        raise ValueError(",".join(issues))

    missing = _missing_required_fields(capture, posting)
    if missing:
        raise ValueError("required_field_extraction_incomplete:" + ",".join(missing))

    return CollectedJob(posting=posting, evidence=capture.evidence)


def postprocess_collection_batch(batch: CollectionBatch) -> PostprocessedCollection:
    """원문별 실패를 격리해 저장 가능한 공고와 거부 근거를 반환한다."""

    collected_jobs: list[CollectedJob] = []
    rejected_items: list[dict[str, Any]] = []
    for index, capture in enumerate(batch.job_captures):
        raise_if_cancelled()
        try:
            collected_jobs.append(
                _postprocess_capture(capture, batch.submission.collection_intent)
            )
        except (RunCancelled, RunDeadlineExceeded, ModelRequestTimeout):
            raise
        except Exception as exc:
            rejected_items.append(
                {
                    "index": index,
                    "url": capture.url,
                    "issues": [f"postprocessing_error:{type(exc).__name__}:{exc}"],
                }
            )
            logger.warning("공고 원문 후처리 실패 index=%s: %s", index, exc)

    return PostprocessedCollection(
        submission=batch.submission,
        collected_jobs=collected_jobs,
        rejected_items=rejected_items,
        site_name=batch.site_name,
    )


__all__ = [
    "detail_extraction_model_spec",
    "extract_job_from_capture",
    "get_detail_extraction_llm",
    "postprocess_collection_batch",
]

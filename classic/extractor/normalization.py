"""Classic DOM 원문을 공통 공고 계약으로 정제한다."""

from __future__ import annotations

from typing import Protocol

from agent.application.job_normalization_service import complete_extracted_job
from shared.schema.jd_schema import (
    CollectedJob,
    JobCollectionEvidence,
    JobField,
    JobPosting,
)

from classic.automation.sites.base import DomExtraction
from classic.extractor.llm_engine import LLMEngine


class DomJobNormalizer(Protocol):
    """DOM 원문 정제기의 실행 계약."""

    def normalize(
        self,
        extraction: DomExtraction,
        *,
        url: str,
        source_platform: str,
        required_fields: list[JobField],
    ) -> CollectedJob: ...


def normalize_dom_posting(
    extraction: DomExtraction,
    *,
    url: str,
    source_platform: str,
    engine: LLMEngine,
) -> JobPosting:
    """DOM 본문을 정제하고 화면에서 확정한 메타데이터를 결합한다."""

    full_text = str(extraction.get("full_text") or "").strip()
    if not full_text:
        raise ValueError("상세 페이지의 DOM 본문이 비어 있습니다.")
    posting = JobPosting.model_validate(engine.extract_from_text(full_text))
    posting = posting.model_copy(
        update={
            "company_name": posting.company_name or extraction.get("company_name"),
            "position": posting.position or extraction.get("position"),
        }
    )
    posting = complete_extracted_job(
        posting,
        current_url=url,
        raw_ocr_text=full_text,
    )
    if not posting.source_platform:
        posting = posting.model_copy(update={"source_platform": source_platform})
    return posting


class LLMDomJobNormalizer:
    """기존 경량 추출 모델을 사용하는 Classic 정제기."""

    def __init__(self, model_name: str | None = None) -> None:
        self.engine = LLMEngine(model_name)

    def normalize(
        self,
        extraction: DomExtraction,
        *,
        url: str,
        source_platform: str,
        required_fields: list[JobField],
    ) -> CollectedJob:
        posting = normalize_dom_posting(
            extraction,
            url=url,
            source_platform=source_platform,
            engine=self.engine,
        )
        return CollectedJob(
            posting=posting,
            evidence=JobCollectionEvidence(required_fields=required_fields),
        )


__all__ = [
    "DomJobNormalizer",
    "LLMDomJobNormalizer",
    "normalize_dom_posting",
]

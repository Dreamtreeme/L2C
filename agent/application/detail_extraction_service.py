"""누적한 상세 OCR을 채용공고 스키마로 정제한다."""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts.detail_extraction import build_detail_extraction_system_prompt
from agent.runtime.detail_runtime import detail_buffer_text
from agent.utils.logger import logger
from agent.utils.model_dump import dump_model


def detail_extraction_model_spec() -> str:
    from agent.config import get_settings
    from agent.llm.policy import lightweight_model_name

    return (
        get_settings().models.detail_final_extraction_model
        or lightweight_model_name()
    )


def get_detail_extraction_llm() -> Any:
    """상세 OCR 정제용 Gemini 구조화 모델을 지연 초기화한다."""

    from agent.llm.clients import get_structured_google_model
    from shared.schema.jd_schema import JobPosting

    return get_structured_google_model(
        detail_extraction_model_spec(),
        JobPosting,
        temperature=0.0,
        execution_role="detail",
    )


def extract_job_from_job_detail_buffer(state: dict, current_url: str) -> dict[str, Any]:
    """상태의 OCR 버퍼를 공고 한 건으로 정제한다."""

    from agent.runtime.job_field_contract import (
        detail_coverage_status,
        required_fields_from_state,
    )

    buffer = dict(state.get("job_detail_buffer", {}) or {})
    ocr_text = detail_buffer_text(buffer)
    if not ocr_text.strip():
        return {}
    required_fields = required_fields_from_state(state)
    coverage = detail_coverage_status(
        dict(state.get("job_detail_coverage", {}) or {}),
        required_fields,
    )
    messages = [
        SystemMessage(
            content=build_detail_extraction_system_prompt(
                "누적 OCR 본문에서 채용공고 1건을 JobPosting 스키마로 정리하십시오. "
                "OCR에 없는 사실은 만들지 말고, 알 수 없는 필드는 비우십시오. "
                "required_fields는 반드시 OCR과 field_evidence를 확인해 채우십시오. "
                "카드 목록에서 추정한 회사명이나 직무명은 사실 근거로 사용하지 않습니다. "
                "company_name과 position은 상세 OCR의 헤더 또는 본문 근거를 우선하십시오. "
                "unavailable_fields는 공고가 제공하지 않는 것으로 이미 판정된 필드이므로 "
                "추측해서 채우지 마십시오. 현재 상세 URL은 보존하십시오."
            )
        ),
        HumanMessage(
            content=json.dumps(
                {
                    "current_url": current_url,
                    "required_fields": required_fields,
                    "field_evidence": coverage["field_evidence"],
                    "unavailable_fields": coverage[
                        "unavailable_fields"
                    ],
                    "ocr_text": ocr_text,
                },
                ensure_ascii=False,
                indent=2,
            )
        ),
    ]
    started = time.perf_counter()
    llm = get_detail_extraction_llm()
    from agent.observability.run_context import invoke_with_metrics

    response = invoke_with_metrics(
        llm,
        messages,
        "detail_extraction",
        stream=True,
    )
    extracted = dump_model(response)
    duration = time.perf_counter() - started
    logger.info(
        "Detail OCR final extraction completed",
        duration_sec=round(duration, 6),
        model=detail_extraction_model_spec(),
        ocr_chars=len(ocr_text),
        ocr_lines=len(buffer.get("lines") or []),
    )
    extracted = {key: value for key, value in extracted.items() if value not in (None, "", [], {})}
    if current_url and not extracted.get("url"):
        extracted["url"] = current_url
    # 구조화 결과와 별개로 실제 누적 OCR과 대표 화면을 출처 증거로 보존합니다.
    extracted["raw_ocr_text"] = ocr_text
    screens = [str(item) for item in (buffer.get("screens") or []) if str(item)]
    screen_evidence = [
        dict(item)
        for item in (buffer.get("screen_evidence") or [])
        if isinstance(item, dict) and str(item.get("path") or "").strip()
    ]
    representative = next(
        (
            str(item["path"])
            for item in screen_evidence
            if int(item.get("added_lines") or 0) > 0
        ),
        screens[0] if screens else "",
    )
    if representative:
        extracted["_evidence_screenshot_path"] = representative
    return extracted


__all__ = [
    "detail_extraction_model_spec",
    "extract_job_from_job_detail_buffer",
    "get_detail_extraction_llm",
]

"""관찰 결과를 공고 수집 상태에 반영하는 순수 전이 노드."""

from __future__ import annotations

from typing import Any

from agent.graph.state import GraphState
from agent.graph.worker_state import job_detail_key_from_state
from agent.runtime.detail_runtime import detail_context_matches, update_job_detail_buffer
from agent.runtime.job_field_contract import (
    detail_coverage_matches,
    merge_job_detail_coverage,
)


def collection_node(state: GraphState) -> dict[str, Any]:
    """OCR 결과를 상세 버퍼에 합치고 수집 상태를 한 번만 갱신한다."""

    if not state.get("ocr_complete"):
        return {}

    current_url = str(state.get("current_url") or "")
    detail_key = job_detail_key_from_state(state)
    detail_buffer = update_job_detail_buffer(
        dict(state.get("job_detail_buffer", {}) or {}),
        list(state.get("current_markers") or []),
        current_url,
        str(state.get("current_screenshot") or ""),
        page_role=str(state.get("current_page_role") or ""),
        detail_key=detail_key,
    )
    detail_followup = dict(state.get("job_detail_followup", {}) or {})
    if detail_followup and not detail_context_matches(
        detail_followup,
        current_url,
        detail_key,
    ):
        detail_followup = {}
    return_to_job_results = dict(state.get("return_to_job_results", {}) or {})
    if return_to_job_results and return_to_job_results.get("url") != current_url:
        return_to_job_results = {}
    detail_coverage = dict(state.get("job_detail_coverage", {}) or {})
    if detail_context_matches(detail_buffer, current_url, detail_key):
        if not detail_coverage_matches(
            detail_coverage,
            current_url,
            detail_key,
        ):
            detail_coverage = merge_job_detail_coverage(
                {},
                {},
                state=state,
                current_url=current_url,
                detail_key=detail_key,
            )

    return {
        "job_detail_buffer": detail_buffer,
        "job_detail_coverage": detail_coverage,
        "job_detail_followup": detail_followup,
        "return_to_job_results": return_to_job_results,
    }


__all__ = ["collection_node"]

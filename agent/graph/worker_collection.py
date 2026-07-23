"""관찰 결과를 공고 수집 상태에 반영하는 순수 전이 노드."""

from __future__ import annotations

from typing import Any

from agent.graph.state import GraphState
from agent.graph.worker_state import detail_key_from_state
from agent.runtime.detail_runtime import update_detail_ocr_buffer


def apply_observation_node(state: GraphState) -> dict[str, Any]:
    """OCR 결과를 상세 버퍼에 합치고 수집 상태를 한 번만 갱신한다."""

    if not state.get("ocr_complete"):
        return {}

    current_url = str(state.get("current_url") or "")
    detail_buffer = update_detail_ocr_buffer(
        dict(state.get("detail_ocr_buffer", {}) or {}),
        list(state.get("current_markers") or []),
        current_url,
        str(state.get("current_screenshot") or ""),
        page_role=str(state.get("current_page_role") or ""),
        detail_key=detail_key_from_state(state),
    )
    detail_followup = dict(state.get("detail_followup_required", {}) or {})
    if detail_followup and detail_followup.get("url") != current_url:
        detail_followup = {}

    return {
        "detail_ocr_buffer": detail_buffer,
        "detail_followup_required": detail_followup,
    }


__all__ = ["apply_observation_node"]

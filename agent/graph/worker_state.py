"""작업자 그래프의 평면 상태에서 공통 값을 읽는 순수 헬퍼."""

from __future__ import annotations

from typing import Any

from agent.graph.state import GraphState
from agent.runtime.job_collection import job_list_value
from agent.runtime.site_context import infer_site_page_role


def extracted_job_count(extracted_jd: dict[str, Any]) -> int:
    jobs = job_list_value(extracted_jd)
    if isinstance(jobs, list):
        return len([job for job in jobs if isinstance(job, dict) and job])
    if isinstance(jobs, dict):
        return 1
    return 1 if extracted_jd else 0


def target_count_from_state(state: GraphState) -> int:
    params = state.get("recipe_params", {}) or {}
    try:
        return max(0, int(params.get("target_count") or 0))
    except (TypeError, ValueError):
        return 0


def count_mode_from_state(state: GraphState) -> str:
    params = state.get("recipe_params", {}) or {}
    raw = params.get("count_mode") or ""
    return str(getattr(raw, "value", raw)).strip().lower()


def infer_current_page_role(
    current_url: str,
    markers: list[dict[str, Any]],
) -> str:
    """현재 화면의 Reflex 적용 범위를 보수적으로 분류한다."""

    return infer_site_page_role(
        current_url,
        [
            marker.get("text")
            for marker in markers or []
            if isinstance(marker, dict)
        ],
    )


def detail_key_from_state(state: GraphState) -> str:
    """같은 URL의 패널형 상세 화면도 공고별로 OCR 버퍼를 분리한다."""

    card = dict(state.get("active_result_card", {}) or {})
    queue_id = str(card.get("queue_id") or "").strip()
    if queue_id:
        return queue_id
    company = str(card.get("company") or "").strip()
    title = str(card.get("title") or "").strip()
    return "|".join(part for part in (company, title) if part)


__all__ = [
    "count_mode_from_state",
    "detail_key_from_state",
    "extracted_job_count",
    "infer_current_page_role",
    "target_count_from_state",
]

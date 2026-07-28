"""작업자 그래프의 평면 상태에서 공통 값을 읽는 순수 헬퍼."""

from __future__ import annotations

from typing import Any

from agent.graph.state import GraphState
from agent.runtime.job_collection import job_count
from agent.runtime.site_context import infer_site_page_role


def extracted_job_count(extracted_jd: dict[str, Any]) -> int:
    return job_count(extracted_jd)


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


def job_detail_key_from_state(state: GraphState) -> str:
    """같은 URL의 패널형 상세 화면도 공고별로 OCR 버퍼를 분리한다."""

    card = dict(state.get("active_job_card", {}) or {})
    queue_id = str(card.get("queue_id") or "").strip()
    if queue_id:
        return queue_id
    company = str(card.get("company") or "").strip()
    title = str(card.get("title") or "").strip()
    return "|".join(part for part in (company, title) if part)


def return_to_job_results_for_url(
    state: GraphState,
    current_url: str | None = None,
) -> dict[str, Any]:
    """현재 상세 URL에서 완료 후 목록 복귀가 남아 있는지 반환한다."""

    pending = dict(state.get("return_to_job_results", {}) or {})
    pending_url = str(pending.get("url") or "").strip()
    resolved_url = str(
        current_url if current_url is not None else state.get("current_url") or ""
    ).strip()
    if not pending_url or pending_url != resolved_url:
        return {}
    return pending


__all__ = [
    "count_mode_from_state",
    "job_detail_key_from_state",
    "return_to_job_results_for_url",
    "extracted_job_count",
    "infer_current_page_role",
    "target_count_from_state",
]

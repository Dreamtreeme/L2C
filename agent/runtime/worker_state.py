"""작업자 상태에서 공통 값을 읽는 순수 질의 함수."""

from __future__ import annotations

from typing import Any

from agent.runtime.worker_contracts import WorkerState
from agent.runtime.job_collection import job_count
from agent.runtime.site_context import infer_site_page_role


def current_observation_matches_capture(state: WorkerState) -> bool:
    """OCR 관찰이 현재 캡처에 속하는지 확인한다."""

    observation = state["observation"]
    if not observation.get("ocr_complete"):
        return False
    current_capture_id = str(observation.get("current_capture_id") or "")
    ocr_capture_id = str(observation.get("ocr_capture_id") or "")
    return bool(
        current_capture_id and ocr_capture_id and current_capture_id == ocr_capture_id
    )


def collected_job_count(state: WorkerState) -> int:
    return job_count(state["collection"].get("collected_jobs", []))


def target_count_from_state(state: WorkerState) -> int:
    params = state["request"].get("recipe_params", {}) or {}
    try:
        return max(0, int(params.get("target_count") or 0))
    except (TypeError, ValueError):
        return 0


def count_mode_from_state(state: WorkerState) -> str:
    params = state["request"].get("recipe_params", {}) or {}
    raw = params.get("count_mode") or ""
    return str(getattr(raw, "value", raw)).strip().lower()


def infer_current_page_role(
    current_url: str,
    markers: list[dict[str, Any]],
) -> str:
    """현재 화면의 Reflex 적용 범위를 보수적으로 분류한다."""

    return infer_site_page_role(
        current_url,
        [marker.get("text") for marker in markers or [] if isinstance(marker, dict)],
    )


def return_to_job_results_for_url(
    state: WorkerState,
    current_url: str | None = None,
) -> dict[str, Any]:
    """현재 상세 URL에서 완료 후 목록 복귀가 남아 있는지 반환한다."""

    pending = dict(state["collection"].get("return_to_job_results", {}) or {})
    pending_url = str(pending.get("url") or "").strip()
    resolved_url = str(
        current_url
        if current_url is not None
        else state["observation"].get("current_url") or ""
    ).strip()
    if not pending_url or pending_url != resolved_url:
        return {}
    return pending


__all__ = [
    "count_mode_from_state",
    "current_observation_matches_capture",
    "return_to_job_results_for_url",
    "collected_job_count",
    "infer_current_page_role",
    "target_count_from_state",
]

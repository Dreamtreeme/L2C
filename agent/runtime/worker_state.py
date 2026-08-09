"""작업자 상태에서 공통 값을 읽는 순수 질의 함수."""

from __future__ import annotations

from typing import Any

from agent.runtime.worker_contracts import WorkerState
from agent.runtime.site_context import infer_site_page_role


def current_observation_ready(state: WorkerState) -> bool:
    """현재 화면 관찰에 OCR 결과가 준비됐는지 확인한다."""

    observation = state["observation"]
    return bool(
        observation.get("observation_id")
        and observation.get("current_screenshot")
        and observation.get("ocr_complete")
    )


def job_capture_count(state: WorkerState) -> int:
    return len(state["collection"].get("job_captures", []))


def target_count_from_state(state: WorkerState) -> int:
    return state["request"]["collection_intent"].target_count


def count_mode_from_state(state: WorkerState) -> str:
    return state["request"]["collection_intent"].count_mode.value


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
    "current_observation_ready",
    "return_to_job_results_for_url",
    "job_capture_count",
    "infer_current_page_role",
    "target_count_from_state",
]

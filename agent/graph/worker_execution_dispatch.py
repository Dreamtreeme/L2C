"""작업자 행동 요청을 물리 도구 또는 그래프 상태 변경으로 전달한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent.runtime.worker_contracts import WorkerState, WorkerStateUpdate
from agent.runtime.worker_data_services import WorkerDataServices
from agent.runtime.detail_runtime import detail_buffer_text, detail_evidence_screenshot
from agent.runtime.job_capture import store_job_capture
from agent.runtime.job_field_contract import (
    detail_coverage_status,
    merge_job_detail_coverage,
    required_fields_from_state,
)
from agent.runtime.job_identity import source_card_key
from agent.runtime.job_card_queue import (
    active_job_card,
    job_detail_key_from_state,
    normalize_job_card_queue,
)
from shared.schema.jd_schema import JobCapture, JobCollectionEvidence


def _empty_state_update() -> WorkerStateUpdate:
    return {}


@dataclass(frozen=True)
class StateActionOutcome:
    """상태 행동의 결과와 작업자 상태 변경."""

    result: dict[str, Any]
    state_update: WorkerStateUpdate = field(default_factory=_empty_state_update)


@dataclass(frozen=True)
class DetailReadingAssessment:
    """상세 화면 근거가 최종 정제를 실행할 만큼 모였는지 판정한 결과."""

    coverage: dict[str, Any]
    coverage_status: dict[str, Any]
    required_fields: list[str]


def dispatch_ui_action(
    action_name: str,
    args: dict[str, Any],
    get_bbox: Callable[[int], list[int]],
    *,
    action_tools: Any,
    current_url: str = "",
) -> dict[str, Any]:
    """마우스와 키보드를 사용하는 물리 행동 하나를 실행한다."""

    if action_name == "click_marker":
        return action_tools.click_marker(get_bbox(args["marker_id"]))
    if action_name == "focus_marker":
        return action_tools.focus_marker(get_bbox(args["marker_id"]))
    if action_name == "type_in_marker":
        return action_tools.type_in_marker(
            get_bbox(args["marker_id"]),
            args["text"],
        )
    if action_name == "scroll":
        marker_id = args.get("marker_id")
        bbox = get_bbox(marker_id) if marker_id is not None else None
        return action_tools.scroll(
            direction=args.get("direction", "down"),
            bbox=bbox,
            amount=args.get("amount", "page"),
        )
    if action_name == "press_key":
        return action_tools.press_key(args["key"])
    if action_name == "open_browser":
        return action_tools.open_browser(
            args["url"],
            current_url=current_url,
        )
    if action_name == "close_current_tab":
        return action_tools.close_current_tab()
    if action_name == "switch_tab":
        return action_tools.switch_tab(args["direction"])
    if action_name == "go_back":
        return action_tools.go_back()
    raise ValueError(f"Unknown UI action: {action_name}")


def _active_source_card_key(
    state: WorkerState,
    current_url: str,
) -> str:
    """상세 화면과 현재 카드 큐를 연결하는 로컬 식별자를 만든다."""

    active_card = active_job_card(
        list(state["collection"].get("job_card_queue", []) or [])
    )
    company = str(active_card.get("company") or "").strip()
    title = str(active_card.get("title") or "").strip()
    return source_card_key(current_url, company, title)


def _detail_followup(
    state: WorkerState,
    *,
    current_url: str,
    reason: str,
    missing_fields: list[str],
) -> dict[str, Any]:
    """같은 상세 화면의 추가 판독 횟수와 누락 필드를 기록한다."""

    detail_key = job_detail_key_from_state(state)
    previous = dict(state["collection"].get("job_detail_followup", {}) or {})
    same_detail = previous.get("url") == current_url or (
        detail_key and previous.get("detail_key") == detail_key
    )
    attempts = int(previous.get("attempts") or 0) + 1 if same_detail else 1
    return {
        "url": current_url,
        "detail_key": detail_key,
        "reason": reason,
        "missing_fields": list(missing_fields),
        "attempts": attempts,
    }


def _assess_detail_reading(
    args: dict[str, Any],
    state: WorkerState,
    current_url: str,
) -> DetailReadingAssessment:
    detail_key = job_detail_key_from_state(state)
    coverage = merge_job_detail_coverage(
        dict(state["collection"].get("job_detail_coverage", {}) or {}),
        args,
        state=state,
        current_url=current_url,
        detail_key=detail_key,
    )
    required_fields = required_fields_from_state(state)
    return DetailReadingAssessment(
        coverage=coverage,
        coverage_status=detail_coverage_status(coverage, required_fields),
        required_fields=required_fields,
    )


def _detail_retry_update(
    state: WorkerState,
    assessment: DetailReadingAssessment,
    *,
    current_url: str,
    reason: str,
    missing_fields: list[str],
) -> WorkerStateUpdate:
    return {
        "collection": {
            "job_detail_buffer": (
                state["collection"].get("job_detail_buffer") or {}
            ).copy(),
            "job_detail_coverage": assessment.coverage,
            "job_detail_followup": _detail_followup(
                state,
                current_url=current_url,
                reason=reason,
                missing_fields=missing_fields,
            ),
        }
    }


def _incomplete_detail_evidence_outcome(
    current_captures: list[JobCapture],
    state: WorkerState,
    assessment: DetailReadingAssessment,
    current_url: str,
) -> StateActionOutcome:
    missing = list(assessment.coverage_status["missing_fields"])
    return StateActionOutcome(
        result={
            "action": "finish_detail_reading",
            "status": "skipped",
            "result": (
                "Detail reading is not complete because required field "
                f"evidence is missing: {', '.join(missing)}"
            ),
            "reason": "required_field_evidence_incomplete",
            "required_fields": assessment.required_fields,
            "field_coverage": assessment.coverage_status,
        },
        state_update=_detail_retry_update(
            state,
            assessment,
            current_url=current_url,
            reason="required_field_evidence_incomplete",
            missing_fields=missing,
        ),
    )


def _empty_detail_outcome(
    current_captures: list[JobCapture],
) -> StateActionOutcome:
    return StateActionOutcome(
        result={
            "action": "finish_detail_reading",
            "status": "skipped",
            "result": "No accumulated detail OCR text to extract.",
            "reason": "empty_job_detail_buffer",
        },
        state_update={
            "collection": {
                "job_detail_buffer": {},
                "job_detail_coverage": {},
            }
        },
    )


def _prepare_job_capture(
    state: WorkerState,
    assessment: DetailReadingAssessment,
    *,
    current_url: str,
    raw_ocr_text: str,
) -> JobCapture:
    buffer = (state["collection"].get("job_detail_buffer") or {}).copy()
    return JobCapture(
        url=current_url,
        raw_ocr_text=raw_ocr_text,
        evidence=JobCollectionEvidence(
            required_fields=assessment.required_fields,
            unavailable_fields=assessment.coverage_status["unavailable_fields"],
            page_exhausted=bool(assessment.coverage_status["page_exhausted"]),
            field_evidence=dict(assessment.coverage_status["field_evidence"]),
            screenshot_path=detail_evidence_screenshot(buffer),
            source_card_key=_active_source_card_key(state, current_url),
        ),
    )


def _merge_completed_detail(
    current_captures: list[JobCapture],
    capture: JobCapture,
) -> StateActionOutcome:
    merged_captures = store_job_capture(current_captures, capture)
    return StateActionOutcome(
        result={
            "action": "finish_detail_reading",
            "status": "success",
            "result": (
                f"Detail OCR buffer captured (total_captures={len(merged_captures)})"
            ),
            "incoming_captures": 1,
            "total_captures": len(merged_captures),
            "ocr_chars": len(capture.raw_ocr_text),
        },
        state_update={
            "collection": {
                "job_captures": merged_captures,
                "job_detail_buffer": {},
                "job_detail_coverage": {},
                "job_detail_followup": {},
            }
        },
    )


def _finish_detail_reading(
    args: dict[str, Any],
    current_captures: list[JobCapture],
    *,
    current_url: str,
    state: WorkerState,
) -> StateActionOutcome:
    try:
        assessment = _assess_detail_reading(args, state, current_url)
        if assessment.coverage_status["missing_fields"]:
            return _incomplete_detail_evidence_outcome(
                current_captures,
                state,
                assessment,
                current_url,
            )

        buffer = (state["collection"].get("job_detail_buffer") or {}).copy()
        raw_ocr_text = detail_buffer_text(buffer)
        if not raw_ocr_text:
            return _empty_detail_outcome(current_captures)
        capture = _prepare_job_capture(
            state,
            assessment,
            current_url=current_url,
            raw_ocr_text=raw_ocr_text,
        )
        return _merge_completed_detail(
            current_captures,
            capture,
        )
    except Exception as exc:
        return StateActionOutcome(
            result={
                "action": "finish_detail_reading",
                "status": "error",
                "result": f"Failed to extract detail OCR buffer: {exc}",
            },
            state_update={},
        )


def _set_job_card_queue(
    args: dict[str, Any],
    current_captures: list[JobCapture],
    *,
    current_url: str,
    state: WorkerState,
    data_services: WorkerDataServices,
) -> StateActionOutcome:
    queue, memory = normalize_job_card_queue(args, state, current_url)
    queue, existing_cards = data_services.mark_existing_job_cards(
        queue,
        current_url,
    )
    selector_trace = dict(state["decision"].get("job_card_selection_trace", {}) or {})
    availability_source = (
        args if args.get("available_job_count") is not None else selector_trace
    )
    availability: dict[str, Any] = {}
    raw_available_count = availability_source.get("available_job_count")
    try:
        available_count = (
            int(raw_available_count) if raw_available_count is not None else -1
        )
    except (TypeError, ValueError):
        available_count = -1
    count_evidence = str(availability_source.get("count_evidence") or "").strip()[:160]
    if available_count >= len(queue) and count_evidence:
        availability = {
            "available_job_count": available_count,
            "count_evidence": count_evidence,
        }

    return StateActionOutcome(
        result={
            "action": "set_job_card_queue",
            "status": "success" if queue else "skipped",
            "result": (
                f"Job card queue stored: {len(queue)} card(s)."
                if queue
                else "No valid visible job cards were queued."
            ),
            "queued_count": len(queue),
            "queued_titles": [item.get("title", "") for item in queue],
            "existing_card_count": len(existing_cards),
            "existing_cards": existing_cards,
        },
        state_update={
            "collection": {
                "job_card_queue": queue,
                "job_results_memory": memory,
                "job_results_availability": availability,
            }
        },
    )


def dispatch_state_action(
    action_name: str,
    args: dict[str, Any],
    current_captures: list[JobCapture],
    *,
    current_url: str = "",
    state: WorkerState,
    data_services: WorkerDataServices,
) -> StateActionOutcome:
    """공고 추출과 카드 큐처럼 그래프 상태를 변경하는 행동을 실행한다."""

    if action_name == "finish_detail_reading":
        return _finish_detail_reading(
            args,
            current_captures,
            current_url=current_url,
            state=state,
        )
    if action_name == "set_job_card_queue":
        return _set_job_card_queue(
            args,
            current_captures,
            current_url=current_url,
            state=state,
            data_services=data_services,
        )
    raise ValueError(f"Unknown state action: {action_name}")


__all__ = [
    "StateActionOutcome",
    "dispatch_state_action",
    "dispatch_ui_action",
]

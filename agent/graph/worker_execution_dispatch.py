"""작업자 행동 요청을 물리 도구 또는 그래프 상태 변경으로 전달한다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.application.detail_extraction_service import (
    extract_job_from_job_detail_buffer,
)
from agent.runtime.worker_contracts import WorkerState
from agent.graph.worker_execution_policy import (
    merge_extracted_info,
    should_skip_job_update_without_detail_url,
)
from agent.runtime.worker_state import job_detail_key_from_state
from agent.application.duplicate_job_service import mark_existing_job_cards
from agent.runtime.job_collection import job_items
from agent.runtime.job_field_contract import (
    detail_coverage_status,
    merge_job_detail_coverage,
    required_fields_from_state,
)
from agent.runtime.job_identity import source_card_key
from agent.runtime.job_card_queue import (
    active_job_card,
    normalize_job_card_queue,
)
from agent.runtime.vision_worker_runtime import current_vision_worker_runtime
from agent.utils.job_fields import missing_job_fields


@dataclass(frozen=True)
class StateActionUpdate:
    """상태 행동이 선택적으로 갱신하는 작업자 필드."""

    job_card_queue: list[dict[str, Any]] | None = None
    job_results_memory: dict[str, Any] | None = None
    job_results_availability: dict[str, Any] | None = None
    job_detail_buffer: dict[str, Any] | None = None
    job_detail_coverage: dict[str, Any] | None = None
    job_detail_followup: dict[str, Any] | None = None


@dataclass(frozen=True)
class StateActionOutcome:
    """상태 행동의 결과와 작업자 상태 변경."""

    result: dict[str, Any]
    jobs: dict[str, Any]
    state_update: StateActionUpdate = field(default_factory=StateActionUpdate)


def dispatch_ui_action(
    action_name: str,
    args: dict[str, Any],
    get_bbox: Callable[[int], list[int]],
    *,
    current_url: str = "",
) -> dict[str, Any]:
    """마우스와 키보드를 사용하는 물리 행동 하나를 실행한다."""

    action_tools = current_vision_worker_runtime().get_action_tools()
    if action_name == "click_marker":
        return action_tools.click_marker(get_bbox(args["marker_id"]))
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
    if action_name == "close_browser":
        return action_tools.close_browser()
    if action_name == "close_current_tab":
        return action_tools.close_current_tab()
    if action_name == "switch_tab":
        return action_tools.switch_tab(args["direction"])
    if action_name == "go_back":
        return action_tools.go_back()
    raise ValueError(f"Unknown UI action: {action_name}")


def _attach_active_card_identity(
    data: dict[str, Any],
    *,
    state: WorkerState,
    current_url: str,
) -> dict[str, Any]:
    """상세 화면에서 추출한 공고에 목록 카드 식별 정보를 붙인다."""

    active_card = active_job_card(
        list(state.get("job_card_queue", []) or [])
    )
    company = str(active_card.get("company") or "").strip()
    title = str(active_card.get("title") or "").strip()
    card_key = source_card_key(current_url, company, title)
    if not card_key:
        return data

    extracted_jobs = job_items(data)

    for job in extracted_jobs:
        job.setdefault("_source_card_key", card_key)
        job.setdefault("_source_context_url", current_url)
        job.setdefault("_listing_company", company)
        job.setdefault("_listing_title", title)
    return data


def _update_extracted_info(
    args: dict[str, Any],
    current_jobs: dict[str, Any],
    *,
    current_url: str,
    state: WorkerState,
) -> StateActionOutcome:
    try:
        new_data = _attach_active_card_identity(
            json.loads(args["data_json"]),
            state=state,
            current_url=current_url,
        )
        detail_buffer = dict(state.get("job_detail_buffer", {}) or {})
        detail_buffer_active = bool(
            detail_buffer.get("url") == current_url
            and detail_buffer.get("lines")
        )
        if detail_buffer_active:
            return StateActionOutcome(
                result={
                    "action": "update_extracted_info",
                    "status": "skipped",
                    "result": (
                        "Skipped intermediate extraction: accumulated detail OCR must be "
                        "finalized with finish_detail_reading."
                    ),
                    "reason": "detail_buffer_requires_finish",
                },
                jobs=current_jobs,
            )
        if should_skip_job_update_without_detail_url(new_data, current_url):
            return StateActionOutcome(
                result={
                    "action": "update_extracted_info",
                    "status": "skipped",
                    "result": (
                        "Skipped extracted data merge: this site requires a detail URL "
                        "or an explicit job url in data_json"
                    ),
                    "reason": "job_update_requires_detail_url",
                },
                jobs=current_jobs,
            )

        merged_jobs, summary = merge_extracted_info(
            current_jobs,
            new_data,
            current_url=current_url,
        )
        return StateActionOutcome(
            result={
                "action": "update_extracted_info",
                "status": "success",
                "result": (
                    "Extracted data merged "
                    f"(incoming_jobs={summary['incoming_jobs']}, "
                    f"total_jobs={summary['total_jobs']}, "
                    f"fields={summary['fields']})"
                ),
            },
            jobs=merged_jobs,
        )
    except Exception as exc:
        return StateActionOutcome(
            result={
                "action": "update_extracted_info",
                "status": "error",
                "result": f"Failed to parse data_json: {exc}",
            },
            jobs=current_jobs,
        )


def _detail_followup(
    state: WorkerState,
    *,
    current_url: str,
    reason: str,
    missing_fields: list[str],
) -> dict[str, Any]:
    """같은 상세 화면의 추가 판독 횟수와 누락 필드를 기록한다."""

    detail_key = job_detail_key_from_state(state)
    previous = dict(state.get("job_detail_followup", {}) or {})
    same_detail = previous.get("url") == current_url or (
        detail_key
        and previous.get("detail_key") == detail_key
    )
    attempts = int(previous.get("attempts") or 0) + 1 if same_detail else 1
    return {
        "url": current_url,
        "detail_key": detail_key,
        "reason": reason,
        "missing_fields": list(missing_fields),
        "attempts": attempts,
    }


def _classify_extraction_missing_fields(
    extracted_job: dict[str, Any],
    required_fields: list[str],
    coverage_status: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """페이지를 더 읽을 수 있는 경우에만 구조화 누락을 재시도한다."""

    unavailable_fields = list(coverage_status["unavailable_fields"])
    missing_fields = missing_job_fields(
        extracted_job,
        required_fields,
        unavailable_fields=unavailable_fields,
    )
    if not missing_fields or not coverage_status["page_exhausted"]:
        return unavailable_fields, missing_fields, []

    # 같은 OCR을 다시 정제해도 화면 근거는 늘지 않는다. 원문과 누락 목록을
    # 보존한 채 부분 결과를 확정해 반복 호출을 막는다.
    unavailable_fields.extend(
        field
        for field in missing_fields
        if field not in unavailable_fields
    )
    return unavailable_fields, [], missing_fields


def _finish_detail_reading(
    args: dict[str, Any],
    current_jobs: dict[str, Any],
    *,
    current_url: str,
    state: WorkerState,
) -> StateActionOutcome:
    try:
        detail_key = job_detail_key_from_state(state)
        coverage = merge_job_detail_coverage(
            dict(state.get("job_detail_coverage", {}) or {}),
            args,
            state=state,
            current_url=current_url,
            detail_key=detail_key,
        )
        required_fields = required_fields_from_state(state)
        coverage_status = detail_coverage_status(
            coverage,
            required_fields,
        )
        if coverage_status["missing_fields"]:
            missing = list(coverage_status["missing_fields"])
            return StateActionOutcome(
                result={
                    "action": "finish_detail_reading",
                    "status": "skipped",
                    "result": (
                        "Detail reading is not complete because required field "
                        f"evidence is missing: {', '.join(missing)}"
                    ),
                    "reason": "required_field_evidence_incomplete",
                    "required_fields": required_fields,
                    "field_coverage": coverage_status,
                },
                jobs=current_jobs,
                state_update=StateActionUpdate(
                    job_detail_buffer=dict(
                        state.get("job_detail_buffer", {}) or {}
                    ),
                    job_detail_coverage=coverage,
                    job_detail_followup=_detail_followup(
                        state,
                        current_url=current_url,
                        reason="required_field_evidence_incomplete",
                        missing_fields=missing,
                    ),
                ),
            )

        extraction_state = {
            **state,
            "job_detail_coverage": coverage,
        }
        extracted_job = extract_job_from_job_detail_buffer(
            extraction_state,
            current_url,
        )
        if not extracted_job:
            return StateActionOutcome(
                result={
                    "action": "finish_detail_reading",
                    "status": "skipped",
                    "result": "No accumulated detail OCR text to extract.",
                    "reason": "empty_job_detail_buffer",
                },
                jobs=current_jobs,
                state_update=StateActionUpdate(
                    job_detail_buffer={},
                    job_detail_coverage={},
                ),
            )

        (
            unavailable_fields,
            missing,
            extraction_missing_fields,
        ) = _classify_extraction_missing_fields(
            extracted_job,
            required_fields,
            coverage_status,
        )
        if missing:
            return StateActionOutcome(
                result={
                    "action": "finish_detail_reading",
                    "status": "skipped",
                    "result": (
                        "Final detail extraction did not produce all fields that had "
                        f"visible evidence: {', '.join(missing)}"
                    ),
                    "reason": "required_field_extraction_incomplete",
                    "required_fields": required_fields,
                    "missing_fields": missing,
                    "field_coverage": coverage_status,
                },
                jobs=current_jobs,
                state_update=StateActionUpdate(
                    job_detail_buffer=dict(
                        state.get("job_detail_buffer", {}) or {}
                    ),
                    job_detail_coverage=coverage,
                    job_detail_followup=_detail_followup(
                        state,
                        current_url=current_url,
                        reason="required_field_extraction_incomplete",
                        missing_fields=missing,
                    ),
                ),
            )

        if extraction_missing_fields:
            extracted_job["_collection_extraction_missing_fields"] = list(
                extraction_missing_fields
            )
        extracted_job["_collection_required_fields"] = required_fields
        extracted_job["_collection_unavailable_fields"] = unavailable_fields
        extracted_job["_collection_page_exhausted"] = bool(
            coverage_status["page_exhausted"]
        )
        extracted_job["_collection_field_evidence"] = dict(
            coverage_status["field_evidence"]
        )
        wrapped_job = _attach_active_card_identity(
            {"jobs": [extracted_job]},
            state=state,
            current_url=current_url,
        )
        extracted_job = wrapped_job["jobs"][0]
        merged_jobs, summary = merge_extracted_info(
            current_jobs,
            {"jobs": [extracted_job]},
            current_url=current_url,
        )
        return StateActionOutcome(
            result={
                "action": "finish_detail_reading",
                "status": "success",
                "result": (
                    "Detail OCR buffer extracted and merged "
                    f"(incoming_jobs={summary['incoming_jobs']}, "
                    f"total_jobs={summary['total_jobs']}, "
                    f"fields={summary['fields']})"
                ),
                "incoming_jobs": summary["incoming_jobs"],
                "total_jobs": summary["total_jobs"],
                "fields": summary["fields"],
                "extraction_missing_fields": extraction_missing_fields,
            },
            jobs=merged_jobs,
            state_update=StateActionUpdate(
                job_detail_buffer={},
                job_detail_coverage={},
                job_detail_followup={},
            ),
        )
    except Exception as exc:
        return StateActionOutcome(
            result={
                "action": "finish_detail_reading",
                "status": "error",
                "result": f"Failed to extract detail OCR buffer: {exc}",
            },
            jobs=current_jobs,
        )


def _set_job_card_queue(
    args: dict[str, Any],
    current_jobs: dict[str, Any],
    *,
    current_url: str,
    state: WorkerState,
) -> StateActionOutcome:
    queue, memory = normalize_job_card_queue(args, state, current_url)
    queue, existing_cards = mark_existing_job_cards(queue, current_url)
    selector_trace = dict(state.get("job_card_selection_trace", {}) or {})
    availability_source = (
        args
        if args.get("available_job_count") is not None
        else selector_trace
    )
    availability: dict[str, Any] = {}
    try:
        available_count = int(availability_source.get("available_job_count"))
        count_confidence = float(
            availability_source.get("count_confidence") or 0.0
        )
    except (TypeError, ValueError):
        available_count = -1
        count_confidence = 0.0
    count_evidence = str(
        availability_source.get("count_evidence") or ""
    ).strip()[:160]
    if (
        available_count >= len(queue)
        and count_confidence >= 0.8
        and count_evidence
    ):
        availability = {
            "available_job_count": available_count,
            "count_evidence": count_evidence,
            "count_confidence": count_confidence,
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
        jobs=current_jobs,
        state_update=StateActionUpdate(
            job_card_queue=queue,
            job_results_memory=memory,
            job_results_availability=availability,
        ),
    )


def dispatch_state_action(
    action_name: str,
    args: dict[str, Any],
    current_jobs: dict[str, Any],
    *,
    current_url: str = "",
    state: WorkerState | None = None,
) -> StateActionOutcome:
    """공고 추출과 카드 큐처럼 그래프 상태를 변경하는 행동을 실행한다."""

    state = state or {}
    if action_name == "update_extracted_info":
        return _update_extracted_info(
            args,
            current_jobs,
            current_url=current_url,
            state=state,
        )
    if action_name == "finish_detail_reading":
        return _finish_detail_reading(
            args,
            current_jobs,
            current_url=current_url,
            state=state,
        )
    if action_name == "set_job_card_queue":
        return _set_job_card_queue(
            args,
            current_jobs,
            current_url=current_url,
            state=state,
        )
    raise ValueError(f"Unknown state action: {action_name}")


__all__ = [
    "StateActionOutcome",
    "StateActionUpdate",
    "dispatch_state_action",
    "dispatch_ui_action",
]

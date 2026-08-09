"""작업자 행동 요청을 물리 도구 또는 그래프 상태 변경으로 전달한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent.runtime.worker_contracts import WorkerState, apply_worker_state_update
from agent.runtime.worker_data_services import WorkerDataServices
from agent.runtime.detail_runtime import detail_evidence_screenshot
from agent.runtime.job_collection import store_collected_job
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
from agent.utils.job_fields import missing_job_fields
from shared.schema.jd_schema import CollectedJob, JobCollectionEvidence, JobPosting


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
    collected_jobs: list[CollectedJob]
    state_update: StateActionUpdate = field(default_factory=StateActionUpdate)


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
    previous = dict(
        state["collection"].get("job_detail_followup", {}) or {}
    )
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
    extracted_job: JobPosting,
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
) -> StateActionUpdate:
    return StateActionUpdate(
        job_detail_buffer=dict(
            state["collection"].get("job_detail_buffer", {}) or {}
        ),
        job_detail_coverage=assessment.coverage,
        job_detail_followup=_detail_followup(
            state,
            current_url=current_url,
            reason=reason,
            missing_fields=missing_fields,
        ),
    )


def _incomplete_detail_evidence_outcome(
    current_jobs: list[CollectedJob],
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
        collected_jobs=current_jobs,
        state_update=_detail_retry_update(
            state,
            assessment,
            current_url=current_url,
            reason="required_field_evidence_incomplete",
            missing_fields=missing,
        ),
    )


def _extract_detail_job(
    state: WorkerState,
    assessment: DetailReadingAssessment,
    current_url: str,
    data_services: WorkerDataServices,
) -> JobPosting | None:
    extraction_state = apply_worker_state_update(
        state,
        {"collection": {"job_detail_coverage": assessment.coverage}},
    )
    return data_services.extract_job_detail(
        extraction_state,
        current_url,
    )


def _empty_detail_outcome(
    current_jobs: list[CollectedJob],
) -> StateActionOutcome:
    return StateActionOutcome(
        result={
            "action": "finish_detail_reading",
            "status": "skipped",
            "result": "No accumulated detail OCR text to extract.",
            "reason": "empty_job_detail_buffer",
        },
        collected_jobs=current_jobs,
        state_update=StateActionUpdate(
            job_detail_buffer={},
            job_detail_coverage={},
        ),
    )


def _incomplete_detail_extraction_outcome(
    current_jobs: list[CollectedJob],
    state: WorkerState,
    assessment: DetailReadingAssessment,
    current_url: str,
    missing_fields: list[str],
) -> StateActionOutcome:
    return StateActionOutcome(
        result={
            "action": "finish_detail_reading",
            "status": "skipped",
            "result": (
                "Final detail extraction did not produce all fields that had "
                f"visible evidence: {', '.join(missing_fields)}"
            ),
            "reason": "required_field_extraction_incomplete",
            "required_fields": assessment.required_fields,
            "missing_fields": missing_fields,
            "field_coverage": assessment.coverage_status,
        },
        collected_jobs=current_jobs,
        state_update=_detail_retry_update(
            state,
            assessment,
            current_url=current_url,
            reason="required_field_extraction_incomplete",
            missing_fields=missing_fields,
        ),
    )


def _prepare_completed_detail_job(
    extracted_job: JobPosting,
    state: WorkerState,
    assessment: DetailReadingAssessment,
    *,
    current_url: str,
    unavailable_fields: list[str],
    extraction_missing_fields: list[str],
) -> CollectedJob:
    buffer = dict(state["collection"].get("job_detail_buffer", {}) or {})
    return CollectedJob(
        posting=extracted_job,
        evidence=JobCollectionEvidence(
            required_fields=assessment.required_fields,
            unavailable_fields=unavailable_fields,
            extraction_missing_fields=extraction_missing_fields,
            page_exhausted=bool(assessment.coverage_status["page_exhausted"]),
            field_evidence=dict(assessment.coverage_status["field_evidence"]),
            screenshot_path=detail_evidence_screenshot(buffer),
            source_card_key=_active_source_card_key(state, current_url),
        ),
    )


def _merge_completed_detail(
    current_jobs: list[CollectedJob],
    collected_job: CollectedJob,
    *,
    extraction_missing_fields: list[str],
) -> StateActionOutcome:
    merged_jobs = store_collected_job(current_jobs, collected_job)
    fields = sorted(
        collected_job.posting.model_dump(
            exclude_none=True,
            exclude_defaults=True,
        )
    )
    return StateActionOutcome(
        result={
            "action": "finish_detail_reading",
            "status": "success",
            "result": (
                "Detail OCR buffer extracted and merged "
                f"(total_jobs={len(merged_jobs)}, fields={fields})"
            ),
            "incoming_jobs": 1,
            "total_jobs": len(merged_jobs),
            "fields": fields,
            "extraction_missing_fields": extraction_missing_fields,
        },
        collected_jobs=merged_jobs,
        state_update=StateActionUpdate(
            job_detail_buffer={},
            job_detail_coverage={},
            job_detail_followup={},
        ),
    )


def _finish_detail_reading(
    args: dict[str, Any],
    current_jobs: list[CollectedJob],
    *,
    current_url: str,
    state: WorkerState,
    data_services: WorkerDataServices,
) -> StateActionOutcome:
    try:
        assessment = _assess_detail_reading(args, state, current_url)
        if assessment.coverage_status["missing_fields"]:
            return _incomplete_detail_evidence_outcome(
                current_jobs,
                state,
                assessment,
                current_url,
            )

        extracted_job = _extract_detail_job(
            state,
            assessment,
            current_url,
            data_services,
        )
        if not extracted_job:
            return _empty_detail_outcome(current_jobs)

        unavailable_fields, missing_fields, extraction_missing_fields = (
            _classify_extraction_missing_fields(
                extracted_job,
                assessment.required_fields,
                assessment.coverage_status,
            )
        )
        if missing_fields:
            return _incomplete_detail_extraction_outcome(
                current_jobs,
                state,
                assessment,
                current_url,
                missing_fields,
            )

        completed_job = _prepare_completed_detail_job(
            extracted_job,
            state,
            assessment,
            current_url=current_url,
            unavailable_fields=unavailable_fields,
            extraction_missing_fields=extraction_missing_fields,
        )
        return _merge_completed_detail(
            current_jobs,
            completed_job,
            extraction_missing_fields=extraction_missing_fields,
        )
    except Exception as exc:
        return StateActionOutcome(
            result={
                "action": "finish_detail_reading",
                "status": "error",
                "result": f"Failed to extract detail OCR buffer: {exc}",
            },
            collected_jobs=current_jobs,
        )


def _set_job_card_queue(
    args: dict[str, Any],
    current_jobs: list[CollectedJob],
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
    selector_trace = dict(
        state["decision"].get("job_card_selection_trace", {}) or {}
    )
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
        collected_jobs=current_jobs,
        state_update=StateActionUpdate(
            job_card_queue=queue,
            job_results_memory=memory,
            job_results_availability=availability,
        ),
    )


def dispatch_state_action(
    action_name: str,
    args: dict[str, Any],
    current_jobs: list[CollectedJob],
    *,
    current_url: str = "",
    state: WorkerState | None = None,
    data_services: WorkerDataServices,
) -> StateActionOutcome:
    """공고 추출과 카드 큐처럼 그래프 상태를 변경하는 행동을 실행한다."""

    state = state or {}
    if action_name == "finish_detail_reading":
        return _finish_detail_reading(
            args,
            current_jobs,
            current_url=current_url,
            state=state,
            data_services=data_services,
        )
    if action_name == "set_job_card_queue":
        return _set_job_card_queue(
            args,
            current_jobs,
            current_url=current_url,
            state=state,
            data_services=data_services,
        )
    raise ValueError(f"Unknown state action: {action_name}")


__all__ = [
    "StateActionOutcome",
    "StateActionUpdate",
    "dispatch_state_action",
    "dispatch_ui_action",
]

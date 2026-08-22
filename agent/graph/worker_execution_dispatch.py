"""작업자 행동 요청을 물리 도구 또는 그래프 상태 변경으로 전달한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent.runtime.worker_contracts import WorkerState, WorkerStateUpdate
from agent.runtime.worker_data_services import WorkerDataServices
from agent.runtime.detail_runtime import (
    detail_buffer_text,
    detail_evidence_screenshot,
)
from agent.runtime.job_identity import source_card_key
from agent.runtime.job_card_queue import (
    active_job_card,
    job_detail_key_from_state,
    normalize_job_card_queue,
)
from shared.schema.jd_schema import JobDraft, JobReviewStatus


def _empty_state_update() -> WorkerStateUpdate:
    return {}


@dataclass(frozen=True)
class StateActionOutcome:
    """상태 행동의 결과와 작업자 상태 변경."""

    result: dict[str, Any]
    state_update: WorkerStateUpdate = field(default_factory=_empty_state_update)


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


def _empty_detail_outcome() -> StateActionOutcome:
    return StateActionOutcome(
        result={
            "action": "review_job_detail",
            "status": "skipped",
            "result": "No accumulated detail OCR text to review.",
            "reason": "empty_job_detail_buffer",
        },
        state_update={},
    )


def _prepare_job_draft(
    state: WorkerState,
    *,
    current_url: str,
    raw_ocr_text: str,
) -> JobDraft:
    buffer = (state["collection"].get("job_detail_buffer") or {}).copy()
    stats = dict(buffer.get("stats") or {})
    transition = dict(state["transition"].get("transition_result") or {})
    return JobDraft(
        url=current_url,
        detail_key=job_detail_key_from_state(state),
        raw_ocr_text=raw_ocr_text,
        required_fields=state["request"]["collection_intent"].required_fields,
        screenshot_path=detail_evidence_screenshot(buffer),
        source_card_key=_active_source_card_key(state, current_url),
        screen_count=int(str(stats.get("screen_count") or 0)),
        last_action=str(transition.get("action") or ""),
        transition_status=str(transition.get("status") or ""),
        transition_reason=str(
            transition.get("reason") or transition.get("outcome") or ""
        ),
    )


def _request_job_review(draft: JobDraft) -> StateActionOutcome:
    return StateActionOutcome(
        result={
            "action": "review_job_detail",
            "status": "success",
            "result": "Accumulated detail OCR queued for review.",
            "ocr_chars": len(draft.raw_ocr_text),
            "screen_count": draft.screen_count,
        },
        state_update={
            "collection": {
                "pending_job_draft": draft,
            }
        },
    )


def _unchanged_review_outcome() -> StateActionOutcome:
    return StateActionOutcome(
        result={
            "action": "review_job_detail",
            "status": "skipped",
            "reason": "detail_evidence_unchanged",
            "result": (
                "직전 검토 이후 상세 OCR 근거가 추가되지 않았습니다. "
                "본문을 더 읽거나 다른 화면 행동을 수행해야 합니다."
            ),
        },
        state_update={},
    )


def _request_primary_job_review(draft: JobDraft) -> StateActionOutcome:
    return _request_job_review(
        draft.model_copy(update={"review_model_tier": "primary"})
    )


def _job_review_request_mode(
    state: WorkerState,
    *,
    current_url: str,
) -> tuple[JobDraft | None, str]:
    """현재 상세 근거가 신규 검토, 고성능 재검토, 중복 중 무엇인지 반환한다."""

    buffer = (state["collection"].get("job_detail_buffer") or {}).copy()
    raw_ocr_text = detail_buffer_text(buffer)
    if not raw_ocr_text:
        return None, "empty"
    draft = _prepare_job_draft(
        state,
        current_url=current_url,
        raw_ocr_text=raw_ocr_text,
    )
    last_review = state["collection"].get("last_job_review")
    if (
        last_review is not None
        and last_review.status == JobReviewStatus.NEEDS_MORE
        and last_review.draft_fingerprint == draft.fingerprint()
    ):
        return (
            draft,
            "primary" if last_review.model_tier == "lightweight" else "unchanged",
        )
    return draft, "lightweight"


def can_request_job_detail_review(state: WorkerState, current_url: str) -> bool:
    """실제 정제 호출로 이어질 상세 근거가 있는지 반환한다."""

    _draft, mode = _job_review_request_mode(state, current_url=current_url)
    return mode in {"lightweight", "primary"}


def _review_job_detail(
    *,
    current_url: str,
    state: WorkerState,
) -> StateActionOutcome:
    try:
        draft, mode = _job_review_request_mode(state, current_url=current_url)
        if draft is None:
            return _empty_detail_outcome()
        if mode == "primary":
            return _request_primary_job_review(draft)
        if mode == "unchanged":
            return _unchanged_review_outcome()
        return _request_job_review(draft)
    except Exception as exc:
        return StateActionOutcome(
            result={
                "action": "review_job_detail",
                "status": "error",
                "result": f"Failed to prepare detail review: {exc}",
            },
            state_update={},
        )


def _set_job_card_queue(
    args: dict[str, Any],
    *,
    current_url: str,
    state: WorkerState,
    data_services: WorkerDataServices,
) -> StateActionOutcome:
    queue = normalize_job_card_queue(args, state)
    queue, existing_cards = data_services.mark_existing_job_cards(
        queue,
        current_url,
    )
    availability: dict[str, Any] = {}
    raw_available_count = args.get("available_job_count")
    try:
        available_count = (
            int(str(raw_available_count)) if raw_available_count is not None else -1
        )
    except (TypeError, ValueError):
        available_count = -1
    count_evidence = str(args.get("count_evidence") or "").strip()[:160]
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
                "job_results_availability": availability,
            },
            "progress": {
                "stage": "results",
            },
        },
    )


def dispatch_state_action(
    action_name: str,
    args: dict[str, Any],
    *,
    current_url: str = "",
    state: WorkerState,
    data_services: WorkerDataServices,
) -> StateActionOutcome:
    """공고 추출과 카드 큐처럼 그래프 상태를 변경하는 행동을 실행한다."""

    if action_name == "review_job_detail":
        return _review_job_detail(
            current_url=current_url,
            state=state,
        )
    if action_name == "set_job_card_queue":
        return _set_job_card_queue(
            args,
            current_url=current_url,
            state=state,
            data_services=data_services,
        )
    raise ValueError(f"Unknown state action: {action_name}")


__all__ = [
    "StateActionOutcome",
    "can_request_job_detail_review",
    "dispatch_state_action",
    "dispatch_ui_action",
]

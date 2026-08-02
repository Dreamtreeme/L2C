"""선택 정책의 결과를 GraphState 갱신 값으로 조립한다."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.action_request import ActionRequest, build_action_request
from agent.graph.state import GraphState
from agent.graph.worker_state import count_mode_from_state, target_count_from_state
from agent.runtime.job_card_queue import (
    job_card_queue_scope_complete,
    return_action_from_transition,
    skip_active_job_card,
)
from agent.runtime.transition_runtime import build_transition_observation


def low_information_stop_update(capture_count: int) -> dict[str, Any]:
    request = build_action_request(
        "screen_policy",
        "stop after repeated low-information captures",
        [
            {
                "name": "finish_task",
                "args": {
                    "result": (
                        "브라우저 화면이 준비되지 않아 현재까지 확보한 정보만으로 "
                        "수집을 종료했습니다."
                    )
                },
                "id": "screen_policy_low_information_stop",
            }
        ],
    )
    return {
        "pending_action": request,
        "job_page_policy_trace": {
            "policy": "low_information_stop",
            "capture_count": capture_count,
        },
    }


def duplicate_detail_update(
    state: GraphState,
    *,
    transition_result: dict[str, Any],
    current_url: str,
    active_card: dict[str, Any],
    duplicate_trace: dict[str, Any],
) -> dict[str, Any]:
    skipped_queue_id = str(active_card.get("queue_id") or "")
    queue, cleared_card = skip_active_job_card(
        [
            dict(item)
            for item in state.get("job_card_queue", []) or []
            if isinstance(item, dict)
        ],
        active_card,
        reason="existing_detail_url",
        url=current_url,
        job_id=duplicate_trace.get("job_id"),
    )
    queue_complete = job_card_queue_scope_complete(
        queue,
        count_mode=count_mode_from_state(state),
        target_count=target_count_from_state(state),
    )
    request = build_action_request(
        "duplicate_job_policy",
        "skip detail OCR for an already collected job",
        [
            {
                "name": "finish_task" if queue_complete else "go_back",
                "args": (
                    {
                        "result": (
                            "현재 검색 결과 큐의 모든 공고 처리를 마쳤습니다."
                        )
                    }
                    if queue_complete
                    else {
                        "reason": (
                            "이미 수집한 공고 URL이므로 상세 읽기를 생략합니다."
                        ),
                        "expected_after": "검색 결과 목록으로 돌아간다.",
                    }
                ),
                "id": "skip_existing_job_detail",
            }
        ],
    )
    return {
        "pending_action": request,
        "transition_request": {},
        "transition_result": {
            **transition_result,
            "status": "ready",
            "outcome": "existing_job_detail",
            "reason": "existing_job_detail",
            "needs_ocr": False,
        },
        "job_card_queue": queue,
        "active_job_card": cleared_card,
        "job_page_policy_trace": {
            "policy": (
                "finish_existing_job_queue"
                if queue_complete
                else "skip_existing_job_detail"
            ),
            "queue_id": skipped_queue_id,
            **duplicate_trace,
        },
    }


def _queue_return_transition_record(
    state: GraphState,
    transition_result: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any] | None:
    return_action = return_action_from_transition(transition_result)
    if not return_action:
        return None
    memory = dict(state.get("job_results_memory", {}) or {})
    saved_signature = dict(memory.get("screen_signature", {}) or {})
    anchors = [
        str(text)
        for text in saved_signature.get("anchors", []) or []
        if str(text)
    ]
    evidence_markers = [
        {"id": index, "text": text}
        for index, text in enumerate(anchors)
    ]
    return_match = (
        trace.get("return_match")
        if isinstance(trace.get("return_match"), dict)
        else {}
    )
    started_at = float(
        transition_result.get("started_at") or time.time()
    )
    return build_transition_observation(
        transition_result,
        status="ready",
        outcome="queue_return_phash_match",
        source=str(transition_result.get("source") or ""),
        reason="queue_return_phash_match",
        elapsed_sec=max(0.0, time.time() - started_at),
        attempt=int(transition_result.get("attempts") or 0) + 1,
        markers=evidence_markers,
        screenshot=str(state.get("current_screenshot") or ""),
        marked_image="",
        to_capture_id=str(state.get("current_capture_id") or ""),
        phash_distance=return_match.get("distance"),
        ocr_skipped=True,
    )


def queue_replay_update(
    state: GraphState,
    *,
    request: ActionRequest,
    selected_markers: list[dict[str, Any]],
    trace: dict[str, Any],
    transition_result: dict[str, Any],
    signature: dict[str, Any],
    memory: dict[str, Any],
    saved_signature: dict[str, Any],
    return_action: dict[str, Any] | None,
) -> dict[str, Any]:
    merged_signature = {**saved_signature, **signature}
    if return_action:
        memory["return_action"] = return_action
    transition_record = _queue_return_transition_record(
        state,
        transition_result,
        trace,
    )
    return {
        "pending_action": request,
        "transition_request": {},
        "transition_result": {
            **transition_result,
            "status": "ready",
            "outcome": "queue_return_phash_match",
            "reason": "queue_return_phash_match",
            "needs_ocr": False,
        },
        "current_markers": selected_markers,
        "screen_signature": merged_signature,
        "current_page_role": "search",
        "ocr_capture_id": str(state.get("current_capture_id") or ""),
        "ocr_complete": True,
        "job_results_memory": memory,
        "job_card_replay_trace": trace,
        "return_to_job_results": {},
        **(
            {"transition_records": [transition_record]}
            if transition_record
            else {}
        ),
    }


__all__ = [
    "duplicate_detail_update",
    "low_information_stop_update",
    "queue_replay_update",
]

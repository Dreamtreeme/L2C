"""LLM 호출 전에 실행하는 결정론적 행동 선택 정책."""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.graph.action_request import build_action_request
from agent.graph.state import GraphState
from agent.graph.worker_state import count_mode_from_state, target_count_from_state
from agent.runtime.duplicate_job_policy import existing_job_url_trace
from agent.runtime.result_card_queue import (
    queue_replay_after_return,
    result_card_queue_scope_complete,
    return_action_from_transition,
    skip_active_result_card,
)
from agent.runtime.site_context import looks_like_job_detail_url
from agent.utils.logger import logger


def select_deterministic_action_node(state: GraphState) -> dict[str, Any]:
    """중복 공고와 목록 복귀 큐를 검사해 원자 행동 하나를 선택한다."""

    if state.get("pending_action") is not None:
        return {}
    if state.get("low_information_screen"):
        capture_count = int(state.get("low_information_capture_count") or 0)
        max_cycles = get_settings().vision.low_information_max_capture_cycles
        if capture_count >= max_cycles:
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
                "page_policy_trace": {
                    "policy": "low_information_stop",
                    "capture_count": capture_count,
                },
            }
        return {}

    observed_transition = dict(state.get("observed_transition", {}) or {})
    current_url = str(state.get("current_url") or "")
    active_card = dict(state.get("active_result_card", {}) or {})

    if active_card and looks_like_job_detail_url(current_url):
        duplicate_trace = existing_job_url_trace(
            current_url,
            dict(state.get("extracted_jd", {}) or {}),
        )
        if duplicate_trace.get("matched"):
            skipped_queue_id = str(active_card.get("queue_id") or "")
            queue, active_card = skip_active_result_card(
                [
                    dict(item)
                    for item in state.get("result_card_queue", []) or []
                    if isinstance(item, dict)
                ],
                active_card,
                reason="existing_detail_url",
                url=current_url,
                job_id=duplicate_trace.get("job_id"),
            )
            queue_complete = result_card_queue_scope_complete(
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
                            {"result": "현재 검색 결과 큐의 모든 공고 처리를 마쳤습니다."}
                            if queue_complete
                            else {
                                "reason": "이미 수집한 공고 URL이므로 상세 읽기를 생략합니다.",
                                "expected_after": "검색 결과 목록으로 돌아간다.",
                            }
                        ),
                        "id": "skip_existing_job_detail",
                    }
                ],
            )
            logger.info(
                "Existing job detail selected for skip",
                url=current_url,
                source=duplicate_trace.get("source", ""),
            )
            return {
                "pending_action": request,
                "pending_transition": {},
                "transition_status": "ready",
                "transition_outcome": "existing_job_detail",
                "transition_source": str(observed_transition.get("source") or ""),
                "transition_reason": "existing_job_detail",
                "ocr_required": False,
                "result_card_queue": queue,
                "active_result_card": active_card,
                "page_policy_trace": {
                    "policy": (
                        "finish_existing_job_queue"
                        if queue_complete
                        else "skip_existing_job_detail"
                    ),
                    "queue_id": skipped_queue_id,
                    **duplicate_trace,
                },
            }

    if observed_transition:
        ocr_complete = bool(state.get("ocr_complete"))
        markers = list(state.get("current_markers") or []) if ocr_complete else []
        signature_value = (
            state.get("screen_signature")
            if ocr_complete
            else state.get("raw_screen_signature")
        )
        signature = dict(signature_value or {})
        selection_state = {
            **state,
            "current_url": current_url,
            "current_markers": markers,
            "screen_signature": signature,
            "result_card_queue": list(state.get("result_card_queue", []) or []),
            "result_page_memory": dict(state.get("result_page_memory", {}) or {}),
            "active_result_card": dict(state.get("active_result_card", {}) or {}),
        }
        request, selected_markers, trace = queue_replay_after_return(
            selection_state,
            observed_transition,
            current_url,
            markers,
            signature,
            require_anchors=ocr_complete,
        )
        if request is not None:
            memory = dict(state.get("result_page_memory", {}) or {})
            saved_signature = dict(memory.get("screen_signature", {}) or {})
            merged_signature = {**saved_signature, **signature}
            return_action = return_action_from_transition(observed_transition)
            if return_action:
                memory["return_action"] = return_action
            logger.info(
                "Result card queue action selected",
                queue_id=trace.get("queue_id", ""),
                ocr_skipped=not ocr_complete,
            )
            return {
                "pending_action": request,
                "pending_transition": {},
                "transition_status": "ready",
                "transition_outcome": "queue_return_phash_match",
                "transition_source": str(observed_transition.get("source") or ""),
                "transition_reason": "queue_return_phash_match",
                "ocr_required": False,
                "current_markers": selected_markers,
                "screen_signature": merged_signature,
                "current_page_role": "search",
                "result_page_memory": memory,
                "queue_replay_trace": trace,
                "detail_return_pending": {},
            }

    return {
        "queue_replay_trace": {},
        "page_policy_trace": {},
    }


__all__ = ["select_deterministic_action_node"]

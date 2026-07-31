"""LLM 호출 전에 실행하는 결정론적 행동 선택 정책."""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.graph.state import GraphState
from agent.graph.worker_selection_policy import (
    SelectionPolicy,
    decide_duplicate_detail,
    decide_queue_return,
    decide_selection_entry,
)
from agent.graph.worker_selection_state import (
    duplicate_detail_update,
    low_information_stop_update,
    queue_replay_update,
    queue_return_wait_update,
)
from agent.runtime.duplicate_job_policy import existing_job_url_trace
from agent.runtime.job_card_queue import (
    replay_job_card_after_return,
    return_action_from_transition,
)
from agent.runtime.site_context import looks_like_job_detail_url
from agent.utils.logger import logger


def selection_node(state: GraphState) -> dict[str, Any]:
    """중복 공고와 목록 복귀 큐를 검사해 원자 행동 하나를 선택한다."""

    capture_count = int(state.get("low_information_capture_count") or 0)
    entry_decision = decide_selection_entry(
        has_pending_action=state.get("pending_action") is not None,
        low_information_screen=bool(state.get("low_information_screen")),
        low_information_capture_count=capture_count,
        low_information_max_cycles=(
            get_settings().vision.low_information_max_capture_cycles
        ),
        has_active_reflex_recipe=bool(state.get("active_reflex_recipe")),
    )
    if entry_decision.policy == SelectionPolicy.KEEP_PENDING_ACTION:
        return {}
    if entry_decision.policy in {
        SelectionPolicy.WAIT_LOW_INFORMATION,
        SelectionPolicy.STOP_LOW_INFORMATION,
    }:
        if entry_decision.policy == SelectionPolicy.STOP_LOW_INFORMATION:
            return low_information_stop_update(capture_count)
        return {}

    if entry_decision.policy == SelectionPolicy.DEFER_TO_ACTIVE_REFLEX:
        return {
            "job_card_replay_trace": {},
            "job_page_policy_trace": {},
        }

    transition_result = dict(state.get("transition_result", {}) or {})
    current_url = str(state.get("current_url") or "")
    active_card = dict(state.get("active_job_card", {}) or {})

    if active_card and looks_like_job_detail_url(current_url):
        duplicate_trace = existing_job_url_trace(
            current_url,
            dict(state.get("extracted_jd", {}) or {}),
        )
        duplicate_decision = decide_duplicate_detail(
            has_active_card=bool(active_card),
            is_job_detail_url=True,
            duplicate_matched=bool(duplicate_trace.get("matched")),
        )
        if duplicate_decision.policy == SelectionPolicy.SKIP_DUPLICATE_DETAIL:
            logger.info(
                "Existing job detail selected for skip",
                url=current_url,
                source=duplicate_trace.get("source", ""),
            )
            return duplicate_detail_update(
                state,
                transition_result=transition_result,
                current_url=current_url,
                active_card=active_card,
                duplicate_trace=duplicate_trace,
            )

    if transition_result.get("action"):
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
            "job_card_queue": list(state.get("job_card_queue", []) or []),
            "job_results_memory": dict(state.get("job_results_memory", {}) or {}),
            "active_job_card": dict(state.get("active_job_card", {}) or {}),
        }
        request, selected_markers, trace = replay_job_card_after_return(
            selection_state,
            transition_result,
            current_url,
            markers,
            signature,
            require_anchors=ocr_complete,
        )
        return_action = return_action_from_transition(
            transition_result
        )
        memory = dict(state.get("job_results_memory", {}) or {})
        saved_signature = dict(memory.get("screen_signature", {}) or {})
        target_phash = str(saved_signature.get("phash") or "")
        queue_decision = decide_queue_return(
            replay_available=request is not None,
            is_return_action=bool(return_action),
            ocr_complete=ocr_complete,
            replay_reason=str(trace.get("reason") or ""),
            transition_needs_ocr=bool(transition_result.get("needs_ocr")),
            target_phash_available=bool(target_phash),
        )
        if queue_decision.policy == SelectionPolicy.WAIT_FOR_RESULTS_SCREEN:
            logger.info(
                "Waiting for cached job results pHash",
                phash_distance=trace.get("distance"),
                max_distance=trace.get("max_distance"),
            )
            return queue_return_wait_update(
                state,
                transition_result=transition_result,
                trace=trace,
                target_phash=target_phash,
            )
        if queue_decision.policy == SelectionPolicy.REPLAY_JOB_CARD:
            assert request is not None
            logger.info(
                "Job card queue action selected",
                queue_id=trace.get("queue_id", ""),
                ocr_skipped=not ocr_complete,
            )
            return queue_replay_update(
                state,
                request=request,
                selected_markers=selected_markers,
                trace=trace,
                transition_result=transition_result,
                signature=signature,
                memory=memory,
                saved_signature=saved_signature,
                return_action=return_action,
            )

    return {
        "job_card_replay_trace": {},
        "job_page_policy_trace": {},
    }


__all__ = ["selection_node"]

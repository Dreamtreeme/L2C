"""LLM 호출 전에 실행하는 결정론적 행동 선택 정책."""

from __future__ import annotations

import time
from typing import Any

from agent.config import get_settings
from agent.graph.action_request import build_action_request
from agent.graph.state import GraphState
from agent.graph.worker_state import count_mode_from_state, target_count_from_state
from agent.runtime.duplicate_job_policy import existing_job_url_trace
from agent.runtime.followup_runtime import select_followup_after_transition
from agent.runtime.job_card_queue import (
    replay_job_card_after_return,
    job_card_queue_scope_complete,
    return_action_from_transition,
    skip_active_job_card,
)
from agent.runtime.site_context import looks_like_job_detail_url
from agent.runtime.transition_runtime import build_transition_observation
from agent.utils.logger import logger


def _queue_return_transition_record(
    state: GraphState,
    transition_result: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any] | None:
    """목록 pHash 재사용이 성공한 경우 복귀 행동의 전환 증거를 남긴다."""

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


def selection_node(state: GraphState) -> dict[str, Any]:
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
                "job_page_policy_trace": {
                    "policy": "low_information_stop",
                    "capture_count": capture_count,
                },
            }
        return {}

    transition_result = dict(state.get("transition_result", {}) or {})
    current_url = str(state.get("current_url") or "")
    active_card = dict(state.get("active_job_card", {}) or {})

    if active_card and looks_like_job_detail_url(current_url):
        duplicate_trace = existing_job_url_trace(
            current_url,
            dict(state.get("extracted_jd", {}) or {}),
        )
        if duplicate_trace.get("matched"):
            skipped_queue_id = str(active_card.get("queue_id") or "")
            queue, active_card = skip_active_job_card(
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
                "transition_request": {},
                "transition_result": {
                    **transition_result,
                    "status": "ready",
                    "outcome": "existing_job_detail",
                    "reason": "existing_job_detail",
                    "needs_ocr": False,
                },
                "job_card_queue": queue,
                "active_job_card": active_card,
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

    if transition_result.get("action"):
        request, followup_trace = select_followup_after_transition(
            state,
            transition_result,
        )
        if request is not None:
            logger.info(
                "Contextual follow-up action selected",
                trigger_action=transition_result.get("action", ""),
                action=followup_trace.get("action", ""),
            )
            return {
                "pending_action": request,
                "transition_request": {},
                "followup_action_trace": followup_trace,
            }

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
        if (
            request is None
            and return_action
            and not ocr_complete
            and trace.get("reason") == "phash_mismatch"
        ):
            memory = dict(state.get("job_results_memory", {}) or {})
            saved_signature = dict(memory.get("screen_signature", {}) or {})
            target_phash = str(saved_signature.get("phash") or "")
            if target_phash:
                pending = dict(
                    state.get("transition_request")
                    or transition_result
                )
                pending["pending_target_phash"] = target_phash
                pending["pending_target_max_distance"] = int(
                    trace.get("max_distance") or 0
                )
                logger.info(
                    "Waiting for cached job results pHash",
                    phash_distance=trace.get("distance"),
                    max_distance=trace.get("max_distance"),
                )
                return {
                    "transition_request": pending,
                    "transition_result": {
                        **transition_result,
                        "status": "pending",
                        "reason": "queue_return_phash_wait",
                        "needs_ocr": False,
                    },
                    "job_card_replay_trace": trace,
                }
        if request is not None:
            memory = dict(state.get("job_results_memory", {}) or {})
            saved_signature = dict(memory.get("screen_signature", {}) or {})
            merged_signature = {**saved_signature, **signature}
            if return_action:
                memory["return_action"] = return_action
            transition_record = _queue_return_transition_record(
                state,
                transition_result,
                trace,
            )
            logger.info(
                "Job card queue action selected",
                queue_id=trace.get("queue_id", ""),
                ocr_skipped=not ocr_complete,
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
                "job_results_memory": memory,
                "job_card_replay_trace": trace,
                "return_to_job_results": {},
                **(
                    {"transition_records": [transition_record]}
                    if transition_record
                    else {}
                ),
            }

    return {
        "followup_action_trace": {},
        "job_card_replay_trace": {},
        "job_page_policy_trace": {},
    }


__all__ = ["selection_node"]

"""LLM 호출 전에 실행하는 결정론적 행동 선택 정책."""

from __future__ import annotations

import time
from typing import Any

from langgraph.runtime import Runtime

from agent.config import get_settings
from agent.runtime.worker_contracts import (
    ActionRequest,
    WorkerState,
    attach_action_transition,
    build_action_request,
)
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.runtime.worker_state import (
    count_mode_from_state,
    target_count_from_state,
)
from agent.runtime.job_card_queue import (
    active_job_card,
    job_card_queue_scope_complete,
    replay_job_card_on_results,
    skip_active_job_card,
)
from agent.runtime.site_context import looks_like_job_detail_url
from agent.runtime.transition_runtime import build_transition_observation
from agent.utils.logger import logger


def _low_information_stop() -> dict[str, Any]:
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
        "decision": {"pending_action": request},
    }


def _skip_duplicate_detail(
    state: WorkerState,
    *,
    transition_result: dict[str, Any],
    current_url: str,
    duplicate_trace: dict[str, Any],
) -> dict[str, Any]:
    queue = skip_active_job_card(
        [
            dict(item)
            for item in state["collection"].get("job_card_queue", []) or []
            if isinstance(item, dict)
        ],
        reason="existing_detail_url",
        url=current_url,
        job_id=duplicate_trace.get("job_id"),
    )
    queue_complete = job_card_queue_scope_complete(
        queue,
        count_mode=count_mode_from_state(state),
        target_count=target_count_from_state(state),
    )
    update: dict[str, Any] = {
        "transition": {
            "transition_request": {},
            "transition_result": {
                **transition_result,
                "status": "ready",
                "outcome": "existing_job_detail",
                "reason": "existing_job_detail",
                "needs_ocr": False,
            },
        },
        "collection": {"job_card_queue": queue},
    }
    if queue_complete:
        update["lifecycle"] = {"is_finished": True}
    return update


def _queue_results_transition(
    state: WorkerState,
    transition_result: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any] | None:
    if not str(transition_result.get("action") or ""):
        return None
    saved_signature = dict(
        (state["collection"].get("job_results_memory", {}) or {}).get(
            "screen_signature",
            {},
        )
    )
    anchors = [
        str(text)
        for text in saved_signature.get("anchors", []) or []
        if str(text)
    ]
    results_match = (
        trace.get("results_match")
        if isinstance(trace.get("results_match"), dict)
        else {}
    )
    started_at = float(
        transition_result.get("started_at") or time.time()
    )
    return build_transition_observation(
        transition_result,
        status="ready",
        outcome="queue_results_phash_match",
        source=str(transition_result.get("source") or ""),
        reason="queue_results_phash_match",
        elapsed_sec=max(0.0, time.time() - started_at),
        attempt=1,
        markers=[
            {"id": index, "text": text}
            for index, text in enumerate(anchors)
        ],
        screenshot=str(
            state["observation"].get("current_screenshot") or ""
        ),
        marked_image="",
        after_observation_id=str(
            state["observation"].get("observation_id") or ""
        ),
        phash_distance=results_match.get("distance"),
        ocr_skipped=True,
    )


def _replay_queued_card(
    state: WorkerState,
    *,
    request: ActionRequest,
    selected_markers: list[dict[str, Any]],
    trace: dict[str, Any],
    transition_result: dict[str, Any],
    signature: dict[str, Any],
    memory: dict[str, Any],
    saved_signature: dict[str, Any],
) -> dict[str, Any]:
    transition = _queue_results_transition(
        state,
        transition_result,
        trace,
    )
    update: dict[str, Any] = {
        "decision": {"pending_action": request},
        "transition": {
            "transition_request": {},
            "transition_result": {
                **transition_result,
                "status": "ready",
                "outcome": "queue_results_phash_match",
                "reason": "queue_results_phash_match",
                "needs_ocr": False,
            },
        },
        "observation": {
            "current_markers": selected_markers,
            "screen_signature": {**saved_signature, **signature},
            "current_page_role": "search",
            "ocr_complete": True,
        },
        "collection": {
            "job_results_memory": memory,
        },
    }
    if transition:
        update["transition"]["action_events"] = attach_action_transition(
            state["transition"].get("action_events", []) or [],
            transition,
        )
    return update


def selection_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> dict[str, Any]:
    """중복 공고와 확인된 목록 화면의 큐를 검사해 원자 행동 하나를 선택한다."""

    decision = state["decision"]
    observation = state["observation"]
    transition = state["transition"]
    replay = state["replay"]
    collection = state["collection"]
    if decision.get("pending_action") is not None:
        return {}

    capture_count = int(
        observation.get("low_information_capture_count") or 0
    )
    if observation.get("low_information_screen"):
        if (
            capture_count
            >= get_settings().vision.low_information_max_capture_cycles
        ):
            return _low_information_stop()
        return {}

    if replay.get("active_reflex_recipe"):
        return {}

    transition_result = dict(transition.get("transition_result", {}) or {})
    current_url = str(observation.get("current_url") or "")
    active_card = active_job_card(
        list(collection.get("job_card_queue", []) or [])
    )

    if active_card and looks_like_job_detail_url(current_url):
        duplicate_trace = runtime.context.data.find_existing_job_url(
            current_url,
            list(collection.get("job_captures", [])),
        )
        if duplicate_trace.get("matched"):
            logger.info(
                "Existing job detail selected for skip",
                url=current_url,
                source=duplicate_trace.get("source", ""),
            )
            return _skip_duplicate_detail(
                state,
                transition_result=transition_result,
                current_url=current_url,
                duplicate_trace=duplicate_trace,
            )

    if transition_result.get("action"):
        if transition_result.get("status") == "unknown":
            return {}
        ocr_complete = bool(observation.get("ocr_complete"))
        markers = (
            list(observation.get("current_markers") or [])
            if ocr_complete
            else []
        )
        signature_value = (
            observation.get("screen_signature")
            if ocr_complete
            else observation.get("raw_screen_signature")
        )
        signature = dict(signature_value or {})
        request, selected_markers, trace = replay_job_card_on_results(
            state,
            transition_result,
            current_url,
            markers,
            signature,
            require_anchors=ocr_complete,
        )
        memory = dict(collection.get("job_results_memory", {}) or {})
        saved_signature = dict(memory.get("screen_signature", {}) or {})
        if request is not None:
            logger.info(
                "Job card queue action selected",
                queue_id=trace.get("queue_id", ""),
                ocr_skipped=not ocr_complete,
            )
            return _replay_queued_card(
                state,
                request=request,
                selected_markers=selected_markers,
                trace=trace,
                transition_result=transition_result,
                signature=signature,
                memory=memory,
                saved_signature=saved_signature,
            )

    return {}


__all__ = ["selection_node"]

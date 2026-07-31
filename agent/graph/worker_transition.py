"""작업자 그래프의 화면 전환 판정 노드."""

from __future__ import annotations

import time
from typing import Any

from agent.config import get_settings
from agent.graph.state import GraphState
from agent.graph.worker_transition_policy import (
    decide_after_ocr,
    decide_before_ocr,
    decide_transition_probe,
    verify_reflex_after_state,
)
from agent.graph.worker_transition_state import (
    active_reflex_recipe_after_transition,
    blocked_recipe_keys,
    reused_observation,
    transition_record,
    transition_result,
)
from agent.runtime.transition_runtime import transition_has_visual_change
from agent.utils.logger import logger


def transition_node(state: GraphState) -> dict[str, Any]:
    """직전 원자 행동과 현재 캡처를 비교하고 OCR 필요 여부를 결정한다."""

    request = dict(state.get("transition_request", {}) or {})
    low_information = bool(state.get("low_information_screen"))
    ocr_complete = bool(state.get("ocr_complete"))

    if low_information:
        return {
            "transition_result": transition_result(
                request,
                status="pending" if request else "idle",
                reason="low_information_screen",
            ),
        }

    if not request:
        return {
            "transition_result": transition_result(
                {},
                status="idle",
                reason="no_transition_request",
                needs_ocr=not ocr_complete,
            ),
        }

    if state.get("transition_probe_unchanged"):
        started_at = float(request.get("started_at") or time.time())
        elapsed_sec = max(0.0, time.time() - started_at)
        timeout_sec = get_settings().vision.page_ready_timeout_sec
        attempt = int(request.get("attempts") or 0) + 1
        decision = decide_transition_probe(
            elapsed_sec=elapsed_sec,
            timeout_sec=timeout_sec,
        )
        if decision.status == "pending":
            request["attempts"] = attempt
            return {
                "transition_request": request,
                "transition_result": transition_result(
                    request,
                    status=decision.status,
                    reason=decision.reason,
                    needs_ocr=decision.needs_ocr,
                ),
                "transition_probe_unchanged": False,
            }

        record = transition_record(
            request,
            status="unknown",
            outcome="",
            source=str(request.get("source") or ""),
            reason="transition_timeout",
            attempt=attempt,
            state=state,
            phash_distance=None,
            visual_change_ratio=None,
            ocr_skipped=True,
        )
        return {
            "transition_request": {},
            "transition_result": transition_result(
                request,
                status=decision.status,
                reason=decision.reason,
                needs_ocr=decision.needs_ocr,
            ),
            "transition_records": [record],
            "transition_probe_unchanged": False,
            "active_reflex_recipe": active_reflex_recipe_after_transition(
                state,
                source=str(request.get("source") or ""),
                status="unknown",
            ),
        }

    image_path = str(state.get("current_screenshot") or "")
    current_url = str(state.get("current_url") or "")
    visual_changed, visual_ratio = transition_has_visual_change(
        request,
        image_path,
    )
    source = str(request.get("source") or "")
    blocked_keys = blocked_recipe_keys(state)

    if not ocr_complete:
        decision = decide_before_ocr(
            source=source,
            visual_changed=visual_changed,
        )
        if decision.status == "unknown":
            reason = decision.reason
            recipe_key = str(request.get("recipe_key") or "")
            if (
                decision.block_reflex_recipe
                and recipe_key
                and recipe_key not in blocked_keys
            ):
                blocked_keys.append(recipe_key)
            attempt = int(request.get("attempts") or 0) + 1
            observation_update = reused_observation(state, request)
            record_state = {**state, **observation_update}
            record = transition_record(
                request,
                status="unknown",
                outcome="",
                source=source,
                reason=reason,
                attempt=attempt,
                state=record_state,
                phash_distance=None,
                visual_change_ratio=visual_ratio,
                ocr_skipped=True,
            )
            logger.info(
                "Transition no-effect detected before OCR",
                source=source,
                action=request.get("action", ""),
                visual_change_ratio=visual_ratio,
            )
            return {
                "transition_request": {},
                "transition_result": transition_result(
                    request,
                    status="unknown",
                    reason=reason,
                    visual_change_ratio=visual_ratio,
                ),
                "transition_records": [record],
                "reflex_blocked_recipe_keys": blocked_keys,
                "active_reflex_recipe": active_reflex_recipe_after_transition(
                    state,
                    source=source,
                    status="unknown",
                ),
                **observation_update,
            }

        return {
            "transition_result": transition_result(
                request,
                status=decision.status,
                reason=decision.reason,
                visual_change_detected=visual_changed,
                visual_change_ratio=visual_ratio,
                needs_ocr=decision.needs_ocr,
            ),
        }

    markers = list(state.get("current_markers") or [])
    before_url = str(request.get("before_url") or "")
    url_changed = bool(
        before_url
        and current_url
        and before_url != current_url
    )
    if source == "reflex":
        matched, reason, after_state_match = (
            verify_reflex_after_state(request, state)
        )
        request["after_state_match"] = after_state_match
        decision = decide_after_ocr(
            source=source,
            markers_present=bool(markers),
            url_changed=url_changed,
            visual_changed=visual_changed,
            reflex_matched=matched,
            reflex_reason=reason,
        )
    else:
        decision = decide_after_ocr(
            source=source,
            markers_present=bool(markers),
            url_changed=url_changed,
            visual_changed=visual_changed,
        )
    status = decision.status
    reason = decision.reason
    outcome = ""

    attempt = int(request.get("attempts") or 0) + 1
    record = transition_record(
        request,
        status=status,
        outcome=outcome,
        source=source,
        reason=reason,
        attempt=attempt,
        state=state,
        phash_distance=None,
        visual_change_ratio=visual_ratio,
        ocr_skipped=False,
    )
    evaluated_request = dict(request)
    if decision.block_reflex_recipe:
        recipe_key = str(request.get("recipe_key") or "")
        if recipe_key and recipe_key not in blocked_keys:
            blocked_keys.append(recipe_key)
    request = {}

    logger.info(
        "Transition evaluated",
        source=source,
        status=status,
        reason=reason,
    )
    return {
        "transition_request": request,
        "transition_result": transition_result(
            evaluated_request,
            status=status,
            outcome=outcome,
            reason=reason,
            visual_change_detected=visual_changed,
            visual_change_ratio=visual_ratio,
        ),
        "transition_records": [record],
        "reflex_blocked_recipe_keys": blocked_keys,
        "active_reflex_recipe": active_reflex_recipe_after_transition(
            state,
            source=source,
            status=status,
        ),
        "transition_probe_unchanged": False,
    }


__all__ = ["transition_node"]

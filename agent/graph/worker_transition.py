"""작업자 그래프의 화면 전환 판정 노드."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.state import GraphState
from agent.runtime.transition_runtime import (
    build_transition_observation,
    transition_accepts_visual_change,
    transition_has_visual_change,
    transition_no_effect_by_phash,
    transition_phash_distance,
)
from agent.utils.logger import logger


def _blocked_recipe_keys(state: GraphState) -> list[str]:
    return [
        str(key)
        for key in (state.get("reflex_blocked_recipe_keys") or [])
        if str(key)
    ]


def _reused_observation(
    state: GraphState,
    pending: dict[str, Any],
) -> dict[str, Any]:
    """pHash 무변화가 확인된 경우에만 행동 전 OCR 관찰을 현재 캡처에 연결한다."""

    previous = dict(state.get("previous_screen_observation") or {})
    if not previous:
        return {}

    expected_capture_id = str(pending.get("from_capture_id") or "")
    previous_capture_id = str(previous.get("capture_id") or "")
    if (
        not expected_capture_id
        or not previous_capture_id
        or expected_capture_id != previous_capture_id
    ):
        return {}

    before_screenshot = str(pending.get("before_screenshot") or "")
    previous_screenshot = str(previous.get("screenshot") or "")
    if (
        not before_screenshot
        or not previous_screenshot
        or before_screenshot != previous_screenshot
    ):
        return {}

    before_url = str(pending.get("before_url") or "")
    previous_url = str(previous.get("current_url") or "")
    if before_url and previous_url and before_url != previous_url:
        return {}

    markers = [
        dict(marker)
        for marker in previous.get("markers", []) or []
        if isinstance(marker, dict)
    ]
    if not markers:
        return {}

    screen_signature = dict(previous.get("screen_signature") or {})
    raw_signature = dict(state.get("raw_screen_signature") or {})
    if raw_signature.get("phash"):
        screen_signature["phash"] = raw_signature["phash"]
    if raw_signature.get("size"):
        screen_signature["size"] = raw_signature["size"]

    current_observation = {
        **previous,
        "capture_id": str(state.get("current_capture_id") or ""),
        "screenshot": str(state.get("current_screenshot") or ""),
        "current_url": str(state.get("current_url") or previous_url),
        "markers": markers,
        "screen_signature": screen_signature,
    }
    return {
        "current_markers": markers,
        "ui_context": str(previous.get("ui_context") or ""),
        "marked_image": str(previous.get("marked_image") or ""),
        "screen_signature": screen_signature,
        "current_page_role": str(previous.get("page_role") or ""),
        "analysis_mode": str(previous.get("analysis_mode") or "full"),
        "ocr_complete": True,
        "previous_screen_observation": current_observation,
    }


def _transition_record(
    pending: dict[str, Any],
    *,
    status: str,
    outcome: str,
    source: str,
    reason: str,
    attempt: int,
    state: GraphState,
    phash_distance: int | None,
    visual_change_ratio: float | None,
    ocr_skipped: bool,
) -> dict[str, Any]:
    started_at = float(pending.get("started_at") or time.time())
    return build_transition_observation(
        pending,
        status=status,
        outcome=outcome,
        source=source,
        reason=reason,
        elapsed_sec=max(0.0, time.time() - started_at),
        attempt=attempt,
        markers=list(state.get("current_markers", []) or []),
        screenshot=str(state.get("current_screenshot") or ""),
        marked_image=str(state.get("marked_image") or ""),
        to_capture_id=str(state.get("current_capture_id") or ""),
        phash_distance=phash_distance,
        visual_change_ratio=visual_change_ratio,
        ocr_skipped=ocr_skipped,
    )


def _transition_result(
    request: dict[str, Any],
    *,
    status: str,
    outcome: str = "",
    reason: str = "",
    visual_change_detected: bool = False,
    visual_change_ratio: float | None = None,
    needs_ocr: bool = False,
) -> dict[str, Any]:
    """전환 요청의 식별 정보와 판정 값을 하나의 결과로 합친다."""

    return {
        **request,
        "status": status,
        "outcome": outcome,
        "reason": reason,
        "visual_change_detected": visual_change_detected,
        "visual_change_ratio": visual_change_ratio,
        "needs_ocr": needs_ocr,
    }


def transition_node(state: GraphState) -> dict[str, Any]:
    """직전 원자 행동과 현재 캡처를 비교하고 OCR 필요 여부를 결정한다."""

    request = dict(state.get("transition_request", {}) or {})
    low_information = bool(state.get("low_information_screen"))
    ocr_complete = bool(state.get("ocr_complete"))

    if low_information:
        return {
            "transition_result": _transition_result(
                request,
                status="pending" if request else "idle",
                reason="low_information_screen",
            ),
        }

    if not request:
        return {
            "transition_result": _transition_result(
                {},
                status="idle",
                reason="no_transition_request",
                needs_ocr=not ocr_complete,
            ),
        }

    if state.get("transition_probe_unchanged"):
        contract = (
            request.get("contract")
            if isinstance(request.get("contract"), dict)
            else {}
        )
        started_at = float(request.get("started_at") or time.time())
        elapsed_sec = max(0.0, time.time() - started_at)
        timeout_sec = float(contract.get("timeout_sec") or 12.0)
        attempt = int(request.get("attempts") or 0) + 1
        if elapsed_sec < timeout_sec:
            request["attempts"] = attempt
            wait_reason = (
                "target_screen_not_ready"
                if request.get("pending_target_phash")
                else "screen_unchanged_while_waiting"
            )
            return {
                "transition_request": request,
                "transition_result": _transition_result(
                    request,
                    status="pending",
                    reason=wait_reason,
                    needs_ocr=False,
                ),
                "transition_probe_unchanged": False,
            }

        record = _transition_record(
            request,
            status="unknown",
            outcome="",
            source=str(request.get("source") or ""),
            reason="transition_timeout",
            attempt=attempt,
            state=state,
            phash_distance=0,
            visual_change_ratio=0.0,
            ocr_skipped=True,
        )
        return {
            "transition_request": {},
            "transition_result": _transition_result(
                request,
                status="unknown",
                reason="transition_timeout",
                needs_ocr=False,
            ),
            "transition_records": [record],
            "transition_probe_unchanged": False,
        }

    image_path = str(state.get("current_screenshot") or "")
    current_url = str(state.get("current_url") or "")
    raw_signature = dict(state.get("raw_screen_signature", {}) or {})
    visual_changed, visual_ratio = transition_has_visual_change(
        request,
        image_path,
    )
    source = str(request.get("source") or "")
    blocked_keys = _blocked_recipe_keys(state)

    if not ocr_complete:
        no_effect, phash_distance = transition_no_effect_by_phash(
            request,
            current_url,
            raw_signature,
        )
        if no_effect and not visual_changed:
            reason = "reflex_no_screen_change" if source == "reflex" else "no_screen_change"
            recipe_key = str(request.get("recipe_key") or "")
            if source == "reflex" and recipe_key and recipe_key not in blocked_keys:
                blocked_keys.append(recipe_key)
            attempt = int(request.get("attempts") or 0) + 1
            reused_observation = _reused_observation(state, request)
            record_state = {**state, **reused_observation}
            record = _transition_record(
                request,
                status="unknown",
                outcome="",
                source=source,
                reason=reason,
                attempt=attempt,
                state=record_state,
                phash_distance=phash_distance,
                visual_change_ratio=visual_ratio,
                ocr_skipped=True,
            )
            logger.info(
                "Transition no-effect detected before OCR",
                source=source,
                action=request.get("action", ""),
                phash_distance=phash_distance,
            )
            return {
                "transition_request": {},
                "transition_result": _transition_result(
                    request,
                    status="unknown",
                    reason=reason,
                    visual_change_ratio=visual_ratio,
                ),
                "transition_records": [record],
                "reflex_blocked_recipe_keys": blocked_keys,
                **reused_observation,
            }

        return {
            "transition_result": _transition_result(
                request,
                status="needs_ocr",
                reason="ocr_required",
                visual_change_detected=visual_changed,
                visual_change_ratio=visual_ratio,
                needs_ocr=True,
            ),
        }

    from agent.recipe.transition import evaluate_transition

    markers = list(state.get("current_markers") or [])
    started_at = float(request.get("started_at") or time.time())
    evaluation = evaluate_transition(
        request.get("contract"),
        markers,
        params=dict(request.get("params", {}) or {}),
        elapsed_sec=max(0.0, time.time() - started_at),
    )
    status = str(evaluation.get("status") or "unknown")
    outcome = str(evaluation.get("outcome") or "")
    screen_signature = dict(state.get("screen_signature", {}) or {})
    same_url, phash_distance, no_effect_max_distance = transition_phash_distance(
        request,
        current_url,
        screen_signature,
    )
    reason = str(evaluation.get("reason") or "")
    if (
        source == "reflex"
        and status == "ready"
        and same_url
        and phash_distance is not None
        and phash_distance <= no_effect_max_distance
        and not visual_changed
    ):
        status = "unknown"
        outcome = ""
        reason = "reflex_no_screen_change"
    elif (
        status == "pending"
        and same_url
        and phash_distance is not None
        and transition_accepts_visual_change(request)
        and (phash_distance > no_effect_max_distance or visual_changed)
    ):
        status = "ready"
        outcome = ""
        reason = (
            "screen_change_pixels_matched"
            if visual_changed
            else "screen_change_phash_matched"
        )

    attempt = int(request.get("attempts") or 0) + 1
    record = _transition_record(
        request,
        status=status,
        outcome=outcome,
        source=source,
        reason=reason,
        attempt=attempt,
        state=state,
        phash_distance=phash_distance,
        visual_change_ratio=visual_ratio,
        ocr_skipped=False,
    )
    evaluated_request = dict(request)
    if status == "pending":
        request["attempts"] = attempt
        request["pending_screen_phash"] = str(
            screen_signature.get("phash") or ""
        )
        request["pending_screenshot"] = str(
            state.get("current_screenshot") or ""
        )
    else:
        if status == "unknown" and source == "reflex":
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
        "transition_result": _transition_result(
            evaluated_request,
            status=status,
            outcome=outcome,
            reason=reason,
            visual_change_detected=visual_changed,
            visual_change_ratio=visual_ratio,
        ),
        "transition_records": [record],
        "reflex_blocked_recipe_keys": blocked_keys,
        "transition_probe_unchanged": False,
    }


__all__ = ["transition_node"]

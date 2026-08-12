"""작업자 그래프의 화면 전환 판정 노드."""

from __future__ import annotations

import time
from typing import Any

from langgraph.runtime import Runtime

from agent.recipe.phash_replay import match_target_by_screen_signature
from agent.runtime.site_context import (
    is_job_detail_context,
    normalize_page_role,
)
from agent.runtime.job_card_queue import release_active_job_card
from agent.runtime.target_matching import screen_context_signature_match
from agent.runtime.worker_contracts import (
    WorkerState,
    apply_worker_state_update,
    attach_action_transition,
)
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.runtime.transition_runtime import (
    build_transition_observation,
    transition_has_visual_change,
    transition_marker_texts,
)
from agent.utils.logger import logger
from agent.utils.text import recipe_url_scope_matches, url_template
from shared.schema.recipe_schema import ActionTarget, ReplaySession


def _verify_reflex_after_state(
    request: dict[str, Any],
    state: WorkerState,
) -> tuple[bool, str, dict[str, Any]]:
    """저장된 레시피 도착 화면과 현재 관찰이 같은지 확인한다."""

    if request.get("execution_failed"):
        return False, "recipe_action_group_failed", {}
    expected = dict(request.get("expected_after_state") or {})
    if not expected:
        return False, "recipe_after_state_missing", {}

    expected_url = str(expected.get("url_template") or "")
    observation = state["observation"]
    current_url = str(observation.get("current_url") or "")
    if (
        expected_url
        and current_url
        and not recipe_url_scope_matches(expected_url, current_url)
    ):
        return False, "recipe_after_url_mismatch", {
            "expected_url_template": expected_url,
            "current_url": current_url,
        }

    before_role = normalize_page_role(request.get("before_page_role"))
    expected_role = normalize_page_role(expected.get("page_role"))
    current_role = normalize_page_role(observation.get("current_page_role"))
    if (
        before_role
        and expected_role
        and expected_role != before_role
        and current_role
    ):
        matched = current_role == expected_role
        return matched, (
            "recipe_after_page_role_matched"
            if matched
            else "recipe_after_page_role_mismatch"
        ), {
            "before_page_role": before_role,
            "expected_page_role": expected_role,
            "current_page_role": current_role,
        }

    anchor_target = expected.get("anchor_target")
    anchor_signature = dict(expected.get("anchor_roi_signature") or {})
    if isinstance(anchor_target, dict) and anchor_signature:
        marker_id, match = match_target_by_screen_signature(
            ActionTarget.model_validate(anchor_target),
            anchor_signature,
            dict(observation.get("screen_signature") or {}),
            list(observation.get("current_markers") or []),
            current_image_path=str(observation.get("current_screenshot") or ""),
        )
        if marker_id is None:
            return False, str(
                match.get("reason") or "recipe_after_anchor_mismatch"
            ), match
        return True, "recipe_after_anchor_matched", {
            **match,
            "marker_id": marker_id,
        }

    context_signature = dict(expected.get("screen_context_signature") or {})
    if context_signature:
        match = screen_context_signature_match(
            context_signature,
            dict(observation.get("screen_signature") or {}),
        )
        matched = bool(match.get("matched"))
        return matched, (
            "recipe_after_context_matched"
            if matched
            else str(match.get("reason") or "recipe_after_context_mismatch")
        ), match

    before_url = url_template(str(request.get("before_url") or ""))
    if (
        expected_url
        and current_url
        and expected_url != before_url
        and recipe_url_scope_matches(expected_url, current_url)
    ):
        return True, "recipe_after_url_matched", {
            "expected_url_template": expected_url,
            "current_url": current_url,
        }
    return False, "recipe_after_state_unverifiable", {}


def _blocked_recipe_keys(state: WorkerState) -> list[str]:
    return [
        str(key)
        for key in (
            state["replay"].get("reflex_blocked_recipe_keys") or []
        )
        if str(key)
    ]


def _replay_session(state: WorkerState) -> ReplaySession | None:
    raw_session = state["replay"].get("replay_session")
    if isinstance(raw_session, ReplaySession):
        return raw_session
    return ReplaySession.model_validate(raw_session) if raw_session else None


def _replay_session_after_transition(
    state: WorkerState,
    *,
    source: str,
    status: str,
) -> ReplaySession | None:
    session = _replay_session(state)
    if not session or source != "reflex":
        return session
    if status != "ready":
        return None
    return session.advance()


def _reused_observation(
    state: WorkerState,
    request: dict[str, Any],
) -> dict[str, Any]:
    """변화가 없을 때 직전 캡처의 OCR만 동일 화면에 다시 연결한다."""

    observation = state["observation"]
    previous = dict(observation.get("previous_observation") or {})
    if not previous:
        return {}
    if str(request.get("before_observation_id") or "") != str(
        previous.get("observation_id") or ""
    ):
        return {}
    if str(request.get("before_screenshot") or "") != str(
        previous.get("screenshot") or ""
    ):
        return {}
    before_url = str(request.get("before_url") or "")
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
    signature = dict(previous.get("screen_signature") or {})
    raw_signature = dict(observation.get("raw_screen_signature") or {})
    for key in ("phash", "size"):
        if raw_signature.get(key):
            signature[key] = raw_signature[key]
    current_observation = {
        **previous,
        "observation_id": str(observation.get("observation_id") or ""),
        "screenshot": str(observation.get("current_screenshot") or ""),
        "current_url": str(observation.get("current_url") or previous_url),
        "markers": markers,
        "screen_signature": signature,
    }
    return {
        "current_markers": markers,
        "ui_context": str(previous.get("ui_context") or ""),
        "marked_image": str(previous.get("marked_image") or ""),
        "screen_signature": signature,
        "current_page_role": str(previous.get("page_role") or ""),
        "ocr_complete": True,
        "previous_observation": current_observation,
    }


def _transition_record(
    request: dict[str, Any],
    *,
    status: str,
    source: str,
    reason: str,
    attempt: int,
    state: WorkerState,
    visual_change_ratio: float | None,
    ocr_skipped: bool,
) -> dict[str, Any]:
    started_at = float(request.get("started_at") or time.time())
    observation = state["observation"]
    return build_transition_observation(
        request,
        status=status,
        outcome="",
        source=source,
        reason=reason,
        elapsed_sec=max(0.0, time.time() - started_at),
        attempt=attempt,
        markers=list(observation.get("current_markers", []) or []),
        screenshot=str(observation.get("current_screenshot") or ""),
        marked_image=str(observation.get("marked_image") or ""),
        after_observation_id=str(observation.get("observation_id") or ""),
        current_url=str(observation.get("current_url") or ""),
        page_role=str(observation.get("current_page_role") or ""),
        screen_signature=dict(observation.get("screen_signature") or {}),
        visual_change_ratio=visual_change_ratio,
        ocr_skipped=ocr_skipped,
    )


def _transition_result(
    request: dict[str, Any],
    *,
    status: str,
    reason: str = "",
    visual_change_detected: bool = False,
    visual_change_ratio: float | None = None,
    needs_ocr: bool = False,
) -> dict[str, Any]:
    return {
        **request,
        "status": status,
        "outcome": "",
        "reason": reason,
        "visual_change_detected": visual_change_detected,
        "visual_change_ratio": visual_change_ratio,
        "needs_ocr": needs_ocr,
    }


def _result_without_transition(
    state: WorkerState,
    request: dict[str, Any],
) -> dict[str, Any] | None:
    if state["observation"].get("low_information_screen"):
        return {
            "transition": {
                "transition_result": _transition_result(
                    request,
                    status="pending" if request else "idle",
                    reason="low_information_screen",
                ),
            }
        }
    if not request:
        return {
            "transition": {
                "transition_result": _transition_result(
                    {},
                    status="idle",
                    reason="no_transition_request",
                    needs_ocr=not bool(
                        state["observation"].get("ocr_complete")
                    ),
                ),
            }
        }
    return None


def _blocked_keys_after_decision(
    state: WorkerState,
    request: dict[str, Any],
    *,
    should_block: bool,
) -> list[str]:
    keys = _blocked_recipe_keys(state)
    recipe_key = str(request.get("recipe_key") or "")
    if should_block and recipe_key and recipe_key not in keys:
        keys.append(recipe_key)
    return keys


def _record_replay_outcome(
    state: WorkerState,
    request: dict[str, Any],
    *,
    status: str,
    record_replay_result,
) -> None:
    """경로가 끝났거나 실패했을 때 한 번만 실제 재생 결과를 저장한다."""

    if str(request.get("source") or "") != "reflex":
        return
    session = _replay_session(state)
    recipe_key = str(
        (session.recipe_key if session else "") or request.get("recipe_key") or ""
    )
    if not recipe_key:
        return
    if not session or not session.pending_is_current():
        return
    succeeded = status == "ready"
    if succeeded and not session.is_last_transition():
        return
    try:
        record_replay_result(recipe_key, succeeded)
    except Exception as exc:
        logger.warning(
            "Recipe replay outcome persistence failed",
            recipe_key=recipe_key,
            error=str(exc),
        )


def _input_text_confirmed_by_ocr(
    state: WorkerState,
    request: dict[str, Any],
) -> bool:
    """입력 문자열이 이전 화면에는 없고 현재 OCR에 나타났는지 확인한다."""

    if request.get("action") != "type_in_marker":
        return False
    input_text = str(
        ((request.get("step") or {}).get("args") or {}).get("text") or ""
    ).casefold().replace(" ", "")
    if not input_text:
        return False
    observation = state["observation"]
    previous_markers = list(
        (observation.get("previous_observation") or {}).get("markers") or []
    )
    previous_texts = transition_marker_texts(previous_markers)
    current_texts = transition_marker_texts(
        list(observation.get("current_markers") or [])
    )

    def contains_input(texts: list[str]) -> bool:
        return any(input_text in text.casefold().replace(" ", "") for text in texts)

    return contains_input(current_texts) and not contains_input(previous_texts)


def _queue_click_used_cached_marker(
    state: WorkerState,
    request: dict[str, Any],
) -> bool:
    """현재 화면 OCR 없이 만든 큐 좌표를 클릭했는지 확인한다."""

    if (
        request.get("source") != "job_card_queue"
        or request.get("action") != "click_marker"
    ):
        return False
    step = request.get("step") if isinstance(request.get("step"), dict) else {}
    args = step.get("args") if isinstance(step.get("args"), dict) else {}
    marker_id = args.get("marker_id")
    previous = dict(state["observation"].get("previous_observation") or {})
    return any(
        marker.get("id") == marker_id
        and marker.get("type") == "queue_cached_card"
        for marker in previous.get("markers", []) or []
        if isinstance(marker, dict)
    )


def _evaluate_before_ocr(
    state: WorkerState,
    request: dict[str, Any],
    *,
    visual_changed: bool,
    visual_ratio: float | None,
    record_replay_result,
) -> dict[str, Any]:
    source = str(request.get("source") or "")
    action = str(request.get("action") or "")
    input_requires_ocr = action == "type_in_marker"
    cached_queue_marker_failed = (
        not visual_changed
        and _queue_click_used_cached_marker(state, request)
    )
    if cached_queue_marker_failed:
        refresh_request = {**request, "queue_marker_refresh": True}
        return {
            "transition": {
                "transition_request": refresh_request,
                "transition_result": _transition_result(
                    refresh_request,
                    status="needs_ocr",
                    reason="queue_cached_marker_refresh_required",
                    visual_change_ratio=visual_ratio,
                    needs_ocr=True,
                ),
            },
            "collection": {
                "job_card_queue": release_active_job_card(
                    list(state["collection"].get("job_card_queue", []) or [])
                )
            },
        }
    if visual_changed or input_requires_ocr:
        return {
            "transition": {
                "no_effect_count": 0,
                "transition_result": _transition_result(
                    request,
                    status="needs_ocr",
                    reason=(
                        "ocr_required"
                        if visual_changed
                        else "input_ocr_required"
                    ),
                    visual_change_detected=visual_changed,
                    visual_change_ratio=visual_ratio,
                    needs_ocr=True,
                ),
            }
        }

    reason = (
        "reflex_no_screen_change"
        if source == "reflex"
        else "no_screen_change"
    )
    observation_update = _reused_observation(state, request)
    record_state = apply_worker_state_update(
        state,
        {"observation": observation_update},
    )
    record = _transition_record(
        request,
        status="unknown",
        source=source,
        reason=reason,
        attempt=1,
        state=record_state,
        visual_change_ratio=visual_ratio,
        ocr_skipped=True,
    )
    logger.info(
        "Transition no-effect detected before OCR",
        source=source,
        action=request.get("action", ""),
        visual_change_ratio=visual_ratio,
    )
    _record_replay_outcome(
        state,
        request,
        status="unknown",
        record_replay_result=record_replay_result,
    )
    update = {
        "transition": {
            "no_effect_count": (
                int(state["transition"].get("no_effect_count") or 0) + 1
            ),
            "transition_request": {},
            "transition_result": _transition_result(
                request,
                status="unknown",
                reason=reason,
                visual_change_ratio=visual_ratio,
            ),
            "action_events": attach_action_transition(
                state["transition"].get("action_events", []) or [],
                record,
            ),
        },
        "replay": {
            "reflex_blocked_recipe_keys": _blocked_keys_after_decision(
                state,
                request,
                should_block=source == "reflex",
            ),
            "replay_session": _replay_session_after_transition(
                state,
                source=source,
                status="unknown",
            ),
        },
        "observation": observation_update,
    }
    if source == "job_card_queue":
        update["collection"] = {
            "job_card_queue": release_active_job_card(
                list(state["collection"].get("job_card_queue", []) or [])
            )
        }
    return update


def _evaluate_after_ocr(
    state: WorkerState,
    request: dict[str, Any],
    *,
    visual_changed: bool,
    visual_ratio: float | None,
    record_replay_result,
) -> dict[str, Any]:
    source = str(request.get("source") or "")
    current_url = str(state["observation"].get("current_url") or "")
    before_url = str(request.get("before_url") or "")
    url_changed = bool(
        before_url
        and current_url
        and before_url != current_url
    )
    markers = list(state["observation"].get("current_markers") or [])
    input_confirmed = _input_text_confirmed_by_ocr(state, request)
    queue_target_reached = is_job_detail_context(
        current_url,
        page_role=str(
            state["observation"].get("current_page_role") or ""
        ),
        marker_texts=[
            marker.get("text")
            for marker in markers
            if isinstance(marker, dict)
        ],
    )

    if source == "reflex":
        matched, reason, after_state_match = _verify_reflex_after_state(
            request,
            state,
        )
        evaluated_request = {
            **request,
            "after_state_match": after_state_match,
        }
        status = "ready" if matched else "unknown"
        block_recipe = not matched
    elif source == "job_card_queue" and not queue_target_reached:
        evaluated_request = request
        status = "unknown"
        reason = "job_card_detail_not_reached"
        block_recipe = False
    elif markers and (url_changed or visual_changed or input_confirmed):
        evaluated_request = request
        status = "ready"
        reason = (
            "screen_change_pixels_matched"
            if visual_changed
            else (
                "screen_change_url_matched"
                if url_changed
                else "input_text_ocr_matched"
            )
        )
        block_recipe = False
    elif not url_changed and not visual_changed:
        evaluated_request = request
        status = "unknown"
        reason = "no_screen_change"
        block_recipe = False
    else:
        evaluated_request = request
        status = "unknown"
        reason = "transition_change_unverified"
        block_recipe = False

    attempt = 1
    record = _transition_record(
        evaluated_request,
        status=status,
        source=source,
        reason=reason,
        attempt=attempt,
        state=state,
        visual_change_ratio=visual_ratio,
        ocr_skipped=False,
    )
    logger.info(
        "Transition evaluated",
        source=source,
        status=status,
        reason=reason,
    )
    _record_replay_outcome(
        state,
        evaluated_request,
        status=status,
        record_replay_result=record_replay_result,
    )
    update = {
        "transition": {
            "no_effect_count": (
                0
                if status == "ready"
                else int(state["transition"].get("no_effect_count") or 0) + 1
            ),
            "transition_request": {},
            "transition_result": _transition_result(
                evaluated_request,
                status=status,
                reason=reason,
                visual_change_detected=visual_changed,
                visual_change_ratio=visual_ratio,
            ),
            "action_events": attach_action_transition(
                state["transition"].get("action_events", []) or [],
                record,
            ),
        },
        "replay": {
            "reflex_blocked_recipe_keys": _blocked_keys_after_decision(
                state,
                request,
                should_block=block_recipe,
            ),
            "replay_session": _replay_session_after_transition(
                state,
                source=source,
                status=status,
            ),
        },
    }
    if source == "job_card_queue" and status != "ready":
        update["collection"] = {
            "job_card_queue": release_active_job_card(
                list(state["collection"].get("job_card_queue", []) or [])
            )
        }
    return update


def transition_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> dict[str, Any]:
    """직전 원자 행동과 현재 캡처를 비교하고 OCR 필요 여부를 결정한다."""

    request = dict(
        state["transition"].get("transition_request", {}) or {}
    )
    initial_result = _result_without_transition(state, request)
    if initial_result is not None:
        return initial_result

    visual_changed, visual_ratio = transition_has_visual_change(
        request,
        str(state["observation"].get("current_screenshot") or ""),
    )
    if not state["observation"].get("ocr_complete"):
        return _evaluate_before_ocr(
            state,
            request,
            visual_changed=visual_changed,
            visual_ratio=visual_ratio,
            record_replay_result=runtime.context.data.record_recipe_replay,
        )
    return _evaluate_after_ocr(
        state,
        request,
        visual_changed=visual_changed,
        visual_ratio=visual_ratio,
        record_replay_result=runtime.context.data.record_recipe_replay,
    )


__all__ = ["transition_node"]

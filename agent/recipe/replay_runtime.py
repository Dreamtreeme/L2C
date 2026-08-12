"""첫 ROI가 일치하는 경험 기반 탐색 경로를 상태 전이 단위로 재생한다."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from langgraph.runtime import Runtime

from agent.recipe.phash_replay import match_target_by_screen_signature
from agent.runtime.site_context import normalize_page_role
from agent.runtime.target_matching import screen_context_signature_match
from agent.runtime.worker_contracts import (
    TransitionRequest,
    WorkerState,
    build_action_request,
)
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.recipe.replay import (
    ReflexReplayContext,
    ReflexSelection,
    load_reflex_replay_context,
    select_reflex_replay,
)
from agent.utils.logger import logger
from agent.utils.text import recipe_url_scope_matches, url_template
from shared.schema.recipe_schema import ReplaySession


def replay_session_from_state(state: WorkerState) -> ReplaySession | None:
    raw_session = state["replay"].get("replay_session")
    if isinstance(raw_session, ReplaySession):
        return raw_session
    return ReplaySession.model_validate(raw_session) if raw_session else None


def blocked_recipe_keys(state: WorkerState) -> list[str]:
    return [
        str(key)
        for key in (state["replay"].get("reflex_blocked_recipe_keys") or [])
        if str(key)
    ]


def replay_session_after_transition(
    state: WorkerState,
    *,
    source: str,
    status: str,
) -> ReplaySession | None:
    session = replay_session_from_state(state)
    if not session or source != "reflex":
        return session
    return session.advance() if status == "ready" else None


def blocked_recipe_keys_after(
    state: WorkerState,
    request: TransitionRequest,
    *,
    should_block: bool,
) -> list[str]:
    keys = blocked_recipe_keys(state)
    recipe_key = str(request.get("recipe_key") or "")
    if should_block and recipe_key and recipe_key not in keys:
        keys.append(recipe_key)
    return keys


def verify_replay_after_state(
    request: TransitionRequest,
    state: WorkerState,
) -> tuple[bool, str, dict[str, Any]]:
    """저장된 레시피 도착 화면과 현재 관찰이 같은지 확인한다."""

    if request.get("execution_failed"):
        return False, "recipe_action_group_failed", {}
    expected = request.get("expected_after_state")
    if expected is None:
        return False, "recipe_after_state_missing", {}

    expected_url = expected.url_template
    observation = state["observation"]
    current_url = str(observation.get("current_url") or "")
    if (
        expected_url
        and current_url
        and not recipe_url_scope_matches(expected_url, current_url)
    ):
        return (
            False,
            "recipe_after_url_mismatch",
            {
                "expected_url_template": expected_url,
                "current_url": current_url,
            },
        )

    before_role = normalize_page_role(request.get("before_page_role"))
    expected_role = normalize_page_role(expected.page_role)
    current_role = normalize_page_role(observation.get("current_page_role"))
    if before_role and expected_role and expected_role != before_role and current_role:
        matched = current_role == expected_role
        return (
            matched,
            (
                "recipe_after_page_role_matched"
                if matched
                else "recipe_after_page_role_mismatch"
            ),
            {
                "before_page_role": before_role,
                "expected_page_role": expected_role,
                "current_page_role": current_role,
            },
        )

    anchor_target = expected.anchor_target
    anchor_signature = dict(expected.anchor_roi_signature)
    if anchor_target is not None and anchor_signature:
        marker_id, match = match_target_by_screen_signature(
            anchor_target,
            anchor_signature,
            (observation.get("screen_signature") or {}).copy(),
            list(observation.get("current_markers") or []),
            current_image_path=str(observation.get("current_screenshot") or ""),
        )
        if marker_id is None:
            return (
                False,
                str(match.get("reason") or "recipe_after_anchor_mismatch"),
                match,
            )
        return (
            True,
            "recipe_after_anchor_matched",
            {
                **match,
                "marker_id": marker_id,
            },
        )

    context_signature = dict(expected.screen_context_signature)
    if context_signature:
        match = screen_context_signature_match(
            context_signature,
            (observation.get("screen_signature") or {}).copy(),
        )
        matched = bool(match.get("matched"))
        return (
            matched,
            (
                "recipe_after_context_matched"
                if matched
                else str(match.get("reason") or "recipe_after_context_mismatch")
            ),
            match,
        )

    before_url = url_template(str(request.get("before_url") or ""))
    if (
        expected_url
        and current_url
        and expected_url != before_url
        and recipe_url_scope_matches(expected_url, current_url)
    ):
        return (
            True,
            "recipe_after_url_matched",
            {
                "expected_url_template": expected_url,
                "current_url": current_url,
            },
        )
    return False, "recipe_after_state_unverifiable", {}


def record_replay_outcome(
    state: WorkerState,
    request: TransitionRequest,
    *,
    status: str,
    persist_result: Callable[[str, bool], object],
) -> None:
    """경로가 끝났거나 실패했을 때 한 번만 실제 재생 결과를 저장한다."""

    if str(request.get("source") or "") != "reflex":
        return
    session = replay_session_from_state(state)
    recipe_key = str(
        (session.recipe_key if session else "") or request.get("recipe_key") or ""
    )
    if not recipe_key or not session or not session.pending_is_current():
        return
    succeeded = status == "ready"
    if succeeded and not session.is_last_transition():
        return
    try:
        persist_result(recipe_key, succeeded)
    except Exception as exc:
        logger.warning(
            "Recipe replay outcome persistence failed",
            recipe_key=recipe_key,
            error=str(exc),
        )


def _miss_result(
    state: WorkerState,
    reason: str,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """현재 활성 경로를 차단하고 자율 탐색 폴백 상태를 만든다."""

    reflex_trace = dict(trace or {})
    reflex_trace.update(
        {
            "hit": False,
            "reason": reason,
        }
    )
    replay_session = replay_session_from_state(state)
    blocked_keys = blocked_recipe_keys(state)
    if replay_session and replay_session.recipe_key:
        active_recipe_key = replay_session.recipe_key
        reflex_trace.update(
            {
                "recipe_key": active_recipe_key,
                "recipe_transition_index": replay_session.current_transition_index,
                "recipe_transition_count": replay_session.transition_count,
                "path_failed": True,
            }
        )
        if active_recipe_key not in blocked_keys:
            blocked_keys.append(active_recipe_key)
    return {
        "replay": {
            "reflex_trace": reflex_trace,
            "replay_session": None,
            "reflex_blocked_recipe_keys": blocked_keys,
        },
    }


def _build_request(selection: ReflexSelection):
    """검증을 통과한 레시피 전이를 실행 요청으로 조립한다."""

    transition_count = len(selection.recipe.transitions)
    return build_action_request(
        "reflex",
        "cached recipe transition",
        selection.tool_calls,
        metadata={
            "execution_unit": "recipe_transition",
            "recipe_key": selection.recipe_key,
            "transition_index": selection.transition_index,
            "transition_count": transition_count,
            "before_state": selection.transition.before.model_dump(mode="json"),
            "expected_after_state": selection.transition.after.model_dump(mode="json"),
            "transition_actions": [
                str(action.action) for action in selection.transition.actions
            ],
        },
    )


def _hit_result(
    context: ReflexReplayContext,
    selection: ReflexSelection,
) -> dict[str, Any]:
    """요청, 활성 경로 상태와 관측 trace를 함께 만든다."""

    transition_count = len(selection.recipe.transitions)
    replay_session = ReplaySession(
        recipe_key=selection.recipe_key,
        current_transition_index=selection.transition_index,
        pending_transition_index=selection.transition_index,
        transition_count=transition_count,
        actions=[
            [str(action.action) for action in transition.actions]
            for transition in selection.recipe.transitions
        ],
    )
    return {
        "decision": {"pending_action": _build_request(selection)},
        "replay": {
            "reflex_trace": {
                "hit": True,
                "recipe_key": selection.recipe_key,
                "candidate_count": context.candidate_count,
                "task_category": context.task_category,
                "actions": [call["name"] for call in selection.tool_calls],
                "tool_calls": selection.tool_call_traces,
                "recipe_transition_index": selection.transition_index,
                "recipe_transition_count": transition_count,
            },
            "replay_session": replay_session,
        },
    }


def attempt_reflex_replay(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> dict[str, Any]:
    """후보 조회, 검증과 요청 조립을 순서대로 실행한다."""

    started = time.perf_counter()
    logger.info("Executing Reflex Node")
    context = load_reflex_replay_context(
        state,
        runtime.context.data.load_site_recipes,
    )
    if not context.recipe_candidates:
        logger.info(
            "Reflex miss: no recipe",
            site=context.site,
            task_category=context.task_category,
        )
        return _miss_result(
            state,
            "no_recipe",
            {
                "candidate_count": 0,
                "site": context.site,
                "task_category": context.task_category,
            },
        )

    selection, rejection_log = select_reflex_replay(state, context)
    if selection is None:
        trace = rejection_log.trace_payload(context.candidate_count)
        logger.info(
            "Reflex miss: no candidate passed marker matching",
            candidates=context.candidate_count,
            last_reason=rejection_log.last_reason,
            reject_reasons=rejection_log.reason_counts,
            candidate_rejections=rejection_log.candidates[:12],
        )
        return _miss_result(state, "no_candidate_passed", trace)

    elapsed = time.perf_counter() - started
    logger.info(
        "Reflex hit",
        recipe_key=selection.recipe_key[:24],
        actions=[call["name"] for call in selection.tool_calls],
        recipe_transition=(
            f"{selection.transition_index + 1}/{len(selection.recipe.transitions)}"
            if len(selection.recipe.transitions) > 1
            else ""
        ),
        goal=selection.recipe.goal[:80],
        duration=f"{elapsed:.3f}s",
    )
    return _hit_result(context, selection)


__all__ = [
    "attempt_reflex_replay",
    "blocked_recipe_keys_after",
    "record_replay_outcome",
    "replay_session_after_transition",
    "verify_replay_after_state",
]

"""첫 ROI가 일치하는 경험 기반 탐색 경로를 상태 전이 단위로 재생한다."""

from __future__ import annotations

import time
from typing import Any

from langgraph.runtime import Runtime

from agent.runtime.worker_contracts import WorkerState, build_action_request
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.recipe.replay import (
    ReflexReplayContext,
    ReflexSelection,
    load_reflex_replay_context,
    select_reflex_replay,
)
from agent.utils.logger import logger
from agent.utils.model_dump import dump_model


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
            "reason": reason or reflex_trace.get("reason", ""),
        }
    )
    active_recipe = dict(
        state["replay"].get("active_reflex_recipe", {}) or {}
    )
    blocked_keys = [
        str(key)
        for key in (
            state["replay"].get("reflex_blocked_recipe_keys") or []
        )
        if str(key)
    ]
    active_recipe_key = str(active_recipe.get("recipe_key") or "")
    if active_recipe_key:
        reflex_trace.update(
            {
                "recipe_key": active_recipe_key,
                "recipe_transition_index": int(
                    active_recipe.get("current_transition_index") or 0
                ),
                "recipe_transition_count": int(
                    active_recipe.get("transition_count") or 0
                ),
                "path_failed": True,
            }
        )
        if active_recipe_key not in blocked_keys:
            blocked_keys.append(active_recipe_key)
    return {
        "replay": {
            "reflex_trace": reflex_trace,
            "active_reflex_recipe": {},
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
            "before_state": dump_model(selection.transition.before),
            "expected_after_state": dump_model(selection.transition.after),
            "transition_actions": [
                str(action.action)
                for action in selection.transition.actions
            ],
        },
    )


def _hit_result(
    context: ReflexReplayContext,
    selection: ReflexSelection,
) -> dict[str, Any]:
    """요청, 활성 경로 상태와 관측 trace를 함께 만든다."""

    transition_count = len(selection.recipe.transitions)
    active_recipe_state = {
        "recipe_key": selection.recipe_key,
        "current_transition_index": selection.transition_index,
        "pending_transition_index": selection.transition_index,
        "transition_count": transition_count,
        "actions": [
            [str(action.action) for action in transition.actions]
            for transition in selection.recipe.transitions
        ],
    }
    return {
        "decision": {"pending_action": _build_request(selection)},
        "replay": {
            "reflex_trace": {
                "hit": True,
                "recipe_key": selection.recipe_key,
                "candidate_count": context.candidate_count,
                "task_category": context.task_category,
                "actions": [
                    call["name"]
                    for call in selection.tool_calls
                ],
                "tool_calls": selection.tool_call_traces,
                "recipe_transition_index": selection.transition_index,
                "recipe_transition_count": transition_count,
            },
            "active_reflex_recipe": active_recipe_state,
        },
    }


def attempt_reflex_replay(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> dict[str, Any]:
    """후보 조회, 검증과 요청 조립을 순서대로 실행한다."""

    started = time.perf_counter()
    logger.info("Executing Reflex Node")
    try:
        context = load_reflex_replay_context(state)
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
                f"{selection.transition_index + 1}/"
                f"{len(selection.recipe.transitions)}"
                if len(selection.recipe.transitions) > 1
                else ""
            ),
            when_to_use=getattr(
                getattr(selection.recipe, "skill_metadata", None),
                "when_to_use",
                "",
            )[:80],
            duration=f"{elapsed:.3f}s",
        )
        return _hit_result(context, selection)
    except Exception as exc:
        logger.debug("reflex node skipped", error=str(exc))
        return _miss_result(
            state,
            "exception",
            {"error": str(exc)},
        )


__all__ = ["attempt_reflex_replay"]

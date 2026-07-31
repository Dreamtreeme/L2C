"""Reflex 안정 경로의 선택과 전환 결과를 같은 관측 계약으로 정리한다."""

from __future__ import annotations

from typing import Any


def reflex_selection_observation(result: dict[str, Any]) -> dict[str, Any]:
    """Reflex 노드 결과에서 경로 선택 또는 중간 실패 정보를 추출한다."""

    trace = dict(result.get("reflex_trace") or {})
    recipe_key = str(trace.get("recipe_key") or "")
    if not recipe_key:
        return {}

    transition_index = _optional_int(
        trace.get("recipe_transition_index")
    )
    transition_count = _optional_int(
        trace.get("recipe_transition_count")
    )
    observation = {
        "reflex_recipe_key": recipe_key,
        "reflex_path_transition_index": transition_index,
        "reflex_path_transition_count": transition_count,
    }
    if trace.get("hit"):
        observation["reflex_path_event"] = (
            "started"
            if transition_index in (None, 0)
            else "transition_selected"
        )
        return observation

    if trace.get("path_failed"):
        observation.update(
            reflex_path_event="failed",
            reflex_path_failure_reason=str(trace.get("reason") or "match_failed"),
            reflex_fallback_required=True,
        )
        return observation
    return {}


def reflex_transition_observation(result: dict[str, Any]) -> dict[str, Any]:
    """전환 판정에서 Reflex 경로 단계의 완료 여부를 추출한다."""

    transition = dict(result.get("transition_result") or {})
    if str(transition.get("source") or "") != "reflex":
        return {}
    recipe_key = str(transition.get("recipe_key") or "")
    if not recipe_key:
        return {}

    status = str(transition.get("status") or "")
    transition_index = _optional_int(
        transition.get("recipe_transition_index")
    )
    transition_count = _optional_int(
        transition.get("recipe_transition_count")
    )
    observation = {
        "reflex_recipe_key": recipe_key,
        "reflex_path_transition_index": transition_index,
        "reflex_path_transition_count": transition_count,
        "reflex_transition_status": status,
    }
    if status == "unknown":
        observation.update(
            reflex_path_event="failed",
            reflex_path_failure_reason=str(
                transition.get("reason") or "transition_unknown"
            ),
            reflex_fallback_required=True,
        )
    elif (
        status == "ready"
        and transition_index is not None
        and transition_count is not None
        and transition_index + 1 >= transition_count
    ):
        observation["reflex_path_event"] = "completed"
    elif status == "ready":
        observation["reflex_path_event"] = "transition_completed"
    return observation


def summarize_reflex_paths(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """단계 hit와 경로 전체 성과를 분리해 집계한다."""

    reflex_steps = [
        item
        for item in steps
        if item.get("component") == "graph:reflex"
        and item.get("action_source") == "reflex"
    ]
    path_events = [
        item
        for item in steps
        if str(item.get("reflex_path_event") or "")
    ]
    started = sum(item.get("reflex_path_event") == "started" for item in path_events)
    completed = sum(item.get("reflex_path_event") == "completed" for item in path_events)
    failed = sum(item.get("reflex_path_event") == "failed" for item in path_events)
    mid_path_failed = sum(
        item.get("reflex_path_event") == "failed"
        and int(
            item.get("reflex_path_transition_index") or 0
        )
        > 0
        for item in path_events
    )
    return {
        "reflex_transition_hit_count": len(reflex_steps),
        "reflex_path_started_count": started,
        "reflex_path_completed_count": completed,
        "reflex_path_failed_count": failed,
        "reflex_path_mid_failure_count": mid_path_failed,
        "reflex_path_fallback_count": sum(
            bool(item.get("reflex_fallback_required")) for item in path_events
        ),
        "reflex_path_incomplete_count": max(0, started - completed - failed),
        "reflex_path_completion_rate": (
            round(completed / started, 6) if started else None
        ),
    }


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "reflex_selection_observation",
    "reflex_transition_observation",
    "summarize_reflex_paths",
]

"""경험 규칙 경로의 선택과 단계 결과를 같은 관측 계약으로 정리한다."""

from __future__ import annotations

from typing import Any


def reflex_selection_observation(result: dict[str, Any]) -> dict[str, Any]:
    """Reflex 노드 결과에서 경로 선택 또는 중간 실패 정보를 추출한다."""

    trace = dict((result.get("replay") or {}).get("reflex_trace") or {})
    recipe_key = str(trace.get("recipe_key") or "")
    resolver_calls = _optional_int(trace.get("resolver_reasoning_call_count")) or 0
    if not recipe_key:
        if not trace.get("hit") and resolver_calls:
            return {
                "reflex_resolver_reasoning_call_count": resolver_calls,
                "reflex_reasoning_call_reduction": -resolver_calls,
            }
        return {}

    step_index = _optional_int(trace.get("recipe_step_index"))
    step_count = _optional_int(trace.get("recipe_step_count"))
    observation = {
        "reflex_recipe_key": recipe_key,
        "reflex_path_step_index": step_index,
        "reflex_path_step_count": step_count,
    }
    if trace.get("hit"):
        observation["reflex_path_event"] = (
            "started"
            if step_index in (None, 0)
            else "step_selected"
        )
        return observation

    if trace.get("path_failed"):
        observation.update(
            reflex_path_event="failed",
            reflex_path_failure_reason=str(trace.get("reason") or "match_failed"),
            reflex_fallback_required=True,
            reflex_resolver_reasoning_call_count=resolver_calls,
            reflex_reasoning_call_reduction=-resolver_calls,
        )
        return observation
    return {}


def reflex_step_observation(result: dict[str, Any]) -> dict[str, Any]:
    """화면 효과 판정에서 경험 경로 단계의 완료 여부를 추출한다."""

    transition = dict(
        (result.get("transition") or {}).get("transition_result") or {}
    )
    if str(transition.get("source") or "") != "reflex":
        return {}
    recipe_key = str(transition.get("recipe_key") or "")
    if not recipe_key:
        return {}

    status = str(transition.get("status") or "")
    source_calls = _optional_int(transition.get("source_reasoning_call_count")) or 0
    resolver_calls = (
        _optional_int(transition.get("resolver_reasoning_call_count")) or 0
    )
    succeeded = status == "ready"
    terminal = status in {"ready", "unknown"}
    step_index = _optional_int(transition.get("recipe_step_index"))
    step_count = _optional_int(transition.get("recipe_step_count"))
    observation = {
        "reflex_recipe_key": recipe_key,
        "reflex_path_step_index": step_index,
        "reflex_path_step_count": step_count,
        "reflex_step_status": status,
    }
    if terminal:
        observation.update(
            reflex_source_reasoning_replaced_count=(
                source_calls if succeeded else 0
            ),
            reflex_resolver_reasoning_call_count=resolver_calls,
            reflex_reasoning_call_reduction=(
                source_calls - resolver_calls if succeeded else -resolver_calls
            ),
        )
    if status == "unknown":
        observation.update(
            reflex_path_event="failed",
            reflex_path_failure_reason=str(
                transition.get("reason") or "step_effect_unknown"
            ),
            reflex_fallback_required=True,
        )
    elif (
        status == "ready"
        and step_index is not None
        and step_count is not None
        and step_index + 1 >= step_count
    ):
        observation["reflex_path_event"] = "completed"
    elif status == "ready":
        observation["reflex_path_event"] = "step_completed"
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
            item.get("reflex_path_step_index") or 0
        )
        > 0
        for item in path_events
    )
    return {
        "reflex_step_hit_count": len(reflex_steps),
        "reflex_step_completed_count": sum(
            item.get("reflex_path_event") in {"step_completed", "completed"}
            for item in path_events
        ),
        "reflex_source_reasoning_replaced_count": sum(
            int(item.get("reflex_source_reasoning_replaced_count") or 0)
            for item in steps
        ),
        "reflex_resolver_reasoning_call_count": sum(
            int(item.get("reflex_resolver_reasoning_call_count") or 0)
            for item in steps
        ),
        "reflex_reasoning_call_reduction": sum(
            int(item.get("reflex_reasoning_call_reduction") or 0)
            for item in steps
        ),
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
    "reflex_step_observation",
    "summarize_reflex_paths",
]

"""작업자 그래프가 사용하는 장기 실행 비전 자원의 단일 접근점."""

from __future__ import annotations

from typing import Any

from agent.graph.state import GraphState


def get_perception() -> Any:
    from agent.runtime.vision_worker_runtime import current_vision_worker_runtime

    return current_vision_worker_runtime().get_perception()


def get_action_tools() -> Any:
    from agent.runtime.vision_worker_runtime import current_vision_worker_runtime

    return current_vision_worker_runtime().get_action_tools()


def check_current_reasoning_screen(
    state: GraphState,
    marker_id: int | None = None,
) -> dict[str, Any]:
    """저장된 화면 서명이 있을 때 행동 대상 주변을 다시 검사한다."""

    from agent.runtime.action_guard import (
        check_reasoning_screen_stale,
        reasoning_screen_guard_enabled,
    )

    if not reasoning_screen_guard_enabled():
        return {
            "checked": False,
            "stale": False,
            "must_refresh": False,
            "reason": "disabled",
        }
    if not str((state.get("screen_signature") or {}).get("phash") or ""):
        return {
            "checked": False,
            "stale": False,
            "must_refresh": True,
            "reason": "previous_phash_missing",
        }

    return check_reasoning_screen_stale(
        state,
        get_perception(),
        marker_id=marker_id,
    )


def prepare_reasoning_models() -> None:
    """브라우저 준비 중 범용 판단 모델과 카드 선택 모델을 미리 생성한다."""

    from agent.graph.tool_schema import ACTION_TOOL_SCHEMAS
    from agent.runtime.vision_worker_runtime import current_vision_worker_runtime

    current_vision_worker_runtime().prepare_reasoning_models(ACTION_TOOL_SCHEMAS)


__all__ = [
    "check_current_reasoning_screen",
    "get_action_tools",
    "get_perception",
    "prepare_reasoning_models",
]

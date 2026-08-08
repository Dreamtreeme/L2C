"""기록·재생 가능한 작업자 행동 종류 계약."""

from __future__ import annotations

from typing import Any

TARGET_REPLAY_ACTIONS = frozenset(
    {
        "click_marker",
        "type_in_marker",
    }
)
CONTEXTUAL_REPLAY_ACTIONS = frozenset(
    {
        "press_key",
        "go_back",
        "close_current_tab",
        "switch_tab",
    }
)
REVIEWABLE_REPLAY_ACTIONS = (
    TARGET_REPLAY_ACTIONS | CONTEXTUAL_REPLAY_ACTIONS
)
RECORDED_REPLAY_ACTIONS = (
    REVIEWABLE_REPLAY_ACTIONS | {"scroll"}
)


def _action_value(action: Any, key: str, default: Any = None) -> Any:
    if isinstance(action, dict):
        return action.get(key, default)
    return getattr(action, key, default)


def is_supported_recipe_action_group(actions: list[Any]) -> bool:
    """중간 화면 관찰 없이 실행해도 되는 최소 행동 묶음인지 확인한다."""

    if len(actions) == 1:
        return str(_action_value(actions[0], "action") or "") in REVIEWABLE_REPLAY_ACTIONS
    if len(actions) != 2:
        return False
    first, second = actions
    if (
        str(_action_value(first, "action") or "") != "type_in_marker"
        or str(_action_value(second, "action") or "") != "press_key"
    ):
        return False
    param = _action_value(second, "param", {})
    if not isinstance(param, dict):
        return False
    return str(param.get("key") or "").strip().casefold() in {
        "enter",
        "return",
    }


__all__ = [
    "CONTEXTUAL_REPLAY_ACTIONS",
    "RECORDED_REPLAY_ACTIONS",
    "REVIEWABLE_REPLAY_ACTIONS",
    "TARGET_REPLAY_ACTIONS",
    "is_supported_recipe_action_group",
]

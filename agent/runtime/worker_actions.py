"""작업자 행동 종류, 기록·재생 범위와 화면 전환 특성의 공통 계약."""

from __future__ import annotations

from typing import Any


TARGET_REPLAY_ACTIONS = frozenset(
    {
        "click_marker",
        "type_in_marker",
    }
)

RETURN_ACTIONS = frozenset(
    {
        "go_back",
        "close_current_tab",
        "switch_tab",
    }
)

CONTEXTUAL_REPLAY_ACTIONS = RETURN_ACTIONS | {"press_key"}
REVIEWABLE_REPLAY_ACTIONS = (
    TARGET_REPLAY_ACTIONS | CONTEXTUAL_REPLAY_ACTIONS
)
RECORDED_REPLAY_ACTIONS = REVIEWABLE_REPLAY_ACTIONS | {"scroll"}

UI_ACTIONS = frozenset(
    RECORDED_REPLAY_ACTIONS | {"open_browser"}
)

STATE_UPDATE_ACTIONS = frozenset(
    {
        "finish_detail_reading",
        "set_job_card_queue",
    }
)

TERMINAL_ACTIONS = frozenset({"finish_task"})

URL_STALE_ACTIONS = UI_ACTIONS - {"type_in_marker", "scroll"}

DIRECT_SCREEN_ACTION_SOURCES = frozenset(
    {
        "reflex",
        "job_card_queue",
        "page_policy",
        "duplicate_job_policy",
    }
)


def _action_value(action: Any, key: str, default: Any = None) -> Any:
    if isinstance(action, dict):
        return action.get(key, default)
    return getattr(action, key, default)


def is_supported_recipe_action_group(actions: list[Any]) -> bool:
    """중간 화면 관찰 없이 실행할 수 있는 행동 묶음인지 확인한다."""

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
    "DIRECT_SCREEN_ACTION_SOURCES",
    "RECORDED_REPLAY_ACTIONS",
    "RETURN_ACTIONS",
    "REVIEWABLE_REPLAY_ACTIONS",
    "STATE_UPDATE_ACTIONS",
    "TARGET_REPLAY_ACTIONS",
    "TERMINAL_ACTIONS",
    "UI_ACTIONS",
    "URL_STALE_ACTIONS",
    "is_supported_recipe_action_group",
]

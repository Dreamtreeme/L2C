"""작업자 행동 종류, 기록·재생 범위와 화면 전환 특성의 공통 계약."""

from __future__ import annotations

from collections.abc import Sequence

from shared.schema.recipe_schema import (
    NAVIGATION_ACTIONS,
    RECIPE_COMMIT_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
    TRAJECTORY_ACTIONS,
    UI_ACTIONS,
    PhysicalAction,
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


def is_supported_recipe_action_group(actions: Sequence[PhysicalAction]) -> bool:
    """중간 화면 관찰 없이 실행할 수 있는 행동 묶음인지 확인한다."""

    if len(actions) == 1:
        return actions[0].action in TARGET_REPLAY_ACTIONS
    if len(actions) != 2:
        return False
    first, second = actions
    if first.action != "type_in_marker":
        return False
    second_action = second.action
    if second_action == "click_marker":
        return True
    if second_action != "press_key":
        return False
    return second.param.key.strip().casefold() in {
        "enter",
        "return",
    }


def is_supported_recipe_tool_group(
    action_names: Sequence[str],
    *,
    commit_key: str = "",
) -> bool:
    """검증된 도구 호출 이름 묶음이 연속 실행 가능한지 확인한다."""

    if len(action_names) == 1:
        return action_names[0] in TARGET_REPLAY_ACTIONS
    if len(action_names) != 2 or action_names[0] != "type_in_marker":
        return False
    if action_names[1] == "click_marker":
        return True
    return bool(
        action_names[1] == "press_key"
        and commit_key.strip().casefold() in {"enter", "return"}
    )


__all__ = [
    "DIRECT_SCREEN_ACTION_SOURCES",
    "NAVIGATION_ACTIONS",
    "RECIPE_COMMIT_ACTIONS",
    "REVIEWABLE_REPLAY_ACTIONS",
    "STATE_UPDATE_ACTIONS",
    "TARGET_REPLAY_ACTIONS",
    "TERMINAL_ACTIONS",
    "TRAJECTORY_ACTIONS",
    "UI_ACTIONS",
    "URL_STALE_ACTIONS",
    "is_supported_recipe_action_group",
    "is_supported_recipe_tool_group",
]

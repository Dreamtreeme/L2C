"""Reflex 후보 기록과 승격에서 사용하는 행동 종류 계약."""

from __future__ import annotations

from typing import Any

from agent.recipe.page_context import normalize_page_role


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


def _step_sequence(step: dict[str, Any]) -> int:
    """정렬할 수 없는 단계 번호는 기록 앞쪽의 기본값으로 처리한다."""

    try:
        return int(step.get("seq") or 0)
    except (TypeError, ValueError):
        return 0


def group_replay_action_sets(
    steps: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """같은 화면의 입력과 즉시 이어지는 클릭을 하나의 전환 단위로 묶는다."""

    ordered = [
        dict(step)
        for step in sorted(
            (item for item in steps or [] if isinstance(item, dict)),
            key=_step_sequence,
        )
    ]
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(ordered):
        current = ordered[index]
        following = ordered[index + 1] if index + 1 < len(ordered) else None
        if (
            following is not None
            and str(current.get("action") or "") == "type_in_marker"
            and str(following.get("action") or "") == "click_marker"
            and _step_sequence(following)
            == _step_sequence(current) + 1
            and normalize_page_role(current.get("page_role"))
            == normalize_page_role(following.get("page_role"))
            and str(current.get("url_template") or "")
            == str(following.get("url_template") or "")
        ):
            groups.append([current, following])
            index += 2
            continue
        groups.append([current])
        index += 1
    return groups


__all__ = [
    "CONTEXTUAL_REPLAY_ACTIONS",
    "RECORDED_REPLAY_ACTIONS",
    "REVIEWABLE_REPLAY_ACTIONS",
    "TARGET_REPLAY_ACTIONS",
    "group_replay_action_sets",
]

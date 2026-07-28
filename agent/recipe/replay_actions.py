"""Reflex 후보 기록과 승격에서 사용하는 행동 종류 계약."""

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


__all__ = [
    "CONTEXTUAL_REPLAY_ACTIONS",
    "RECORDED_REPLAY_ACTIONS",
    "REVIEWABLE_REPLAY_ACTIONS",
    "TARGET_REPLAY_ACTIONS",
]

"""Reflex 후보 기록과 승격에서 사용하는 행동 종류 계약."""

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


def _step_sequence(step: dict[str, Any]) -> int:
    """정렬할 수 없는 단계 번호는 기록 앞쪽의 기본값으로 처리한다."""

    try:
        return int(step.get("seq") or 0)
    except (TypeError, ValueError):
        return 0


def split_stable_replay_paths(
    steps: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """연속해서 승인된 단계를 순서가 보존된 안정 경로로 나눈다."""

    ordered = [
        dict(step)
        for step in sorted(
            (item for item in steps or [] if isinstance(item, dict)),
            key=_step_sequence,
        )
    ]
    paths: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def finish_current() -> None:
        nonlocal current
        while current and str(current[0].get("action") or "") not in TARGET_REPLAY_ACTIONS:
            current.pop(0)
        if current:
            paths.append(current)
        current = []

    previous_seq: int | None = None
    for step in ordered:
        seq = _step_sequence(step)
        if current and previous_seq is not None and seq != previous_seq + 1:
            finish_current()
        current.append(step)
        previous_seq = seq
    finish_current()
    return paths


__all__ = [
    "CONTEXTUAL_REPLAY_ACTIONS",
    "RECORDED_REPLAY_ACTIONS",
    "REVIEWABLE_REPLAY_ACTIONS",
    "TARGET_REPLAY_ACTIONS",
    "split_stable_replay_paths",
]

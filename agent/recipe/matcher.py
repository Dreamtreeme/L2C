"""Reflex Recipe 마커 매칭."""

from __future__ import annotations

from agent.runtime.site_context import normalize_page_role
from agent.runtime.worker_contracts import ScreenMarker
from agent.runtime.worker_actions import (
    RECIPE_COMMIT_ACTIONS,
    TARGET_REPLAY_ACTIONS,
)
from agent.utils.text import normalize_text
from agent.vision.marker_geometry import marker_center
from shared.schema.recipe_schema import PhysicalAction, ScreenCheckpoint


def marker_region(marker: ScreenMarker, markers: list[ScreenMarker]) -> str:
    """현재 마커 집합 안에서 coarse 3x3 영역을 계산한다."""
    if not marker:
        return ""
    xs = []
    ys = []
    for item in markers or []:
        if isinstance(item, dict):
            x, y = marker_center(item)
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return ""

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x, y = marker_center(marker)

    def band(value: int, low: int, high: int, names: tuple[str, str, str]) -> str:
        span = max(1, high - low)
        ratio = (value - low) / span
        if ratio < 1 / 3:
            return names[0]
        if ratio < 2 / 3:
            return names[1]
        return names[2]

    return f"{band(y, min_y, max_y, ('top', 'middle', 'bottom'))}-{band(x, min_x, max_x, ('left', 'center', 'right'))}"


def is_replayable_action(
    action: PhysicalAction,
    checkpoint: ScreenCheckpoint,
) -> bool:
    """저장된 물리 행동이 결정론적 재생 계약을 갖췄는지 확인한다."""

    if not normalize_page_role(checkpoint.page_role):
        return False
    if not action.is_supported_replay_action():
        return False
    if action.action in RECIPE_COMMIT_ACTIONS:
        return True
    if action.action not in TARGET_REPLAY_ACTIONS or action.target is None:
        return False
    return bool(
        normalize_text(action.target.text)
        or normalize_text(action.target.semantic_label)
    )

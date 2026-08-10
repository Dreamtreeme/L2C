"""Reflex Recipe 마커 매칭."""

from __future__ import annotations

from agent.runtime.site_context import normalize_page_role
from agent.runtime.worker_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
)
from agent.utils.text import normalize_text
from agent.vision.marker_geometry import marker_center


def marker_region(marker: dict, markers: list[dict]) -> str:
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


def is_replayable_action(action_item: dict) -> bool:
    """저장된 물리 행동이 결정론적 재생 계약을 갖췄는지 확인한다."""

    if action_item.get("replay_mode", "reasoning") == "reasoning":
        return False
    action = action_item.get("action")
    if not normalize_page_role(action_item.get("page_role", "")):
        return False
    if action in CONTEXTUAL_REPLAY_ACTIONS:
        param = action_item.get("param", {})
        if not isinstance(param, dict):
            return False
        if action == "press_key" and not param.get("key"):
            return False
        if action == "switch_tab" and not param.get("direction"):
            return False
        return bool(
            action_item.get("screen_context_signature")
        )
    if action not in TARGET_REPLAY_ACTIONS:
        return False
    target = action_item.get("target") or {}
    return isinstance(target, dict) and bool(
        normalize_text(target.get("text"))
        or normalize_text(target.get("semantic_label"))
    )

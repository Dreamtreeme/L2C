"""자율탐색 화면과 물리 행동을 공통 경험 계약으로 기록한다."""

from __future__ import annotations

from typing import Any

from agent.recipe.matcher import marker_region
from agent.runtime.site_context import normalize_page_role
from agent.runtime.worker_actions import (
    REVIEWABLE_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
    UI_ACTIONS,
)
from agent.runtime.worker_contracts import WorkerState
from agent.utils.text import normalize_text, url_template
from agent.vision.marker_geometry import marker_bbox
from agent.vision.screen_signature import (
    compact_screen_context_signature,
    compute_target_roi_signature,
)
from agent.vision.target_snapshot import build_marker_target_snapshot, marker_by_id
from shared.schema.recipe_schema import (
    ActionTarget,
    PhysicalAction,
    ScreenCheckpoint,
)


def _replay_mode(action_name: str, args: dict, slot_name: str) -> str:
    if action_name not in REVIEWABLE_REPLAY_ACTIONS:
        return "reasoning"
    if action_name == "type_in_marker":
        return "parameterized" if slot_name else "reasoning"
    if action_name == "press_key":
        key = normalize_text(args.get("key")).casefold()
        return "fixed" if key in {"enter", "return"} else "reasoning"
    return "fixed"


def _input_slot(state: WorkerState, action_name: str, args: dict) -> str:
    slot_name = normalize_text(args.get("slot_name"))
    if action_name != "type_in_marker" or slot_name:
        return slot_name
    intent = state["request"].get("collection_intent")
    search_keyword = normalize_text(
        intent.get("search_keyword", "")
        if isinstance(intent, dict)
        else getattr(intent, "search_keyword", "")
    )
    if search_keyword and normalize_text(args.get("text")) == search_keyword:
        return "search_keyword"
    return ""


def build_screen_checkpoint(
    state: WorkerState,
    *,
    observation_id: str = "",
    current_url: str = "",
) -> ScreenCheckpoint:
    """현재 캡처를 행동 직전 화면 상태로 만든다."""

    observation = state["observation"]
    resolved_url = current_url or str(observation.get("current_url") or "")
    return ScreenCheckpoint(
        observation_id=(observation_id or str(observation.get("observation_id") or "")),
        url_template=url_template(resolved_url),
        page_role=(normalize_page_role(observation.get("current_page_role"))),
        screen_context_signature=compact_screen_context_signature(
            observation.get("screen_signature") or {}
        ),
    )


def _target(
    state: WorkerState,
    args: dict,
) -> tuple[ActionTarget | None, dict]:
    observation = state["observation"]
    markers = list(observation.get("current_markers") or [])
    marker = marker_by_id(markers, args.get("marker_id"))
    if not marker:
        return None, {}

    screen_signature = (observation.get("screen_signature") or {}).copy()
    snapshot = (
        build_marker_target_snapshot(
            markers,
            args.get("marker_id"),
            screen_signature=screen_signature,
        )
        or {}
    )
    target = ActionTarget(
        text=normalize_text(marker.get("text")),
        semantic_label=normalize_text(args.get("target_label")) or None,
        region=marker_region(marker, markers),
        marker_type=normalize_text(marker.get("type")),
        bbox_ratio=list(snapshot.get("bbox_ratio") or []),
        center_ratio=list(snapshot.get("center_ratio") or []),
    )

    screen_size = screen_signature.get("size") or []
    image_path = str(observation.get("current_screenshot") or "")
    if not image_path or not isinstance(screen_size, list) or len(screen_size) != 2:
        return target, {}
    roi_signature = compute_target_roi_signature(
        image_path,
        marker_bbox(marker),
        screen_size,
        capture_context=(screen_signature.get("capture_context") or {}).copy(),
    )
    return target, roi_signature


def _action_param(
    action_name: str,
    args: dict[str, Any],
    slot_name: str,
) -> dict[str, Any]:
    if action_name == "type_in_marker":
        param: dict[str, Any] = {"text": str(args.get("text") or "").strip()}
        if slot_name:
            param["slot_name"] = slot_name
        return param
    names = {
        "press_key": ("key",),
        "scroll": ("direction", "amount"),
        "switch_tab": ("direction",),
        "open_browser": ("url",),
    }.get(action_name, ())
    param = {name: args.get(name) for name in names if args.get(name) not in (None, "")}
    if action_name == "scroll":
        param.setdefault("direction", "down")
        param.setdefault("amount", "page")
    return param


def build_physical_action(
    state: WorkerState,
    action_name: str,
    args: dict,
    seq: int,
) -> PhysicalAction | None:
    """실행한 UI 도구를 경험 전이에 넣을 물리 행동으로 만든다."""

    if action_name not in UI_ACTIONS:
        return None
    slot_name = _input_slot(state, action_name, args)
    target = None
    roi_signature: dict = {}
    if action_name in TARGET_REPLAY_ACTIONS or args.get("marker_id") is not None:
        target, roi_signature = _target(state, args)

    replay_mode = _replay_mode(action_name, args, slot_name)
    if action_name in TARGET_REPLAY_ACTIONS and target is None:
        replay_mode = "reasoning"
    return PhysicalAction(
        source_seq=seq,
        action=action_name,
        target=target,
        roi_signature=roi_signature,
        param=_action_param(action_name, args, slot_name),
        intent=normalize_text(args.get("reason")),
        target_role=normalize_text(args.get("target_role")),
        component=normalize_text(args.get("target_component")),
        slot_refs=[slot_name] if slot_name else [],
        risk_level=normalize_text(args.get("risk_level")),
        replay_mode=replay_mode,
    )


__all__ = ["build_physical_action", "build_screen_checkpoint"]

"""자율탐색 화면과 물리 행동을 공통 경험 계약으로 기록한다."""

from __future__ import annotations

from typing import Any

from agent.runtime.site_context import normalize_page_role
from agent.runtime.worker_actions import (
    TARGET_REPLAY_ACTIONS,
    UI_ACTIONS,
)
from agent.runtime.worker_contracts import ScreenMarker, WorkerState
from agent.utils.text import normalize_text, url_template
from agent.vision.marker_geometry import marker_bbox, marker_center
from agent.vision.screen_signature import (
    compact_screen_context_signature,
    compute_target_roi_signature,
)
from agent.vision.target_snapshot import build_marker_target_snapshot, marker_by_id
from shared.schema.execution_record_schema import (
    ActionTarget,
    ObservedAction,
    ScreenCheckpoint,
)


def _marker_region(marker: ScreenMarker, markers: list[ScreenMarker]) -> str:
    """현재 마커 집합에서 대상의 대략적인 3x3 화면 영역을 계산한다."""

    centers = [marker_center(item) for item in markers]
    if not centers:
        return ""
    xs, ys = zip(*centers)
    x, y = marker_center(marker)

    def band(value: int, low: int, high: int, names: tuple[str, str, str]) -> str:
        ratio = (value - low) / max(1, high - low)
        return names[0] if ratio < 1 / 3 else names[1] if ratio < 2 / 3 else names[2]

    vertical = band(y, min(ys), max(ys), ("top", "middle", "bottom"))
    horizontal = band(x, min(xs), max(xs), ("left", "center", "right"))
    return f"{vertical}-{horizontal}"


def _input_slot(action_name: str, args: dict) -> str:
    """모델이 실제 도구 호출에 명시한 입력 슬롯만 기록한다."""

    slot_name = normalize_text(args.get("slot_name"))
    return slot_name if action_name == "type_in_marker" else ""


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
        region=_marker_region(marker, markers),
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
) -> ObservedAction | None:
    """실행한 UI 도구를 경험 전이에 넣을 물리 행동으로 만든다."""

    if action_name not in UI_ACTIONS:
        return None
    slot_name = _input_slot(action_name, args)
    target = None
    roi_signature: dict = {}
    if action_name in TARGET_REPLAY_ACTIONS or args.get("marker_id") is not None:
        target, roi_signature = _target(state, args)

    return ObservedAction(
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
    )


__all__ = ["build_physical_action", "build_screen_checkpoint"]

"""
비전 실행의 UI 행동과 타깃 ROI를 기록한다.
"""

from __future__ import annotations

from agent.recipe.matcher import marker_region
from agent.runtime.site_context import normalize_page_role
from agent.runtime.worker_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    RECORDED_REPLAY_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
)
from agent.runtime.worker_contracts import WorkerState
from agent.utils.text import normalize_text, url_template
from agent.vision.marker_geometry import marker_bbox
from agent.vision.screen_signature import (
    compact_screen_context_signature,
    compute_target_roi_signature,
)
from agent.vision.target_snapshot import build_marker_target_snapshot, marker_by_id
from shared.schema.feedback_schema import RecordedRecipeStep


def _recorded_replay_mode(
    action_name: str,
    args: dict,
    slot_name: str,
) -> str:
    """모델이 선언한 재사용 방식이 행동 계약과 맞을 때만 보존한다."""

    mode = normalize_text(args.get("replay_mode")).casefold()
    if action_name not in REVIEWABLE_REPLAY_ACTIONS:
        return "reasoning"
    if mode == "parameterized":
        return (
            "parameterized"
            if action_name == "type_in_marker" and slot_name
            else "reasoning"
        )
    return "fixed" if mode == "fixed" else "reasoning"


def _new_recorded_step(
    state: dict,
    action_name: str,
    args: dict,
    seq: int,
    slot_name: str,
    context_signature: dict,
) -> dict:
    url = state.get("current_url", "") or ""
    observed_page_role = normalize_page_role(state.get("current_page_role"))
    declared_page_role = normalize_page_role(args.get("page_role"))
    page_role = observed_page_role or declared_page_role
    return {
        "seq": seq,
        "url_template": url_template(url),
        "page_role": page_role,
        "before_state": {
            "observation_id": str(state.get("observation_id") or ""),
            "url_template": url_template(url),
            "page_role": page_role,
            "screen_context_signature": context_signature,
        },
        "action": action_name,
        "target": None,
        "value": None,
        "param": {},
        "is_param": False,
        "expected_after": normalize_text(args.get("expected_after")),
        "intent": normalize_text(args.get("reason")),
        "target_role": normalize_text(args.get("target_role")),
        "component": normalize_text(args.get("target_component")),
        "slot_refs": [slot_name] if slot_name else [],
        "risk_level": normalize_text(args.get("risk_level")),
        "replay_mode": _recorded_replay_mode(action_name, args, slot_name),
    }


def _target_descriptor(
    marker: dict,
    markers: list[dict],
    args: dict,
    screen_signature: dict,
) -> dict:
    snapshot = (
        build_marker_target_snapshot(
            markers,
            args.get("marker_id"),
            screen_signature=screen_signature,
        )
        or {}
    )
    target = {
        "text": normalize_text(marker.get("text")),
        "region": marker_region(marker, markers),
    }
    marker_type = normalize_text(marker.get("type"))
    if marker_type:
        target["marker_type"] = marker_type
    for key in ("bbox_ratio", "center_ratio"):
        if snapshot.get(key):
            target[key] = snapshot[key]
    target_label = normalize_text(args.get("target_label"))
    if target_label:
        target["semantic_label"] = target_label
    return target


def _target_roi_signature(
    state: dict,
    marker: dict,
    screen_signature: dict,
) -> dict:
    screen_size = screen_signature.get("size") or []
    image_path = str(state.get("current_screenshot") or "")
    if not image_path or not isinstance(screen_size, list) or len(screen_size) != 2:
        return {}
    return compute_target_roi_signature(
        image_path,
        marker_bbox(marker),
        screen_size,
        capture_context=dict(screen_signature.get("capture_context", {}) or {}),
    )


def _record_target_parameters(
    step: dict,
    action_name: str,
    args: dict,
    slot_name: str,
) -> None:
    if action_name == "type_in_marker":
        value = (args.get("text") or "").strip()
        step["value"] = value
        step["param"] = {"text": value}
        if slot_name:
            step["param"]["slot_name"] = slot_name
        step["is_param"] = bool(slot_name)
    elif action_name == "scroll":
        step["value"] = args.get("direction", "down")
        step["param"] = {
            "direction": step["value"],
            "amount": args.get("amount", "page"),
            "targeted": True,
        }


def _record_target_action(
    step: dict,
    state: dict,
    action_name: str,
    args: dict,
    slot_name: str,
    screen_signature: dict,
) -> bool:
    markers = state.get("current_markers", []) or []
    marker = marker_by_id(markers, args.get("marker_id"))
    if not marker:
        return False
    step["target"] = _target_descriptor(
        marker,
        markers,
        args,
        screen_signature,
    )
    roi_signature = _target_roi_signature(state, marker, screen_signature)
    if roi_signature:
        step["roi_signature"] = roi_signature
    _record_target_parameters(step, action_name, args, slot_name)
    return True


def _record_contextual_parameters(
    step: dict,
    action_name: str,
    args: dict,
) -> None:
    keys = {
        "scroll": ("direction", "down"),
        "press_key": ("key", None),
        "switch_tab": ("direction", None),
    }
    key_and_default = keys.get(action_name)
    if key_and_default is None:
        return
    key, default = key_and_default
    value = args.get(key, default)
    step["value"] = value
    step["param"] = {key: value}
    if action_name == "scroll":
        step["param"]["amount"] = args.get("amount", "page")


def record_ui_step(
    recorded_steps: list[RecordedRecipeStep],
    state: WorkerState,
    action_name: str,
    args: dict,
    seq: int,
) -> None:
    """UI 액션 디스패치 직후 재생 후보 단계를 기록한다."""

    if action_name not in RECORDED_REPLAY_ACTIONS:
        return
    observation = state["observation"]
    slot_name = normalize_text(args.get("slot_name"))
    screen_signature = dict(observation.get("screen_signature", {}) or {})
    context_signature = compact_screen_context_signature(screen_signature)
    step = _new_recorded_step(
        observation,
        action_name,
        args,
        seq,
        slot_name,
        context_signature,
    )
    targeted_action = action_name in TARGET_REPLAY_ACTIONS or (
        action_name == "scroll" and args.get("marker_id") is not None
    )
    if targeted_action:
        if not _record_target_action(
            step,
            observation,
            action_name,
            args,
            slot_name,
            screen_signature,
        ):
            return
    else:
        _record_contextual_parameters(step, action_name, args)
    if action_name in CONTEXTUAL_REPLAY_ACTIONS and context_signature:
        step["screen_context_signature"] = context_signature
    recorded_steps.append(RecordedRecipeStep.model_validate(step))

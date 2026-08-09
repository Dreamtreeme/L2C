"""
비전 실행의 UI 행동과 타깃 ROI를 기록한다.
"""

from __future__ import annotations

from agent.recipe.matcher import marker_ordinal, marker_region
from agent.runtime.site_context import normalize_page_role
from agent.runtime.worker_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    RECORDED_REPLAY_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
)
from agent.utils.text import normalize_text, url_template
from agent.vision.marker_geometry import marker_bbox, marker_center
from agent.vision.screen_signature import (
    compact_screen_context_signature,
    compute_target_roi_signature,
)
from agent.vision.target_snapshot import build_marker_target_snapshot, marker_by_id


def _has_letter(text: str) -> bool:
    return any(ch.isalpha() for ch in text or "")


def _text_counts(markers: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        text = normalize_text(marker.get("text"))
        key = text.lower().replace(" ", "")
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


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




def _collect_evidence_candidates(
    target_marker: dict,
    markers: list[dict],
    counts: dict[str, int],
    target_text: str,
    unique_only: bool,
    max_dx: int,
    max_dy: int,
) -> list[tuple[int, int, int, int, str]]:
    seen = {target_text} if target_text else set()
    tx, ty = marker_center(target_marker)
    scored = []
    for marker in markers or []:
        if not isinstance(marker, dict) or marker.get("id") == target_marker.get("id"):
            continue
        text = normalize_text(marker.get("text"))
        key = text.lower().replace(" ", "")
        if not text or text in seen or len(key) < 2 or not _has_letter(text):
            continue
        if unique_only and counts.get(key, 0) != 1:
            continue
        x, y = marker_center(marker)
        dx = abs(x - tx)
        dy = abs(y - ty)
        if dx > max_dx or dy > max_dy:
            continue
        seen.add(text)
        length_bonus = min(len(key), 40)
        rank = dy * 2 + dx - length_bonus * 4
        scored.append((rank, dy, dx, marker.get("id", 0), text))
    return sorted(scored)


def _evidence_texts_for_marker(
    target_marker: dict,
    markers: list[dict],
    max_items: int = 6,
    max_dx: int = 850,
    max_dy: int = 320,
) -> list[str]:
    target_text = normalize_text(target_marker.get("text"))
    counts = _text_counts(markers)
    scored = _collect_evidence_candidates(
        target_marker,
        markers,
        counts,
        target_text,
        unique_only=True,
        max_dx=max_dx,
        max_dy=max_dy,
    )
    if not scored:
        scored = _collect_evidence_candidates(
            target_marker,
            markers,
            counts,
            target_text,
            unique_only=False,
            max_dx=max_dx,
            max_dy=max_dy,
        )
    return [item[-1] for item in scored[:max_items]]


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
        "decision_capture_id": str(state.get("current_capture_id") or ""),
        "url_template": url_template(url),
        "page_role": page_role,
        "observed_page_role": observed_page_role,
        "declared_page_role": declared_page_role,
        "before_state": {
            "capture_id": str(state.get("current_capture_id") or ""),
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
        "needs_user_confirmation": bool(args.get("needs_user_confirmation")),
        "replay_mode": _recorded_replay_mode(action_name, args, slot_name),
    }


def _target_descriptor(
    marker: dict,
    markers: list[dict],
    args: dict,
    screen_signature: dict,
) -> dict:
    snapshot = build_marker_target_snapshot(
        markers,
        args.get("marker_id"),
        screen_signature=screen_signature,
    ) or {}
    target = {
        "text": normalize_text(marker.get("text")),
        "region": marker_region(marker, markers),
        "ordinal": marker_ordinal(marker, markers),
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
    evidence_texts = _evidence_texts_for_marker(marker, markers)
    if evidence_texts:
        target["evidence_texts"] = evidence_texts
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


def record_ui_step(recorded_steps, state, action_name, args, seq) -> None:
    """UI 액션 디스패치 직후 재생 후보 단계를 기록한다."""

    if action_name not in RECORDED_REPLAY_ACTIONS:
        return
    slot_name = normalize_text(args.get("slot_name"))
    screen_signature = dict(state.get("screen_signature", {}) or {})
    context_signature = compact_screen_context_signature(
        screen_signature
    )
    step = _new_recorded_step(
        state,
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
            state,
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
    recorded_steps.append(step)

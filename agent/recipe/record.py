"""
Phase 0: 비전 런의 UI 행동과 타깃 ROI 기록.
execution_node의 기록 단계에서 호출되며, 실패해도 실제 실행 흐름에는 영향을 주지 않는다.
"""

from __future__ import annotations

from agent.recipe.matcher import marker_ordinal, marker_region
from agent.runtime.site_context import normalize_page_role
from agent.runtime.replay_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    RECORDED_REPLAY_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
)
from agent.utils.text import normalize_text, url_template
from agent.utils.logger import logger
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


def record_ui_step(recorded_steps, state, action_name, args, seq) -> None:
    """UI 액션 디스패치 직후 호출. recorded_steps에 in-place append (예외 안전)."""
    try:
        if action_name not in RECORDED_REPLAY_ACTIONS:
            return
        markers = state.get("current_markers", []) or []
        url = state.get("current_url", "") or ""
        slot_name = normalize_text(args.get("slot_name"))
        screen_signature = dict(state.get("screen_signature", {}) or {})
        observed_page_role = normalize_page_role(state.get("current_page_role"))
        declared_page_role = normalize_page_role(args.get("page_role"))
        page_role = observed_page_role or declared_page_role
        context_signature = compact_screen_context_signature(
            screen_signature
        )
        step = {
            "seq": seq,
            "decision_capture_id": str(state.get("current_capture_id") or ""),
            "url_template": url_template(url),
            "page_role": page_role,
            "observed_page_role": observed_page_role,
            "declared_page_role": declared_page_role,
            "before_state": {
                "capture_id": str(
                    state.get("current_capture_id") or ""
                ),
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
            "target_role": normalize_text(args.get("target_role") or args.get("target_role_candidate")),
            "component": normalize_text(args.get("target_component") or args.get("component_candidate")),
            "slot_refs": [slot_name] if slot_name else [],
            "risk_level": normalize_text(args.get("risk_level")),
            "needs_user_confirmation": bool(
                args.get("needs_user_confirmation")
            ),
            "replay_mode": _recorded_replay_mode(
                action_name,
                args,
                slot_name,
            ),
        }
        if action_name in TARGET_REPLAY_ACTIONS or (
            action_name == "scroll" and args.get("marker_id") is not None
        ):
            marker = marker_by_id(markers, args.get("marker_id"))
            if not marker:
                return
            target_snapshot = build_marker_target_snapshot(
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
            if target_snapshot.get("bbox_ratio"):
                target["bbox_ratio"] = target_snapshot["bbox_ratio"]
            if target_snapshot.get("center_ratio"):
                target["center_ratio"] = target_snapshot["center_ratio"]
            screen_size = screen_signature.get("size") or []
            if isinstance(screen_size, list) and len(screen_size) == 2:
                bbox = marker_bbox(marker)
                image_path = str(state.get("current_screenshot") or "")
                roi_signature = (
                    compute_target_roi_signature(
                        image_path,
                        bbox,
                        screen_size,
                        capture_context=dict(screen_signature.get("capture_context", {}) or {}),
                    )
                    if image_path
                    else {}
                )
                if roi_signature:
                    step["roi_signature"] = roi_signature
            target_label = normalize_text(args.get("target_label") or args.get("semantic_label"))
            if target_label:
                target["semantic_label"] = target_label
            evidence_texts = _evidence_texts_for_marker(marker, markers)
            if evidence_texts:
                target["evidence_texts"] = evidence_texts
            step["target"] = target
            if action_name == "type_in_marker":
                val = (args.get("text") or "").strip()
                step["value"] = val
                step["param"] = {"text": val}
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
        elif action_name == "scroll":
            step["value"] = args.get("direction", "down")
            step["param"] = {
                "direction": step["value"],
                "amount": args.get("amount", "page"),
            }
        elif action_name == "press_key":
            step["value"] = args.get("key")
            step["param"] = {"key": step["value"]}
        elif action_name == "switch_tab":
            step["value"] = args.get("direction")
            step["param"] = {"direction": step["value"]}
        if action_name in CONTEXTUAL_REPLAY_ACTIONS:
            if context_signature:
                step["screen_context_signature"] = context_signature
        recorded_steps.append(step)
    except Exception as e:
        logger.debug("reflex record_ui_step skipped", error=str(e))

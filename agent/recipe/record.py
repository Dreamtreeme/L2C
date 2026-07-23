"""
Phase 0: 비전 런의 UI 행동과 타깃 ROI 기록.
action_node에서 호출되며, 전부 예외 안전 -> 실패해도 실제 실행 흐름에 영향 0.
"""

from __future__ import annotations

from agent.recipe.matcher import marker_ordinal, marker_region
from agent.recipe.page_context import normalize_page_role
from agent.recipe.text_utils import normalize_text, url_template
from agent.utils.logger import logger
from agent.vision.marker_geometry import bbox_to_ratio, center_ratio_from_bbox, marker_bbox, marker_center
from agent.vision.screen_signature import compute_target_roi_signature

_TARGET_ACTIONS = {"click_marker", "type_in_marker"}
_RECORDED_ACTIONS = _TARGET_ACTIONS | {
    "scroll",
    "press_key",
    "go_back",
    "close_current_tab",
    "switch_tab",
}


def _squash(s) -> str:
    return normalize_text(s).lower().replace(" ", "")


def _marker(markers, marker_id):
    for m in markers or []:
        if isinstance(m, dict) and m.get("id") == marker_id:
            return m
    return None


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
        if action_name not in _RECORDED_ACTIONS:
            return
        markers = state.get("current_markers", []) or []
        url = state.get("current_url", "") or ""
        goal = state.get("goal", "") or ""
        slot_name = normalize_text(args.get("slot_name"))
        screen_signature = dict(state.get("screen_signature", {}) or {})
        observed_page_role = normalize_page_role(state.get("current_page_role"))
        declared_page_role = normalize_page_role(args.get("page_role"))
        step = {
            "seq": seq,
            "decision_capture_id": str(state.get("current_capture_id") or ""),
            "url_template": url_template(url),
            "page_role": observed_page_role or declared_page_role,
            "observed_page_role": observed_page_role,
            "declared_page_role": declared_page_role,
            "action": action_name,
            "target": None,
            "value": None,
            "param": {},
            "is_param": False,
            "expected_after": normalize_text(args.get("expected_after")),
            "transition_contract": None,
            "intent": normalize_text(args.get("reason")),
            "target_role": normalize_text(args.get("target_role") or args.get("target_role_candidate")),
            "component": normalize_text(args.get("target_component") or args.get("component_candidate")),
            "slot_refs": [slot_name] if slot_name else [],
            "fixed": action_name in {
                "scroll",
                "press_key",
                "go_back",
                "close_current_tab",
                "switch_tab",
            },
        }
        if action_name in _TARGET_ACTIONS or (
            action_name == "scroll" and args.get("marker_id") is not None
        ):
            marker = _marker(markers, args.get("marker_id"))
            if not marker:
                return
            target = {
                "text": normalize_text(marker.get("text")),
                "region": marker_region(marker, markers),
                "ordinal": marker_ordinal(marker, markers),
            }
            marker_type = normalize_text(marker.get("type"))
            if marker_type:
                target["marker_type"] = marker_type
            screen_size = screen_signature.get("size") or []
            if isinstance(screen_size, list) and len(screen_size) == 2:
                bbox = marker_bbox(marker)
                target["bbox_ratio"] = bbox_to_ratio(bbox, screen_size)
                target["center_ratio"] = center_ratio_from_bbox(bbox, screen_size)
                recent_images = state.get("recent_images", []) or []
                image_path = recent_images[-1] if recent_images else ""
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
                step["is_param"] = bool(slot_name) or (bool(val) and _squash(val) in _squash(goal))
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
        recorded_steps.append(step)
    except Exception as e:
        logger.debug("reflex record_ui_step skipped", error=str(e))


def commit_if_finished(recorded_steps, state, current_url) -> None:
    """Defer active Reflex promotion to commander-reviewed worker submissions."""
    try:
        steps = list(recorded_steps or [])
        if not steps:
            return
        logger.info(
            "reflex recipe promotion deferred to worker submission review",
            steps=len(steps),
            current_url=current_url,
        )
    except Exception as e:
        logger.debug("reflex commit_if_finished skipped", error=str(e))

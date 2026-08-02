"""행동 전후 화면 관찰을 전환 상태로 바꾸는 순수 판정 규칙."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.graph.state import GraphState
from agent.recipe.page_context import normalize_page_role


@dataclass(frozen=True)
class TransitionDecision:
    status: str
    reason: str
    needs_ocr: bool = False
    block_reflex_recipe: bool = False


def decide_transition_probe(
    *,
    elapsed_sec: float,
    timeout_sec: float,
) -> TransitionDecision:
    """저비용 화면 대조가 아직 목표 화면에 도달하지 못한 경우를 판정한다."""

    if elapsed_sec < timeout_sec:
        return TransitionDecision(
            status="pending",
            reason="target_screen_not_ready",
        )
    return TransitionDecision(
        status="unknown",
        reason="transition_timeout",
        needs_ocr=True,
    )


def decide_before_ocr(
    *,
    source: str,
    visual_changed: bool,
) -> TransitionDecision:
    """OCR 전 픽셀 변화만으로 다음 관찰 단계를 정한다."""

    if visual_changed:
        return TransitionDecision(
            status="needs_ocr",
            reason="ocr_required",
            needs_ocr=True,
        )
    is_reflex = source == "reflex"
    return TransitionDecision(
        status="unknown",
        reason=(
            "reflex_no_screen_change"
            if is_reflex
            else "no_screen_change"
        ),
        block_reflex_recipe=is_reflex,
    )


def decide_after_ocr(
    *,
    source: str,
    markers_present: bool,
    url_changed: bool,
    visual_changed: bool,
    reflex_matched: bool | None = None,
    reflex_reason: str = "",
) -> TransitionDecision:
    """OCR가 연결된 현재 캡처로 일반 행동 또는 경험 경로 도착을 판정한다."""

    if source == "reflex":
        matched = bool(reflex_matched)
        return TransitionDecision(
            status="ready" if matched else "unknown",
            reason=reflex_reason,
            block_reflex_recipe=not matched,
        )
    if markers_present and (url_changed or visual_changed):
        return TransitionDecision(
            status="ready",
            reason=(
                "screen_change_pixels_matched"
                if visual_changed
                else "screen_change_url_matched"
            ),
        )
    if not url_changed and not visual_changed:
        return TransitionDecision(
            status="unknown",
            reason="no_screen_change",
        )
    return TransitionDecision(
        status="unknown",
        reason="transition_change_unverified",
    )


def verify_reflex_after_state(
    request: dict[str, Any],
    state: GraphState,
) -> tuple[bool, str, dict[str, Any]]:
    """저장된 도착 화면과 현재 OCR 화면이 같은 상태인지 확인한다."""

    if request.get("execution_failed"):
        return False, "recipe_action_group_failed", {}
    expected = (
        dict(request.get("expected_after_state") or {})
        if isinstance(request.get("expected_after_state"), dict)
        else {}
    )
    if not expected:
        return False, "recipe_after_state_missing", {}

    from agent.recipe.phash_replay import (
        match_step_by_screen_signature,
        screen_context_signature_match,
    )
    from agent.recipe.text_utils import recipe_url_scope_matches

    expected_url = str(expected.get("url_template") or "")
    current_url = str(state.get("current_url") or "")
    if (
        expected_url
        and current_url
        and not recipe_url_scope_matches(expected_url, current_url)
    ):
        return False, "recipe_after_url_mismatch", {
            "expected_url_template": expected_url,
            "current_url": current_url,
        }

    before_role = normalize_page_role(request.get("before_page_role"))
    expected_role = normalize_page_role(expected.get("page_role"))
    current_role = normalize_page_role(state.get("current_page_role"))
    if (
        before_role
        and expected_role
        and expected_role != before_role
        and current_role
    ):
        matched = current_role == expected_role
        return matched, (
            "recipe_after_page_role_matched"
            if matched
            else "recipe_after_page_role_mismatch"
        ), {
            "before_page_role": before_role,
            "expected_page_role": expected_role,
            "current_page_role": current_role,
        }

    anchor_target = expected.get("anchor_target")
    anchor_signature = dict(
        expected.get("anchor_roi_signature") or {}
    )
    if isinstance(anchor_target, dict) and anchor_signature:
        marker_id, match = match_step_by_screen_signature(
            {
                "target": anchor_target,
                "roi_signature": anchor_signature,
            },
            dict(state.get("screen_signature") or {}),
            list(state.get("current_markers") or []),
            current_image_path=str(
                state.get("current_screenshot") or ""
            ),
        )
        if marker_id is None:
            return False, str(
                match.get("reason") or "recipe_after_anchor_mismatch"
            ), match
        return True, "recipe_after_anchor_matched", {
            **match,
            "marker_id": marker_id,
        }

    context_signature = dict(
        expected.get("screen_context_signature") or {}
    )
    if context_signature:
        match = screen_context_signature_match(
            context_signature,
            dict(state.get("screen_signature") or {}),
        )
        return (
            bool(match.get("matched")),
            (
                "recipe_after_context_matched"
                if match.get("matched")
                else str(
                    match.get("reason")
                    or "recipe_after_context_mismatch"
                )
            ),
            match,
        )

    before_url = str(
        request.get("before_url_template")
        or request.get("before_url")
        or ""
    )
    if (
        expected_url
        and current_url
        and expected_url != before_url
        and recipe_url_scope_matches(expected_url, current_url)
    ):
        return True, "recipe_after_url_matched", {
            "expected_url_template": expected_url,
            "current_url": current_url,
        }
    return False, "recipe_after_state_unverifiable", {}


__all__ = [
    "TransitionDecision",
    "decide_after_ocr",
    "decide_before_ocr",
    "decide_transition_probe",
    "verify_reflex_after_state",
]

"""결정론적 행동 선택 노드가 사용하는 순수 판정 규칙."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SelectionPolicy(str, Enum):
    CONTINUE = "continue"
    KEEP_PENDING_ACTION = "keep_pending_action"
    WAIT_LOW_INFORMATION = "wait_low_information"
    STOP_LOW_INFORMATION = "stop_low_information"
    DEFER_TO_ACTIVE_REFLEX = "defer_to_active_reflex"
    SKIP_DUPLICATE_DETAIL = "skip_duplicate_detail"
    WAIT_FOR_RESULTS_SCREEN = "wait_for_results_screen"
    REPLAY_JOB_CARD = "replay_job_card"


@dataclass(frozen=True)
class SelectionDecision:
    policy: SelectionPolicy
    reason: str


def decide_selection_entry(
    *,
    has_pending_action: bool,
    low_information_screen: bool,
    low_information_capture_count: int,
    low_information_max_cycles: int,
    has_active_reflex_recipe: bool,
) -> SelectionDecision:
    """선택 노드 진입 시 다른 정책보다 먼저 적용할 경계를 정한다."""

    if has_pending_action:
        return SelectionDecision(
            SelectionPolicy.KEEP_PENDING_ACTION,
            "pending_action_exists",
        )
    if low_information_screen:
        if low_information_capture_count >= low_information_max_cycles:
            return SelectionDecision(
                SelectionPolicy.STOP_LOW_INFORMATION,
                "low_information_limit_reached",
            )
        return SelectionDecision(
            SelectionPolicy.WAIT_LOW_INFORMATION,
            "low_information_screen",
        )
    if has_active_reflex_recipe:
        return SelectionDecision(
            SelectionPolicy.DEFER_TO_ACTIVE_REFLEX,
            "active_reflex_recipe",
        )
    return SelectionDecision(SelectionPolicy.CONTINUE, "selection_available")


def decide_duplicate_detail(
    *,
    has_active_card: bool,
    is_job_detail_url: bool,
    duplicate_matched: bool,
) -> SelectionDecision:
    if has_active_card and is_job_detail_url and duplicate_matched:
        return SelectionDecision(
            SelectionPolicy.SKIP_DUPLICATE_DETAIL,
            "existing_job_detail",
        )
    return SelectionDecision(SelectionPolicy.CONTINUE, "not_duplicate_detail")


def decide_queue_return(
    *,
    replay_available: bool,
    is_return_action: bool,
    ocr_complete: bool,
    replay_reason: str,
    transition_needs_ocr: bool,
    target_phash_available: bool,
) -> SelectionDecision:
    """목록 복귀 뒤 큐 재생, 저비용 대기 또는 일반 폴백을 고른다."""

    if replay_available:
        return SelectionDecision(
            SelectionPolicy.REPLAY_JOB_CARD,
            "queue_card_replay_available",
        )
    should_wait = (
        is_return_action
        and not ocr_complete
        and replay_reason == "phash_mismatch"
        and not transition_needs_ocr
        and target_phash_available
    )
    if should_wait:
        return SelectionDecision(
            SelectionPolicy.WAIT_FOR_RESULTS_SCREEN,
            "queue_return_phash_wait",
        )
    return SelectionDecision(
        SelectionPolicy.CONTINUE,
        replay_reason or "no_queue_replay",
    )


__all__ = [
    "SelectionDecision",
    "SelectionPolicy",
    "decide_duplicate_detail",
    "decide_queue_return",
    "decide_selection_entry",
]

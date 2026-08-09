"""물리·상태 행동 실행 전후에 적용하는 순수 정책 헬퍼."""

from __future__ import annotations

from typing import Any

from agent.runtime.action_permissions import task_permission_reason
from agent.runtime.worker_contracts import WorkerState
from agent.runtime.job_card_queue import (
    job_card_label,
    job_card_entries_from_args,
)


def sensitive_action_reason(
    state: WorkerState,
    action_name: str,
    args: dict[str, Any],
    *,
    source: str = "",
) -> str:
    permission_reason = task_permission_reason(
        state,
        action_name,
        args,
        source=source,
    )
    if permission_reason:
        return permission_reason
    if action_name in {
        "close_current_tab",
        "switch_tab",
        "go_back",
        "scroll",
    }:
        return ""
    if args.get("needs_user_confirmation") is True:
        return "tool_args_requested_user_confirmation"
    if str(args.get("risk_level") or "").strip().lower() == "sensitive":
        return "tool_args_marked_sensitive"
    return ""


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _compact_observed_fields(value: Any) -> list[str]:
    """원본 사전과 이미 압축된 필드 목록을 같은 형태로 정규화한다."""

    if isinstance(value, (dict, list, tuple, set)):
        fields = value
    else:
        return []
    return sorted({str(field).strip() for field in fields if str(field).strip()})


def compact_action_args(action_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if action_name == "finish_detail_reading":
        return {
            "page_role": args.get("page_role", "job_detail"),
            "observed_fields": _compact_observed_fields(args.get("observed_fields")),
            "unavailable_fields": list(args.get("unavailable_fields") or []),
            "page_exhausted": bool(args.get("page_exhausted")),
            "reason": _clip_text(args.get("reason", ""), 120),
        }
    if action_name == "set_job_card_queue":
        cards = job_card_entries_from_args(args)
        titles = [job_card_label(card) for card in cards]
        return {
            "cards": len(cards),
            "titles": [title for title in titles if title][:5],
        }
    compact = {
        key: value for key, value in args.items() if not str(key).startswith("_")
    }
    if isinstance(
        compact.get("observed_fields"),
        (dict, list, tuple, set),
    ):
        compact["observed_fields"] = _compact_observed_fields(
            compact["observed_fields"]
        )
    return compact


def state_snapshot_for_action(state: WorkerState, current_url: str) -> dict[str, Any]:
    observation = state["observation"]
    return {
        "observation_id": str(observation.get("observation_id") or ""),
        "url": current_url or observation.get("current_url", "") or "",
        "screenshot": str(observation.get("current_screenshot") or ""),
        "marked_image": observation.get("marked_image", "") or "",
        "screen_signature": dict(observation.get("screen_signature", {}) or {}),
    }


def repeats_no_effect_target(
    observation: dict[str, Any],
    action_name: str,
    args: dict[str, Any],
) -> bool:
    """같은 화면에서 효과가 없었던 동일 원자 대상만 재실행인지 판정한다."""

    if observation.get("action") != action_name:
        return False
    step = observation.get("step") if isinstance(observation.get("step"), dict) else {}
    previous_args = step.get("args") if isinstance(step.get("args"), dict) else {}
    if action_name in {"click_marker", "type_in_marker"}:
        previous_marker = previous_args.get("marker_id")
        current_marker = args.get("marker_id")
        return previous_marker is not None and previous_marker == current_marker
    if action_name == "press_key":
        return str(previous_args.get("key") or "") == str(args.get("key") or "")
    if action_name == "switch_tab":
        return str(previous_args.get("direction") or "") == str(
            args.get("direction") or ""
        )
    if action_name == "open_browser":
        previous_target = previous_args.get("url")
        current_target = args.get("url")
        return bool(previous_target and previous_target == current_target)
    return action_name in {"go_back", "close_current_tab"}


__all__ = [
    "compact_action_args",
    "repeats_no_effect_target",
    "sensitive_action_reason",
    "state_snapshot_for_action",
]

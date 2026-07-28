"""물리·상태 행동 실행 전후에 적용하는 순수 정책 헬퍼."""

from __future__ import annotations

import json
from typing import Any

from agent.config import get_settings
from agent.graph.state import GraphState
from agent.runtime.job_collection import JOB_LIST_KEYS, job_list_value
from agent.runtime.job_card_queue import (
    job_card_label,
    job_card_entries_from_args,
)
from agent.runtime.site_context import (
    looks_like_job_detail_url,
    persistence_policy_for_url,
)


def _has_job_url(job: dict[str, Any]) -> bool:
    return bool(
        (job.get("url") or job.get("URL") or job.get("공고url") or "").strip()
    )


def should_skip_job_update_without_detail_url(
    new_data: dict[str, Any],
    current_url: str,
) -> bool:
    policy = persistence_policy_for_url(current_url)
    if not policy.get("require_detail_url_for_job_update") or looks_like_job_detail_url(
        current_url
    ):
        return False

    incoming_jobs = job_list_value(new_data)
    if isinstance(incoming_jobs, dict):
        incoming_jobs = [incoming_jobs]
    if not isinstance(incoming_jobs, list):
        return False
    return any(
        isinstance(job, dict) and not _has_job_url(job) for job in incoming_jobs
    )


def _job_identity(job: dict[str, Any]) -> tuple[str, str, str]:
    url = (job.get("url") or job.get("URL") or job.get("공고url") or "").strip()
    company = (job.get("회사명") or job.get("company_name") or "").strip()
    position = (job.get("직무명") or job.get("position") or "").strip()
    return url, company, position


def _merge_value(old: Any, new: Any) -> Any:
    if new in (None, "", [], {}):
        return old
    if isinstance(old, list) or isinstance(new, list):
        old_items = old if isinstance(old, list) else ([old] if old not in (None, "") else [])
        new_items = new if isinstance(new, list) else [new]
        merged = []
        seen = set()
        for item in old_items + new_items:
            key = (
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                if isinstance(item, (dict, list))
                else str(item)
            )
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
        return merged
    if isinstance(old, dict) and isinstance(new, dict):
        merged = dict(old)
        for key, value in new.items():
            merged[key] = _merge_value(merged.get(key), value)
        return merged
    return new


def merge_extracted_info(
    current_jd: dict[str, Any],
    new_data: dict[str, Any],
    current_url: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    merged = dict(current_jd)
    summary: dict[str, Any] = {"incoming_jobs": 0, "total_jobs": 0, "fields": []}
    incoming_jobs = job_list_value(new_data)
    if isinstance(incoming_jobs, dict):
        incoming_jobs = [incoming_jobs]

    if isinstance(incoming_jobs, list):
        existing_jobs = job_list_value(merged)
        if not isinstance(existing_jobs, list):
            existing_jobs = []
        for incoming in incoming_jobs:
            if not isinstance(incoming, dict):
                continue
            job = dict(incoming)
            if looks_like_job_detail_url(current_url) and not (
                job.get("url") or job.get("URL") or job.get("공고url")
            ):
                job["url"] = current_url

            summary["incoming_jobs"] += 1
            summary["fields"].extend(job.keys())
            identity = _job_identity(job)
            match_index = None
            for index, existing in enumerate(existing_jobs):
                if not isinstance(existing, dict):
                    continue
                existing_identity = _job_identity(existing)
                if existing_identity == identity and any(identity):
                    match_index = index
                    break
                if identity[0] and identity[0] in existing_identity:
                    match_index = index
                    break
                if identity[1:] == existing_identity[1:] and all(identity[1:]):
                    match_index = index
                    break

            if match_index is None:
                existing_jobs.append(job)
            else:
                existing_jobs[match_index] = _merge_value(existing_jobs[match_index], job)

        merged["공고목록"] = existing_jobs
        summary["total_jobs"] = len(existing_jobs)

    for key, value in new_data.items():
        if key in JOB_LIST_KEYS:
            continue
        summary["fields"].append(key)
        merged[key] = _merge_value(merged.get(key), value)

    summary["fields"] = sorted({str(field) for field in summary["fields"]})
    existing_jobs = job_list_value(merged)
    if not summary["total_jobs"] and isinstance(existing_jobs, list):
        summary["total_jobs"] = len(existing_jobs)
    return merged, summary


def auto_finish_on_target_enabled() -> bool:
    return get_settings().vision.auto_finish_on_target


def sensitive_action_reason(
    _state: GraphState,
    action_name: str,
    args: dict[str, Any],
) -> str:
    if action_name in {
        "close_browser",
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

    if isinstance(value, dict):
        fields = value
    elif isinstance(value, (list, tuple, set)):
        fields = value
    else:
        return []
    return sorted(
        {
            str(field).strip()
            for field in fields
            if str(field).strip()
        }
    )


def compact_action_args(action_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if action_name == "finish_detail_reading":
        return {
            "page_role": args.get("page_role", "job_detail"),
            "observed_fields": _compact_observed_fields(
                args.get("observed_fields")
            ),
            "unavailable_fields": list(
                args.get("unavailable_fields") or []
            ),
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
    if action_name != "update_extracted_info":
        compact = {
            key: value
            for key, value in args.items()
            if not str(key).startswith("_")
        }
        if isinstance(
            compact.get("observed_fields"),
            (dict, list, tuple, set),
        ):
            compact["observed_fields"] = _compact_observed_fields(
                compact["observed_fields"]
            )
        return compact
    try:
        data = json.loads(args.get("data_json", "{}"))
    except Exception:
        return {"data_json": "<invalid json>"}
    jobs = job_list_value(data)
    if isinstance(jobs, dict):
        jobs = [jobs]
    fields = []
    if isinstance(jobs, list):
        for job in jobs:
            if isinstance(job, dict):
                fields.extend(job.keys())
    fields.extend(key for key in data.keys() if key not in JOB_LIST_KEYS)
    return {
        "incoming_jobs": len(jobs) if isinstance(jobs, list) else 0,
        "fields": sorted({str(field) for field in fields}),
        "payload_chars": len(args.get("data_json", "")),
    }


def state_snapshot_for_action(state: GraphState, current_url: str) -> dict[str, Any]:
    recent_images = state.get("recent_images", []) or []
    return {
        "capture_id": str(state.get("current_capture_id") or ""),
        "url": current_url or state.get("current_url", "") or "",
        "screenshot": str(recent_images[-1]) if recent_images else "",
        "marked_image": state.get("marked_image", "") or "",
        "screen_signature": dict(state.get("screen_signature", {}) or {}),
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
        return str(previous_args.get("direction") or "") == str(args.get("direction") or "")
    if action_name == "open_browser":
        previous_target = previous_args.get("url") or previous_args.get("site")
        current_target = args.get("url") or args.get("site")
        return bool(previous_target and previous_target == current_target)
    return action_name in {"go_back", "close_current_tab", "close_browser"}


__all__ = [
    "auto_finish_on_target_enabled",
    "compact_action_args",
    "merge_extracted_info",
    "repeats_no_effect_target",
    "sensitive_action_reason",
    "should_skip_job_update_without_detail_url",
    "state_snapshot_for_action",
]

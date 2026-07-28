"""검증된 행동 요청의 상태 도구와 물리 입력 실행 노드."""

import json
import time
from typing import Any, Dict, Tuple

from agent.application.detail_extraction_service import (
    extract_job_from_detail_ocr_buffer as _extract_job_from_detail_ocr_buffer,
)
from agent.graph.action_request import (
    ActionRequest,
    ActionResult,
    build_action_request,
)
from agent.graph.state import GraphState
from agent.runtime.action_validation import (
    text_input_target_rejection as _text_input_target_rejection,
)
from agent.runtime.duplicate_job_policy import (
    mark_existing_result_cards as _mark_existing_result_cards,
)
from agent.runtime.job_collection import job_list_value as _job_list_value
from agent.runtime.job_content import (
    has_meaningful_job_content as _has_meaningful_job_content,
    job_content_presence as _job_content_presence,
)
from agent.runtime.job_identity import source_card_key as _source_card_key
from agent.runtime.result_card_queue import (
    completed_result_card_count as _completed_result_card_count,
    complete_active_result_card as _complete_active_result_card,
    mark_result_card_active as _mark_result_card_active,
    marker_by_id as _marker_by_id,
    normalize_result_card_queue as _normalize_result_card_queue,
    pending_result_cards as _pending_result_cards,
    normalized_return_action as _normalized_return_action,
    result_card_queue_scope_complete as _result_card_queue_scope_complete,
    result_card_click_matches_queue as _result_card_click_matches_queue,
)
from agent.runtime.transition_runtime import latest_no_effect_transition as _latest_no_effect_transition
from agent.utils.logger import logger

from agent.graph.worker_execution_policy import (
    action_target_metadata as _action_target_metadata,
    auto_finish_on_target_enabled as _auto_finish_on_target_enabled,
    chain_boundary_reached as _chain_boundary_reached,
    compact_action_args as _compact_action_args,
    is_allowed_same_screen_ui_chain as _is_allowed_same_screen_ui_chain,
    is_detail_update as _is_detail_update,
    merge_extracted_info as _merge_extracted_info,
    repeats_no_effect_target as _repeats_no_effect_target,
    sensitive_action_reason as _sensitive_action_reason,
    should_skip_job_update_without_detail_url as _should_skip_job_update_without_detail_url,
    state_snapshot_for_action as _state_snapshot_for_action,
)
from agent.graph.worker_resources import (
    check_current_reasoning_screen as _check_current_reasoning_screen,
    get_action_tools as _get_action_tools,
)
from agent.graph.worker_state import (
    count_mode_from_state as _count_mode_from_state,
    detail_return_pending_for_url as _detail_return_pending_for_url,
    extracted_job_count as _extracted_job_count,
    target_count_from_state as _target_count_from_state,
)


def _dispatch_ui(action_name: str, args: dict, get_bbox, current_url: str = "") -> dict:
    """마우스/키보드 물리 조작 도구를 실행합니다."""
    action_tools = _get_action_tools()
    if action_name == "click_marker":
        bbox = get_bbox(args["marker_id"])
        return action_tools.click_marker(bbox)
    elif action_name == "type_in_marker":
        return action_tools.type_in_marker(get_bbox(args["marker_id"]), args["text"])
    elif action_name == "scroll":
        marker_id = args.get("marker_id")
        bbox = get_bbox(marker_id) if marker_id is not None else None
        return action_tools.scroll(
            direction=args.get("direction", "down"),
            bbox=bbox,
            amount=args.get("amount", "page"),
        )
    elif action_name == "press_key":
        return action_tools.press_key(args["key"])
    elif action_name == "open_browser":
        return action_tools.open_browser(args["url"], current_url=current_url)
    elif action_name == "close_browser":
        return action_tools.close_browser()
    elif action_name == "close_current_tab":
        return action_tools.close_current_tab()
    elif action_name == "switch_tab":
        return action_tools.switch_tab(args["direction"])
    elif action_name == "go_back":
        return action_tools.go_back()
    raise ValueError(f"Unknown UI action: {action_name}")


def _dispatch_state(
    action_name: str, args: dict,
    current_jd: dict,
    current_url: str = "",
    state: GraphState | None = None,
) -> Tuple[dict, dict]:
    """그래프 상태 변경 도구를 실행하고 결과와 수집 데이터를 반환합니다."""

    def attach_active_card_identity(data: dict[str, Any]) -> dict[str, Any]:
        active_card = dict((state or {}).get("active_result_card", {}) or {})
        company = str(active_card.get("company") or "").strip()
        title = str(active_card.get("title") or "").strip()
        card_key = _source_card_key(current_url, company, title)
        if not card_key:
            return data
        jobs = _job_list_value(data)
        if isinstance(jobs, dict):
            job_items = [jobs]
        elif isinstance(jobs, list):
            job_items = [item for item in jobs if isinstance(item, dict)]
        else:
            job_items = [data] if isinstance(data, dict) else []
        for job in job_items:
            job.setdefault("_source_card_key", card_key)
            job.setdefault("_source_context_url", current_url)
            job.setdefault("_listing_company", company)
            job.setdefault("_listing_title", title)
        return data

    if action_name == "update_extracted_info":
        try:
            new_data = attach_active_card_identity(json.loads(args["data_json"]))
            detail_buffer = dict((state or {}).get("detail_ocr_buffer", {}) or {})
            detail_buffer_active = bool(
                detail_buffer.get("url") == current_url
                and detail_buffer.get("lines")
            )
            if detail_buffer_active:
                result_str = (
                    "Skipped intermediate extraction: accumulated detail OCR must be finalized "
                    "with finish_detail_reading."
                )
                status = "skipped"
                reason = "detail_buffer_requires_finish"
            elif _should_skip_job_update_without_detail_url(new_data, current_url):
                result_str = (
                    "Skipped extracted data merge: this site requires a detail URL "
                    "or an explicit job url in data_json"
                )
                status = "skipped"
                reason = "job_update_requires_detail_url"
            else:
                current_jd, summary = _merge_extracted_info(current_jd, new_data, current_url=current_url)
                result_str = (
                    "Extracted data merged "
                    f"(incoming_jobs={summary['incoming_jobs']}, total_jobs={summary['total_jobs']}, "
                    f"fields={summary['fields']})"
                )
                status = "success"
                reason = ""
        except Exception as e:
            result_str = f"Failed to parse data_json: {e}"
            status = "error"
            reason = ""
        result = {"action": "update_extracted_info", "status": status, "result": result_str}
        if reason:
            result["reason"] = reason
    elif action_name == "finish_detail_reading":
        try:
            extracted_job = _extract_job_from_detail_ocr_buffer(state or {}, current_url)
            if not extracted_job:
                result = {
                    "action": "finish_detail_reading",
                    "status": "skipped",
                    "result": "No accumulated detail OCR text to extract.",
                    "reason": "empty_detail_ocr_buffer",
                    "_detail_ocr_buffer": {},
                }
            elif not _has_meaningful_job_content(extracted_job):
                presence = _job_content_presence(extracted_job)
                previous_followup = dict(
                    (state or {}).get("detail_followup_required", {}) or {}
                )
                attempts = (
                    int(previous_followup.get("attempts") or 0) + 1
                    if previous_followup.get("url") == current_url
                    else 1
                )
                result = {
                    "action": "finish_detail_reading",
                    "status": "skipped",
                    "result": (
                        "Detail reading is not complete because neither main_tasks nor "
                        "requirements were extracted. Inspect the accumulated OCR and visible "
                        "page for an original-source link or another way to reveal job content."
                    ),
                    "reason": "detail_content_incomplete",
                    "content_presence": presence,
                    "_detail_ocr_buffer": dict(
                        (state or {}).get("detail_ocr_buffer", {}) or {}
                    ),
                    "_detail_followup_required": {
                        "url": current_url,
                        "reason": "detail_content_incomplete",
                        "missing_fields": [
                            field for field, present in presence.items() if not present
                        ],
                        "attempts": attempts,
                    },
                }
            else:
                extracted_job = attach_active_card_identity(extracted_job)
                current_jd, summary = _merge_extracted_info(
                    current_jd,
                    {"공고목록": [extracted_job]},
                    current_url=current_url,
                )
                result = {
                    "action": "finish_detail_reading",
                    "status": "success",
                    "result": (
                        "Detail OCR buffer extracted and merged "
                        f"(incoming_jobs={summary['incoming_jobs']}, total_jobs={summary['total_jobs']}, "
                        f"fields={summary['fields']})"
                    ),
                    "incoming_jobs": summary["incoming_jobs"],
                    "total_jobs": summary["total_jobs"],
                    "fields": summary["fields"],
                    "_detail_ocr_buffer": {},
                    "_detail_followup_required": {},
                }
        except Exception as e:
            result = {
                "action": "finish_detail_reading",
                "status": "error",
                "result": f"Failed to extract detail OCR buffer: {e}",
            }
    elif action_name == "set_result_card_queue":
        queue, memory = _normalize_result_card_queue(args, state or {}, current_url)
        queue, existing_cards = _mark_existing_result_cards(queue, current_url)
        selector_trace = dict((state or {}).get("result_card_selector_trace", {}) or {})
        availability_source = args if args.get("available_result_count") is not None else selector_trace
        availability = {}
        try:
            available_count = int(availability_source.get("available_result_count"))
            count_confidence = float(availability_source.get("count_confidence") or 0.0)
        except (TypeError, ValueError):
            available_count = -1
            count_confidence = 0.0
        count_evidence = str(availability_source.get("count_evidence") or "").strip()[:160]
        if available_count >= len(queue) and count_confidence >= 0.8 and count_evidence:
            availability = {
                "available_result_count": available_count,
                "count_evidence": count_evidence,
                "count_confidence": count_confidence,
            }
        result = {
            "action": "set_result_card_queue",
            "status": "success" if queue else "skipped",
            "result": f"Result card queue stored: {len(queue)} card(s)." if queue else "No valid visible result cards were queued.",
            "queued_count": len(queue),
            "queued_titles": [item.get("title", "") for item in queue],
            "existing_card_count": len(existing_cards),
            "existing_cards": existing_cards,
            "_result_card_queue": queue,
            "_result_page_memory": memory,
            "_result_availability": availability,
        }
    else:
        raise ValueError(f"Unknown state action: {action_name}")
    return result, current_jd


def action_node(state: GraphState) -> Dict[str, Any]:
    """Reasoning Node가 선택한 도구(들)를 순차적으로 실행(Action Chaining)합니다."""
    from agent.application.run_context import raise_if_cancelled

    raise_if_cancelled()
    started_monotonic = time.perf_counter()
    logger.info("Executing Action Node (with potential Action Chaining)")

    execution_records: list[dict[str, Any]] = []

    raw_action_request = state.get("pending_action")
    try:
        action_request = (
            raw_action_request
            if isinstance(raw_action_request, ActionRequest)
            else ActionRequest.model_validate(raw_action_request)
            if raw_action_request is not None
            else None
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Action request state is invalid", error=str(exc))
        action_request = None

    if action_request is not None:
        request_metadata = dict(action_request.metadata or {})
        decision_capture_id = str(state.get("current_capture_id") or "")
        if decision_capture_id:
            request_metadata.setdefault("decision_capture_id", decision_capture_id)
        action_request.metadata = request_metadata

    if action_request and action_request.summary:
        logger.info(
            "Action request received",
            source=action_request.source,
            summary=action_request.summary,
        )

    if not action_request or not action_request.tool_calls:
        logger.warning("No validated action request is available.")
        elapsed = time.perf_counter() - started_monotonic
        action_result = ActionResult(
            source=action_request.source if action_request else "unknown",
            summary=action_request.summary if action_request else "",
            status="error",
            tool_results=[
                {
                    "action": "none",
                    "status": "error",
                    "error": "No validated action request",
                    "args": {},
                }
            ],
        )
        return {
            "pending_action": None,
            "last_action_result": action_result,
            "execution_records": [],
            "action_history": action_result.tool_results,
        }

    prior_actions = list(state.get("action_history", []) or [])
    new_actions = []
    current_jd        = dict(state.get("extracted_jd", {}))
    is_finished       = state.get("is_finished", False)
    collected_data    = list(state.get("collected_data", []))
    error_count       = state.get("error_count", 0)
    current_url       = state.get("current_url", "")
    current_url_stale = state.get("current_url_stale", True)
    pending_human_approval = bool(state.get("pending_human_approval", False))
    human_approval_request = dict(state.get("human_approval_request", {}) or {})
    latest_markers = list(state.get("current_markers", []) or [])
    latest_ui_context = state.get("ui_context", "")
    latest_marked_image = state.get("marked_image", "")
    latest_recent_images: list = []
    result_card_queue = [dict(item) for item in (state.get("result_card_queue", []) or []) if isinstance(item, dict)]
    result_page_memory = dict(state.get("result_page_memory", {}) or {})
    result_availability = dict(state.get("result_availability", {}) or {})
    active_result_card = dict(state.get("active_result_card", {}) or {})
    detail_ocr_buffer = dict(state.get("detail_ocr_buffer", {}) or {})
    detail_followup_required = dict(state.get("detail_followup_required", {}) or {})
    detail_return_pending = dict(state.get("detail_return_pending", {}) or {})
    screen_changed    = False
    chain_boundary    = False
    previous_ui_action: str | None = None
    pending_transition: dict[str, Any] = {}
    next_pending_action: ActionRequest | None = None
    reflex_transition_contracts = dict(state.get("reflex_transition_contracts", {}) or {})

    def transition_params() -> dict[str, Any]:
        params = dict(state.get("recipe_params", {}) or {})
        params.setdefault("goal", state.get("goal", ""))
        return params

    def transition_step_context(
        action_seq: int,
        action_name: str,
        args: dict,
        tool_call_id: str = "",
    ) -> dict[str, Any]:
        """전환 검증 로그에 남길 행동 step 묶음을 만든다."""

        step = {
            "seq": action_seq,
            "action": action_name,
            "decision_capture_id": str(
                action_request.metadata.get("decision_capture_id") or ""
            ),
            "args": _compact_action_args(action_name, args),
            "page_role": args.get("page_role") or state.get("current_page_role", ""),
            "target_role": args.get("target_role") or args.get("target_role_candidate") or "",
            "component": args.get("target_component") or args.get("component_candidate") or "",
            "expected_after": args.get("expected_after") or "",
        }
        if tool_call_id:
            step["tool_call_id"] = tool_call_id
        if action_request.source == "reflex":
            trace = dict(state.get("reflex_trace", {}) or {})
            call_trace = (trace.get("tool_calls") or {}).get(tool_call_id) if tool_call_id else None
            if isinstance(call_trace, dict):
                step.update(
                    {
                        "recipe_key": trace.get("recipe_key", ""),
                        "recipe_seq": call_trace.get("seq"),
                        "replay_mode": call_trace.get("replay_mode", ""),
                        "match_mode": call_trace.get("match_mode", ""),
                        "target_text": call_trace.get("target_text", ""),
                        "marker_id": call_trace.get("marker_id"),
                        "phash": call_trace.get("phash", {}),
                    }
                )
        return {key: value for key, value in step.items() if value not in (None, "", {}, [])}

    def set_pending_transition(
        action_seq: int,
        action_name: str,
        args: dict,
        contract: dict | None,
        source: str,
        tool_call_id: str = "",
    ) -> None:
        nonlocal pending_transition
        recipe_key = ""
        if source == "reflex":
            recipe_key = str((state.get("reflex_trace", {}) or {}).get("recipe_key") or "")
        pending_transition = {
            "action_seq": action_seq,
            "action": action_name,
            "from_capture_id": str(
                action_request.metadata.get("decision_capture_id")
                or state.get("current_capture_id")
                or ""
            ),
            "expected_after": str(args.get("expected_after") or ""),
            "source": source,
            "recipe_key": recipe_key,
            "tool_call_id": tool_call_id,
            "step": transition_step_context(action_seq, action_name, args, tool_call_id),
            "before_url": current_url or state.get("current_url", "") or "",
            "before_phash": str((state.get("screen_signature", {}) or {}).get("phash") or ""),
            "before_screenshot": str((state.get("recent_images", []) or [])[-1]) if state.get("recent_images") else "",
            "started_at": time.time(),
            "attempts": 0,
            "contract": dict(contract or {}),
            "params": transition_params(),
        }

    def next_action_seq() -> int:
        return len(prior_actions) + len(new_actions)

    def append_execution_record(
        action_name: str,
        args: dict[str, Any],
        enriched_result: dict[str, Any],
        before_snapshot: dict[str, Any],
        after_context: dict[str, Any],
        action_seq: int,
        *,
        record_ui: bool = False,
    ) -> None:
        """기록 노드가 처리할 직렬화 가능한 실행 결과를 만든다."""

        execution_records.append(
            {
                "request": action_request.model_dump(mode="json"),
                "action_name": action_name,
                "args": dict(args),
                "result": dict(enriched_result),
                "before_snapshot": dict(before_snapshot),
                "after_context": dict(after_context),
                "seq": action_seq,
                "record_ui": record_ui,
                "record_state": {
                    "goal": state.get("goal", ""),
                    "current_capture_id": str(
                        before_snapshot.get("capture_id") or ""
                    ),
                    "current_markers": list(state.get("current_markers", []) or []),
                    "current_url": before_snapshot.get("url", ""),
                    "current_page_role": state.get("current_page_role", ""),
                    "screen_signature": dict(state.get("screen_signature", {}) or {}),
                    "recent_images": list(state.get("recent_images", []) or []),
                    "marked_image": state.get("marked_image", ""),
                },
            }
        )

    # marker_id → bbox 변환 헬퍼
    def get_bbox(marker_id: int):
        marker = _marker_by_id(latest_markers, marker_id)
        if marker:
            return marker["bbox"]
        raise ValueError(f"Marker ID {marker_id} not found in current screen.")

    def enrich_result(
        result: dict,
        requested_action: str,
        action_args: dict,
        before_snapshot: dict,
        screen_change_expected: bool = False,
        tool_call_id: str = "",
        tool_call_metadata: dict[str, Any] | None = None,
        action_source: str = "",
    ) -> dict:
        result["args"] = _compact_action_args(requested_action, action_args)
        result["action_source"] = action_source or action_request.source
        if tool_call_id:
            result["tool_call_id"] = tool_call_id
        if tool_call_metadata:
            result["execution_metadata"] = dict(tool_call_metadata)
        result["before_url"] = before_snapshot.get("url", "")
        result["before_screenshot"] = before_snapshot.get("screenshot", "")
        result["before_marked_image"] = before_snapshot.get("marked_image", "")
        result["decision_capture_id"] = str(
            action_request.metadata.get("decision_capture_id")
            or before_snapshot.get("capture_id")
            or ""
        )
        result["screen_change_expected"] = screen_change_expected
        target = _action_target_metadata(state, requested_action, action_args)
        if target:
            result["target"] = target
        if action_request.source == "reflex":
            trace = dict(state.get("reflex_trace", {}) or {})
            if trace:
                result["reflex_recipe_key"] = trace.get("recipe_key", "")
                call_trace = (trace.get("tool_calls") or {}).get(tool_call_id) if tool_call_id else None
                if call_trace:
                    result["reflex_match"] = dict(call_trace)
        if result.get("action") != requested_action:
            result["requested_action"] = requested_action
        return result

    def append_guard_result(
        action_name: str,
        args: dict,
        before_snapshot: dict,
        status: str,
        reason: str,
        message: str,
        step_start: float,
        increments_error: bool = False,
        observation_required: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        nonlocal error_count, current_url_stale, screen_changed
        if observation_required:
            current_url_stale = True
            screen_changed = True
        result = {
            "status": status,
            "action": action_name,
            "result": message if status != "error" else None,
            "error": message if status == "error" else None,
            "reason": reason,
        }
        if observation_required:
            result["observation_required"] = True
        if details:
            result["guard"] = dict(details)
        action_seq = next_action_seq()
        if observation_required:
            set_pending_transition(action_seq, action_name, args, None, "guard")
        enriched = enrich_result(result, action_name, args, before_snapshot, False)
        new_actions.append(enriched)
        append_execution_record(
            action_name,
            args,
            enriched,
            before_snapshot,
            {
                "current_url": current_url,
                "current_url_stale": current_url_stale,
                "screen_changed": observation_required,
                "extracted_jd": current_jd,
                "is_finished": is_finished,
            },
            action_seq,
        )
        if increments_error:
            error_count += 1
        step_elapsed = time.perf_counter() - step_start
        logger.warning(message, action=action_name, reason=reason)

    # 도구 카테고리 라우팅 테이블
    def request_human_approval(action_name: str, args: dict, reason: str, before_snapshot: dict, step_start: float) -> None:
        nonlocal pending_human_approval, human_approval_request
        pending_human_approval = True
        human_approval_request = {
            "status": "needs_human_approval",
            "reason": reason,
            "action": action_name,
            "args": _compact_action_args(action_name, args),
            "current_url": current_url,
            "message": "Autonomous execution stopped before a sensitive or irreversible step.",
        }
        append_guard_result(
            action_name,
            args,
            before_snapshot,
            "skipped",
            reason,
            "Skipped sensitive action; human confirmation is required.",
            step_start,
        )

    UI_ACTIONS = {
        "click_marker",
        "type_in_marker",
        "scroll",
        "press_key",
        "open_browser",
        "close_browser",
        "close_current_tab",
        "switch_tab",
        "go_back",
    }
    SCREEN_CHANGING_ACTIONS = set(UI_ACTIONS)
    URL_STALE_ACTIONS = {
        "click_marker",
        "press_key",
        "open_browser",
        "close_browser",
        "close_current_tab",
        "switch_tab",
        "go_back",
    }
    STATE_ACTIONS = {"update_extracted_info", "finish_detail_reading", "set_result_card_queue"}
    RETURN_ACTIONS = {"go_back", "close_current_tab", "switch_tab"}

    for idx, tool_call in enumerate(action_request.tool_calls):
        action_name = tool_call.name
        args = dict(tool_call.args)
        call_metadata = dict(tool_call.metadata)
        action_context_args = {**args, **call_metadata}
        if action_name == "finish_detail_reading":
            args.setdefault("page_role", "job_detail")
            args.setdefault("detail_complete", True)
        compact_args = _compact_action_args(action_name, args)

        logger.info(
            "Executing requested tool",
            source=action_request.source,
            chain_position=f"{idx + 1}/{len(action_request.tool_calls)}",
            action=action_name,
            args=compact_args,
        )
        step_start = time.perf_counter()
        before_snapshot = _state_snapshot_for_action(state, current_url)
        action_seq = next_action_seq()
        policy_action_request: ActionRequest | None = None

        try:
            return_pending = _detail_return_pending_for_url(
                {
                    **state,
                    "detail_return_pending": detail_return_pending,
                    "current_url": current_url,
                },
                current_url,
            )
            if return_pending and (
                action_name in STATE_ACTIONS
                or action_name in UI_ACTIONS
                and action_name not in RETURN_ACTIONS
            ):
                append_guard_result(
                    action_name,
                    args,
                    before_snapshot,
                    "skipped",
                    "detail_return_pending",
                    (
                        "상세 수집이 이미 완료되었습니다. 같은 공고를 더 읽지 말고 "
                        "검색 결과 화면으로 복귀해야 합니다."
                    ),
                    step_start,
                )
                break
            if chain_boundary:
                append_guard_result(
                    action_name,
                    args,
                    before_snapshot,
                    "skipped",
                    "chain_boundary_after_screen_change",
                    "Skipped chained tool after a screen-changing action; next observation is required.",
                    step_start,
                )
                break
            if action_name in UI_ACTIONS:
                if (
                    action_name in {"click_marker", "type_in_marker"}
                    and action_request.source
                    not in {"reflex", "card_queue", "page_policy", "duplicate_job_policy"}
                ):
                    guard_result = _check_current_reasoning_screen(state)
                    if guard_result.get("stale"):
                        append_guard_result(
                            action_name,
                            args,
                            before_snapshot,
                            "skipped",
                            "screen_changed_during_reasoning",
                            "Skipped UI action because the screen changed while reasoning; a fresh perception is required.",
                            step_start,
                            observation_required=True,
                            details=guard_result,
                        )
                        break
                no_effect_transition = _latest_no_effect_transition(state)
                if _repeats_no_effect_target(
                    no_effect_transition,
                    action_name,
                    args,
                ):
                    append_guard_result(
                        action_name,
                        args,
                        before_snapshot,
                        "skipped",
                        "same_screen_no_effect_action_blocked",
                        (
                            "Blocked an atomic UI action that already had no effect on this screen. "
                            "Choose another navigation method."
                        ),
                        step_start,
                    )
                    break
                sensitive_reason = _sensitive_action_reason(
                    {**state, "current_markers": latest_markers},
                    action_name,
                    args,
                )
                if sensitive_reason:
                    request_human_approval(action_name, args, sensitive_reason, before_snapshot, step_start)
                    break

                if action_name == "type_in_marker":
                    target_rejection = _text_input_target_rejection(
                        latest_markers,
                        args.get("marker_id"),
                    )
                    if target_rejection:
                        append_guard_result(
                            action_name,
                            args,
                            before_snapshot,
                            "error",
                            target_rejection["reason"],
                            (
                                "Blocked type_in_marker because the selected marker does not look like "
                                "a text input target. Choose the visible input container or placeholder marker."
                            ),
                            step_start,
                            increments_error=True,
                        )
                        break

                if previous_ui_action and not _is_allowed_same_screen_ui_chain(previous_ui_action, action_name):
                    append_guard_result(
                        action_name,
                        args,
                        before_snapshot,
                        "skipped",
                        "unsafe_ui_action_chain",
                        f"Skipped unsafe UI chain: {previous_ui_action} -> {action_name}",
                        step_start,
                    )
                    break

                if action_name == "open_browser":
                    result = _dispatch_ui(action_name, args, get_bbox, current_url=current_url)
                else:
                    result = _dispatch_ui(action_name, args, get_bbox)
                action_changed_screen = action_name in SCREEN_CHANGING_ACTIONS
                if action_name == "open_browser":
                    result_payload = result.get("result") if isinstance(result.get("result"), dict) else {}
                    action_changed_screen = bool(result_payload.get("opened"))
                    if not action_changed_screen and not state.get("ui_context"):
                        action_changed_screen = True
                    current_url = result_payload.get("url") or args["url"]
                    current_url_stale = action_changed_screen
                else:
                    current_url_stale = current_url_stale or action_name in URL_STALE_ACTIONS
                screen_changed = screen_changed or action_changed_screen
                previous_ui_action = action_name
                if action_changed_screen:
                    contract = reflex_transition_contracts.get(tool_call.id)
                    transition_source = (
                        action_request.source
                        if action_request.source
                        in {"card_queue", "reflex", "page_policy", "duplicate_job_policy"}
                        else "autonomous"
                    )
                    transition_source = str(
                        call_metadata.get("transition_source") or transition_source
                    )
                    set_pending_transition(
                        action_seq,
                        action_name,
                        args,
                        contract,
                        transition_source,
                        tool_call.id,
                    )
                if action_changed_screen and _chain_boundary_reached(action_name):
                    chain_boundary = True
                if (
                    result.get("status") == "success"
                    and action_name == "click_marker"
                    and _result_card_click_matches_queue(
                        result_card_queue,
                        action_context_args,
                    )
                ):
                    result_card_queue, active_result_card = _mark_result_card_active(
                        result_card_queue,
                        action_context_args,
                    )

            elif action_name in STATE_ACTIONS:
                result, current_jd = _dispatch_state(
                    action_name,
                    args,
                    current_jd,
                    current_url=current_url,
                    state={
                        **state,
                        "extracted_jd": current_jd,
                        "current_url": current_url,
                        "detail_ocr_buffer": detail_ocr_buffer,
                    },
                )
                action_changed_screen = False
                if action_name == "set_result_card_queue":
                    result_card_queue = list(result.pop("_result_card_queue", []) or [])
                    result_page_memory = dict(result.pop("_result_page_memory", {}) or {})
                    observed_availability = dict(result.pop("_result_availability", {}) or {})
                    if observed_availability:
                        result_availability = observed_availability
                    pending_cards = _pending_result_cards(result_card_queue)
                    if pending_cards:
                        first_card = pending_cards[0]
                        marker_id = first_card.get("source_marker_id")
                        if marker_id is not None:
                            queue_id = str(first_card.get("queue_id") or "")
                            policy_action_request = build_action_request(
                                "card_queue",
                                "result_card_queue_first_item",
                                [
                                    {
                                        "name": "click_marker",
                                        "args": {
                                            "marker_id": marker_id,
                                            "target_label": first_card.get("title", ""),
                                            "target_role": "job_card",
                                            "target_component": "job_card_title",
                                            "page_role": "search",
                                            "reason": "result card queue stored; open the first pending card",
                                            "expected_after": "selected job detail page is visible",
                                        },
                                        "id": f"card_queue_{queue_id or 'first'}",
                                        "metadata": {"queue_id": queue_id},
                                    }
                                ],
                            )
                    elif _result_card_queue_scope_complete(
                        result_card_queue,
                        count_mode=_count_mode_from_state(state),
                        target_count=_target_count_from_state(state),
                    ):
                        is_finished = True
                        resolved_count = _completed_result_card_count(result_card_queue)
                        collected_data.append(
                            f"Auto-finished after resolving {resolved_count} existing result card(s)."
                        )
                        result["auto_finished"] = True
                        result["resolved_count"] = resolved_count
                if action_name == "finish_detail_reading":
                    detail_ocr_buffer = dict(result.pop("_detail_ocr_buffer", detail_ocr_buffer) or {})
                    detail_followup_required = dict(
                        result.pop(
                            "_detail_followup_required",
                            detail_followup_required,
                        )
                        or {}
                    )
                if (
                    action_name in {"update_extracted_info", "finish_detail_reading"}
                    and result.get("status") == "success"
                    and _is_detail_update(args)
                ):
                    target_count = _target_count_from_state(state)
                    collected_count = _extracted_job_count(current_jd)
                    detail_complete = args.get("detail_complete")
                    if detail_complete is True:
                        result_card_queue, active_result_card = _complete_active_result_card(result_card_queue, active_result_card)
                        result["detail_policy"] = "detail_complete"
                        pending_cards = _pending_result_cards(result_card_queue)
                        resolved_count = max(
                            collected_count,
                            _completed_result_card_count(result_card_queue),
                        )
                        if pending_cards or (target_count > 0 and resolved_count < target_count):
                            detail_return_pending = {
                                "url": current_url,
                                "reason": "detail_complete",
                                "pending_count": len(pending_cards),
                                "completed_action_seq": action_seq,
                            }
                            no_effect_return = _latest_no_effect_transition(state)
                            return_action = _normalized_return_action(
                                result_page_memory.get("return_action")
                            )
                            failed_return_action = str(
                                no_effect_return.get("action") or ""
                            )
                            if (
                                not return_action
                                or failed_return_action == return_action.get("name")
                            ):
                                # 실패가 확인된 복귀 방식을 자동 반복하지 않고 LLM이 다른 원자 도구를 고르게 합니다.
                                result["detail_policy"] = "return_requires_reasoning"
                                if failed_return_action:
                                    result["failed_return_action"] = failed_return_action
                            else:
                                return_name = str(return_action["name"])
                                return_args = {
                                    **dict(return_action.get("args") or {}),
                                    "reason": "이전에 확인된 검색 결과 복귀 행동을 재사용합니다.",
                                    "expected_after": "검색 결과 목록이 표시된다.",
                                }
                                policy_action_request = build_action_request(
                                    "page_policy",
                                    "reuse_confirmed_result_return_action",
                                    [
                                        {
                                            "name": return_name,
                                            "args": return_args,
                                            "id": f"detail_policy_{return_name}",
                                        }
                                    ],
                                )
                        elif (
                            _auto_finish_on_target_enabled()
                            and _count_mode_from_state(state) == "visible_all"
                            and result_card_queue
                            and not pending_cards
                        ):
                            is_finished = True
                            collected_data.append(
                                f"Auto-finished after collecting all {collected_count} visible jobs."
                            )
                            result["auto_finished"] = True
                            result["count_mode"] = "visible_all"
                            result["collected_count"] = collected_count
                if (
                    action_name in {"update_extracted_info", "finish_detail_reading"}
                    and result.get("status") == "success"
                    and _auto_finish_on_target_enabled()
                    and args.get("detail_complete") is not False
                ):
                    target_count = _target_count_from_state(state)
                    collected_count = _extracted_job_count(current_jd)
                    resolved_count = max(
                        collected_count,
                        _completed_result_card_count(result_card_queue),
                    )
                    if target_count > 0 and resolved_count >= target_count:
                        detail_return_pending = {}
                        is_finished = True
                        collected_data.append(
                            f"Auto-finished after collecting target_count={target_count} jobs."
                        )
                        result["auto_finished"] = True
                        result["target_count"] = target_count
                        result["collected_count"] = collected_count
                        result["resolved_count"] = resolved_count

            elif action_name == "finish_task":
                action_tools = _get_action_tools()
                result = action_tools.finish_task(args["result"])
                is_finished = True
                collected_data.append(args["result"])
                action_changed_screen = False

            else:
                raise ValueError(f"Unknown tool: {action_name}")

            enriched = enrich_result(
                result,
                action_name,
                args,
                before_snapshot,
                action_changed_screen,
                tool_call.id,
                call_metadata,
            )
            new_actions.append(enriched)
            append_execution_record(
                action_name,
                args,
                enriched,
                before_snapshot,
                {
                    "current_url": current_url,
                    "current_url_stale": current_url_stale,
                    "screen_changed": action_changed_screen,
                    "extracted_jd": current_jd,
                    "is_finished": is_finished,
                },
                action_seq,
                record_ui=action_name in UI_ACTIONS,
            )

            step_elapsed = time.perf_counter() - step_start
            logger.info(f"Action Node [{action_name}] completed in {step_elapsed:.2f} seconds")

            if policy_action_request and not is_finished:
                next_pending_action = policy_action_request
                logger.info(
                    "Deterministic follow-up action queued",
                    source=policy_action_request.source,
                    reason=policy_action_request.summary,
                )
                break

            if is_finished:
                break

        except Exception as e:
            logger.error(f"Failed to execute action {action_name}", error=str(e))
            step_elapsed = time.perf_counter() - step_start
            before_snapshot = _state_snapshot_for_action(state, current_url)
            result = {"action": action_name, "status": "error", "error": str(e)}
            enriched = enrich_result(
                result,
                action_name,
                args,
                before_snapshot,
                False,
                tool_call.id,
                call_metadata,
            )
            new_actions.append(enriched)
            append_execution_record(
                action_name,
                args,
                enriched,
                before_snapshot,
                {
                    "current_url": current_url,
                    "current_url_stale": current_url_stale,
                    "screen_changed": False,
                    "extracted_jd": current_jd,
                    "is_finished": is_finished,
                },
                action_seq,
            )
            error_count += 1
            break  # 에러 발생 시 체인 중단

    total_elapsed = time.perf_counter() - started_monotonic
    logger.info(
        "Action Node completed all chained tools",
        duration_sec=round(total_elapsed, 6),
    )

    statuses = [str(item.get("status") or "") for item in new_actions]
    if statuses and all(status == "success" for status in statuses):
        request_status = "success"
    elif any(status == "success" for status in statuses):
        request_status = "partial"
    elif statuses and all(status == "error" for status in statuses):
        request_status = "error"
    else:
        request_status = "partial" if statuses else "error"
    action_result = ActionResult(
        source=action_request.source,
        summary=action_request.summary,
        status=request_status,
        tool_results=new_actions,
        screen_changed=screen_changed,
        is_finished=is_finished,
    )
    return {
        "pending_action":      next_pending_action,
        "last_action_result": action_result,
        "execution_records": execution_records,
        "action_history":    new_actions,
        "extracted_jd":      current_jd,
        "is_finished":       is_finished,
        "collected_data":    collected_data,
        "error_count":       error_count,
        "current_url":       current_url,
        "current_url_stale": current_url_stale,
        "current_markers":   latest_markers,
        "ui_context":        latest_ui_context,
        "marked_image":      latest_marked_image,
        "screen_signature":  dict(state.get("screen_signature", {}) or {}),
        "recent_images":     latest_recent_images,
        "pending_transition": pending_transition,
        "observed_transition": pending_transition,
        "transition_status": "",
        "transition_outcome": "",
        "transition_source": "",
        "transition_reason": "",
        "transition_visual_change_detected": False,
        "transition_visual_change_ratio": None,
        "ocr_required": False,
        "result_card_queue": result_card_queue,
        "result_page_memory": result_page_memory,
        "result_availability": result_availability,
        "active_result_card": active_result_card,
        "queue_replay_trace": {},
        "page_policy_trace": {},
        "detail_ocr_buffer": detail_ocr_buffer,
        "detail_followup_required": detail_followup_required,
        "detail_return_pending": detail_return_pending,
        "pending_human_approval": pending_human_approval,
        "human_approval_request": human_approval_request,
    }

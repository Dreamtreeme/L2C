import json
import os
import time
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from agent.application.detail_extraction_service import (
    extract_job_from_detail_ocr_buffer as _extract_job_from_detail_ocr_buffer,
)
from agent.graph.action_request import build_action_message
from agent.graph.state import GraphState
from agent.graph.tool_schema import (
    click_marker,
    close_browser,
    close_current_tab,
    finish_detail_reading,
    finish_task,
    go_back,
    open_browser,
    press_key,
    scroll,
    set_result_card_queue,
    switch_tab,
    type_in_marker,
    update_extracted_info,
    update_plan_progress,
)
from agent.prompts.commander import COMMANDER_SYSTEM_PROMPT
from agent.runtime.action_validation import (
    IMPLAUSIBLE_TEXT_INPUT_TARGET,
    text_input_target_rejection as _text_input_target_rejection,
)
from agent.runtime.action_guard import check_reasoning_screen_stale as _check_reasoning_screen_stale
from agent.runtime.detail_runtime import (
    build_detail_lightweight_marked_image as _build_detail_lightweight_marked_image,
    build_detail_section_context as _build_detail_section_context,
    compact_detail_ocr_buffer_context as _compact_detail_ocr_buffer_context,
    detail_page_policy_message as _detail_page_policy_message,
    env_enabled as _env_enabled,
    is_icon_marker as _is_icon_marker,
    marker_prompt_rank as _marker_prompt_rank,
    update_detail_ocr_buffer as _update_detail_ocr_buffer,
)
from agent.runtime.duplicate_job_policy import existing_job_url_trace as _existing_job_url_trace
from agent.runtime.job_collection import JOB_LIST_KEYS, job_list_value as _job_list_value
from agent.runtime.result_card_queue import (
    completed_result_card_count as _completed_result_card_count,
    complete_active_result_card as _complete_active_result_card,
    mark_result_card_active as _mark_result_card_active,
    marker_by_id as _marker_by_id,
    normalize_result_card_queue as _normalize_result_card_queue,
    pending_result_cards as _pending_result_cards,
    queue_card_label as _queue_card_label,
    queue_replay_after_return as _queue_replay_after_return,
    result_card_queue_scope_complete as _result_card_queue_scope_complete,
    result_card_click_matches_queue as _result_card_click_matches_queue,
    result_card_entries_from_args as _result_card_entries_from_args,
    skip_active_result_card as _skip_active_result_card,
)
from agent.runtime.result_card_selector import select_result_cards as _select_result_cards
from agent.runtime.reflex_runtime import reflex_node
from agent.runtime.site_context import (
    infer_site_page_role as _infer_site_page_role,
    is_job_detail_context as _is_job_detail_context,
    looks_like_job_detail_url as _looks_like_job_detail_url,
    persistence_policy_for_url as _persistence_policy_for_url,
    site_runtime_guidance as _site_runtime_guidance,
)
from agent.runtime.transition_runtime import (
    build_transition_observation as _transition_observation,
    raw_screen_phash_signature as _raw_screen_phash_signature,
    transition_accepts_visual_change as _transition_accepts_visual_change,
    transition_has_visual_change as _transition_has_visual_change,
    transition_no_effect_by_phash as _transition_no_effect_by_phash,
    transition_phash_distance as _transition_phash_distance,
)
from agent.utils.logger import logger
from agent.vision.marker_geometry import (
    bbox_to_ratio,
    center_ratio_from_bbox,
)

_perception = None
_action_tools = None
_ui_llm_with_tools: dict[tuple[str, ...], Any] = {}

_ACTION_TOOL_SCHEMAS = {
    schema.__name__: schema
    for schema in (
        click_marker,
        type_in_marker,
        scroll,
        press_key,
        open_browser,
        close_browser,
        close_current_tab,
        update_extracted_info,
        finish_detail_reading,
        go_back,
        update_plan_progress,
        set_result_card_queue,
        switch_tab,
        finish_task,
    )
}

def _get_perception():
    """비전 엔진은 실제 브라우저 제어 경로에서만 초기화합니다."""
    global _perception
    if _perception is None:
        from agent.tools.perception import PerceptionEngine

        _perception = PerceptionEngine()
    return _perception


def _get_action_tools():
    """물리 조작 도구는 비전 엔진과 같은 생명주기로 lazy 초기화합니다."""
    global _action_tools
    if _action_tools is None:
        from agent.tools.actions import ActionTools

        _action_tools = ActionTools(_get_perception())
    return _action_tools


def _check_current_reasoning_screen(state: GraphState) -> dict[str, Any]:
    """저장된 pHash가 있을 때만 행동 직전 화면 검사를 초기화한다."""

    if not str((state.get("screen_signature") or {}).get("phash") or ""):
        return {"checked": False, "stale": False, "reason": "previous_phash_missing"}
    return _check_reasoning_screen_stale(state, _get_perception())


def _get_ui_llm_with_tools(allowed_tool_names: tuple[str, ...] | None = None):
    """선택된 사이트가 허용한 도구만 바인딩한 모델을 재사용한다."""
    global _ui_llm_with_tools
    names = tuple(
        name
        for name in (allowed_tool_names or tuple(_ACTION_TOOL_SCHEMAS))
        if name in _ACTION_TOOL_SCHEMAS
    )
    if not names:
        names = tuple(_ACTION_TOOL_SCHEMAS)
    if names not in _ui_llm_with_tools:
        from agent.application.model_clients import get_google_chat_model

        llm = get_google_chat_model("gemini-3.5-flash", temperature=0.1)
        _ui_llm_with_tools[names] = llm.bind_tools(
            [_ACTION_TOOL_SCHEMAS[name] for name in names]
        )
    return _ui_llm_with_tools[names]


def _allowed_tool_names_for_state(state: GraphState) -> tuple[str, ...]:
    """현재 사이트 프로필의 허용 도구 목록을 반환한다."""

    from agent.runtime.site_context import site_profile_for_url

    profile = site_profile_for_url(str(state.get("current_url") or ""))
    tools = profile.get("tools", {}) if isinstance(profile, dict) else {}
    configured = tools.get("allowed_tools", []) if isinstance(tools, dict) else []
    names = tuple(str(name) for name in configured if str(name) in _ACTION_TOOL_SCHEMAS)
    return names or tuple(_ACTION_TOOL_SCHEMAS)


def prepare_reasoning_models() -> None:
    """브라우저 준비 중 범용 판단 모델과 카드 선택 모델을 미리 생성한다."""

    _get_ui_llm_with_tools()
    from agent.runtime.result_card_selector import prepare_result_card_selector_model

    prepare_result_card_selector_model()


def perception_node(
    state: GraphState,
    *,
    max_capture_attempts: int | None = None,
) -> Dict[str, Any]:
    """화면을 캡처하고 마커를 파싱하여 상태를 업데이트합니다."""
    from agent.application.run_context import raise_if_cancelled

    raise_if_cancelled()
    started_monotonic = time.perf_counter()
    logger.info("Executing Perception Node")
    perception = _get_perception()

    pending_before_capture = dict(state.get("pending_transition", {}) or {})
    pending_action = str(pending_before_capture.get("action") or "")
    if pending_action in {
        "click_marker",
        "press_key",
        "open_browser",
        "go_back",
        "close_current_tab",
        "switch_tab",
    }:
        wait_for_change = getattr(perception, "wait_for_transition_change", None)
        before_screenshot = str(pending_before_capture.get("before_screenshot") or "")
        if callable(wait_for_change) and before_screenshot:
            try:
                wait_for_change(before_screenshot)
            except Exception as exc:
                logger.debug("Transition screen change wait skipped", error=str(exc))
    
    # 화면 캡처
    capture_usable = getattr(perception, "capture_usable_screen", None)
    if callable(capture_usable):
        if pending_action == "type_in_marker":
            try:
                input_wait_sec = max(0.0, float(os.getenv("VISION_INPUT_CAPTURE_INITIAL_WAIT_SEC", "0.7")))
            except ValueError:
                input_wait_sec = 0.7
            image_path = capture_usable(
                max_attempts=max_capture_attempts,
                initial_wait_sec=input_wait_sec,
            )
        else:
            image_path = capture_usable(max_attempts=max_capture_attempts)
    else:
        image_path = perception.capture_screen()

    capture_quality = dict(getattr(perception, "last_capture_quality", {}) or {})
    if capture_quality.get("low_information"):
        elapsed = time.perf_counter() - started_monotonic
        retry_count = int(state.get("low_information_retry_count", 0) or 0) + 1
        logger.info(
            "Low-information screen skipped before OCR",
            retry_count=retry_count,
            duration_sec=round(elapsed, 6),
            **capture_quality,
        )
        return {
            "recent_images": [image_path],
            "marked_image": "",
            "current_markers": [],
            "ui_context": "",
            "screen_signature": {},
            "current_url": state.get("current_url", ""),
            "current_page_role": "",
            "current_url_stale": True,
            "pending_transition": state.get("pending_transition", {}),
            "transition_status": "pending" if state.get("pending_transition") else "",
            "transition_outcome": "",
            "transition_source": state.get("transition_source", ""),
            "queue_replay_hit": False,
            "queue_replay_trace": {},
            "page_policy_hit": False,
            "page_policy_trace": {},
            "detail_ocr_buffer": state.get("detail_ocr_buffer", {}),
            "low_information_screen": True,
            "low_information_retry_count": retry_count,
            "step_durations": [
                {
                    "node": "perception",
                    "duration": elapsed,
                    "ocr_skipped": True,
                    "reason": "low_information_screen",
                }
            ],
        }

    current_url = state.get("current_url", "")
    current_url_stale = state.get("current_url_stale", True)
    if current_url_stale or not current_url:
        fetched_url = perception.get_current_url()
        if fetched_url:
            current_url = fetched_url
        current_url_stale = False

    transition_observations = []
    transition_status = ""
    transition_outcome = ""
    transition_source = ""
    pending_transition = dict(state.get("pending_transition", {}) or {})
    observed_transition = dict(pending_transition)
    reflex_blocked_recipe_keys = [
        str(key)
        for key in state.get("reflex_blocked_recipe_keys", []) or []
        if str(key)
    ]
    raw_screen_signature = {}
    visual_change_detected = False
    visual_change_ratio = None
    if pending_transition:
        raw_screen_signature = _raw_screen_phash_signature(image_path)
        visual_change_detected, visual_change_ratio = _transition_has_visual_change(
            pending_transition,
            image_path,
        )
        active_result_card = dict(state.get("active_result_card", {}) or {})
        duplicate_trace = (
            _existing_job_url_trace(current_url, state.get("extracted_jd", {}))
            if active_result_card and _looks_like_job_detail_url(current_url)
            else {"matched": False, "reason": "not_active_job_detail"}
        )
        if duplicate_trace.get("matched"):
            skipped_queue_id = str(active_result_card.get("queue_id") or "")
            result_card_queue, active_result_card = _skip_active_result_card(
                [
                    dict(item)
                    for item in (state.get("result_card_queue", []) or [])
                    if isinstance(item, dict)
                ],
                active_result_card,
                reason="existing_detail_url",
                url=current_url,
                job_id=duplicate_trace.get("job_id"),
            )
            queue_complete = _result_card_queue_scope_complete(
                result_card_queue,
                count_mode=_count_mode_from_state(state),
                target_count=_target_count_from_state(state),
            )
            duplicate_message = build_action_message(
                "duplicate_job_policy",
                "skip detail OCR for an already collected job",
                [
                    {
                        "name": "finish_task" if queue_complete else "go_back",
                        "args": (
                            {
                                "result": "현재 검색 결과 큐의 모든 공고 처리를 마쳤습니다.",
                            }
                            if queue_complete
                            else {
                                "reason": "이미 수집한 공고 URL이므로 상세 읽기를 생략합니다.",
                                "expected_after": "검색 결과 목록으로 돌아간다.",
                            }
                        ),
                        "id": "skip_existing_job_detail",
                    }
                ],
            )
            elapsed = time.perf_counter() - started_monotonic
            logger.info(
                "Existing job detail skipped before OCR",
                url=current_url,
                source=duplicate_trace.get("source", ""),
                queue_id=skipped_queue_id,
                duration_sec=round(elapsed, 6),
            )
            return {
                "recent_images": [image_path],
                "marked_image": "",
                "current_markers": [],
                "ui_context": "",
                "screen_signature": raw_screen_signature,
                "current_url": current_url,
                "current_page_role": "job_detail",
                "current_url_stale": current_url_stale,
                "pending_transition": {},
                "transition_status": "ready",
                "transition_outcome": "existing_job_detail",
                "transition_source": str(pending_transition.get("source") or ""),
                "queue_replay_hit": False,
                "queue_replay_trace": {},
                "page_policy_hit": True,
                "page_policy_trace": {
                    "policy": (
                        "finish_existing_job_queue"
                        if queue_complete
                        else "skip_existing_job_detail"
                    ),
                    "queue_id": skipped_queue_id,
                    **duplicate_trace,
                },
                "detail_ocr_buffer": state.get("detail_ocr_buffer", {}),
                "result_card_queue": result_card_queue,
                "active_result_card": active_result_card,
                "last_action_result": duplicate_message,
                "low_information_screen": False,
                "low_information_retry_count": 0,
                "step_durations": [
                    {
                        "node": "perception",
                        "duration": elapsed,
                        "ocr_skipped": True,
                        "reason": "existing_job_detail",
                    }
                ],
            }
        no_effect, no_effect_distance = _transition_no_effect_by_phash(
            pending_transition,
            current_url,
            raw_screen_signature,
        )
        if no_effect and not visual_change_detected:
            transition_source = str(pending_transition.get("source") or "")
            no_effect_reason = (
                "reflex_no_screen_change"
                if transition_source == "reflex"
                else "no_screen_change"
            )
            started_at = float(pending_transition.get("started_at") or time.time())
            elapsed_sec = max(0.0, time.time() - started_at)
            attempt = int(pending_transition.get("attempts") or 0) + 1
            markers = list(state.get("current_markers", []) or [])
            marked_image = str(state.get("marked_image", "") or "")
            recipe_key = str(pending_transition.get("recipe_key") or "")
            if transition_source == "reflex" and recipe_key and recipe_key not in reflex_blocked_recipe_keys:
                reflex_blocked_recipe_keys.append(recipe_key)
            prior_signature = dict(state.get("screen_signature", {}) or {})
            screen_signature = {**prior_signature, **raw_screen_signature}
            transition_observations.append(
                _transition_observation(
                    pending_transition,
                    transition_status="unknown",
                    transition_outcome="",
                    transition_source=transition_source,
                    reason=no_effect_reason,
                    elapsed_sec=elapsed_sec,
                    attempt=attempt,
                    markers=markers,
                    screenshot=str(image_path),
                    marked_image=marked_image,
                    phash_distance=no_effect_distance,
                    visual_change_ratio=visual_change_ratio,
                    ocr_skipped=True,
                )
            )
            total_elapsed = time.perf_counter() - started_monotonic
            logger.info(
                "Transition no-effect detected by pHash before OCR",
                source=transition_source,
                action=pending_transition.get("action", ""),
                recipe_key=recipe_key[:24],
                phash_distance=no_effect_distance,
                duration=f"{total_elapsed:.2f}s",
            )
            return {
                "recent_images": [image_path],
                "marked_image": marked_image,
                "current_markers": markers,
                "ui_context": state.get("ui_context", ""),
                "screen_signature": screen_signature,
                "current_url": current_url,
                "current_page_role": state.get("current_page_role", ""),
                "current_url_stale": current_url_stale,
                "pending_transition": {},
                "transition_status": "unknown",
                "transition_outcome": "",
                "transition_source": transition_source,
                "transition_observations": transition_observations,
                "reflex_blocked_recipe_keys": reflex_blocked_recipe_keys,
                "queue_replay_hit": False,
                "queue_replay_trace": {},
                "detail_ocr_buffer": state.get("detail_ocr_buffer", {}),
                "low_information_screen": False,
                "low_information_retry_count": 0,
                "step_durations": [{"node": "perception", "duration": total_elapsed, "ocr_skipped": True}],
            }

    if observed_transition:
        queue_msg, cached_markers, queue_trace = _queue_replay_after_return(
            state,
            observed_transition,
            current_url,
            [],
            raw_screen_signature,
            require_anchors=False,
        )
        if queue_msg:
            elapsed = time.perf_counter() - started_monotonic
            result_page_memory = dict(state.get("result_page_memory") or {})
            saved_signature = dict(result_page_memory.get("screen_signature") or {})
            screen_signature = {**saved_signature, **raw_screen_signature}
            logger.info(
                "Result card queue replay prepared before OCR",
                queue_id=queue_trace.get("queue_id", ""),
                title=queue_trace.get("title", ""),
                reason=((queue_trace.get("return_match") or {}).get("reason") or ""),
                duration_sec=round(elapsed, 6),
            )
            return {
                "recent_images": [image_path],
                "marked_image": str(result_page_memory.get("marked_image") or ""),
                "current_markers": cached_markers,
                "ui_context": "",
                "screen_signature": screen_signature,
                "current_url": current_url,
                "current_page_role": "search",
                "current_url_stale": current_url_stale,
                "pending_transition": {},
                "transition_status": "ready",
                "transition_outcome": "queue_return_phash_match",
                "transition_source": str(observed_transition.get("source") or ""),
                "queue_replay_hit": True,
                "queue_replay_trace": queue_trace,
                "page_policy_hit": False,
                "page_policy_trace": {},
                "detail_ocr_buffer": state.get("detail_ocr_buffer", {}),
                "last_action_result": queue_msg,
                "low_information_screen": False,
                "low_information_retry_count": 0,
                "step_durations": [
                    {
                        "node": "perception",
                        "duration": elapsed,
                        "ocr_skipped": True,
                        "reason": "queue_return_phash_match",
                        "queue_replay_hit": True,
                    }
                ],
            }

    analysis = perception.analyze_ui(image_path)
    analysis_mode = str(analysis.get("analysis_mode") or "full")
    markers = analysis.get("markers", [])
    marked_image = analysis.get("marked_image", "")
    
    screen_signature = {}
    try:
        from agent.vision.screen_signature import build_capture_context, compute_screen_signature

        screen_signature = compute_screen_signature(image_path, markers)
        capture_context = build_capture_context(
            list(screen_signature.get("size") or []),
            int(analysis.get("content_top", 0) or 0),
        )
        if capture_context:
            screen_signature["capture_context"] = capture_context
        if raw_screen_signature.get("phash"):
            screen_signature["phash"] = raw_screen_signature.get("phash")
            screen_signature["size"] = raw_screen_signature.get("size") or screen_signature.get("size")
    except Exception as e:
        logger.debug("screen signature skipped", error=str(e))
    try:
        from agent.recipe.transition import evaluate_transition

        if pending_transition:
            started_at = float(pending_transition.get("started_at") or time.time())
            elapsed_sec = max(0.0, time.time() - started_at)
            evaluation = evaluate_transition(
                pending_transition.get("contract"),
                markers,
                params=dict(pending_transition.get("params", {}) or {}),
                elapsed_sec=elapsed_sec,
            )
            transition_status = evaluation["status"]
            transition_outcome = evaluation.get("outcome", "")
            transition_source = str(pending_transition.get("source") or "")
            same_url, phash_distance, no_effect_max_distance = _transition_phash_distance(
                pending_transition,
                current_url,
                screen_signature,
            )
            if (
                transition_source == "reflex"
                and transition_status == "ready"
                and same_url
                and phash_distance is not None
                and phash_distance <= no_effect_max_distance
                and not visual_change_detected
            ):
                transition_status = "unknown"
                transition_outcome = ""
                evaluation = {
                    **evaluation,
                    "status": "unknown",
                    "outcome": "",
                    "reason": "reflex_no_screen_change",
                    "phash_distance": phash_distance,
                }
            elif (
                transition_status == "pending"
                and same_url
                and phash_distance is not None
                and _transition_accepts_visual_change(pending_transition)
                and (
                    phash_distance > no_effect_max_distance
                    or visual_change_detected
                )
            ):
                transition_status = "ready"
                transition_outcome = ""
                evaluation = {
                    **evaluation,
                    "status": "ready",
                    "outcome": "",
                    "reason": (
                        "screen_change_pixels_matched"
                        if visual_change_detected
                        else "screen_change_phash_matched"
                    ),
                    "phash_distance": phash_distance,
                }
            attempt = int(pending_transition.get("attempts") or 0) + 1
            transition_observations.append(
                _transition_observation(
                    pending_transition,
                    transition_status=transition_status,
                    transition_outcome=transition_outcome,
                    transition_source=transition_source,
                    reason=evaluation.get("reason", ""),
                    elapsed_sec=elapsed_sec,
                    attempt=attempt,
                    markers=markers,
                    screenshot=str(image_path),
                    marked_image=str(marked_image or ""),
                    phash_distance=phash_distance,
                    visual_change_ratio=visual_change_ratio,
                    ocr_skipped=False,
                )
            )
            if transition_status == "pending":
                pending_transition["attempts"] = attempt
            else:
                if transition_status == "unknown" and transition_source == "reflex":
                    recipe_key = str(pending_transition.get("recipe_key") or "")
                    if recipe_key and recipe_key not in reflex_blocked_recipe_keys:
                        reflex_blocked_recipe_keys.append(recipe_key)
                pending_transition = {}
    except Exception as e:
        logger.debug("transition observation skipped", error=str(e))

    queue_msg = None
    queue_trace: dict[str, Any] = {}
    if observed_transition:
        queue_msg, markers, queue_trace = _queue_replay_after_return(
            state,
            observed_transition,
            current_url,
            markers,
            screen_signature,
        )
        if queue_msg:
            logger.info(
                "Result card queue replay prepared",
                queue_id=queue_trace.get("queue_id", ""),
                title=queue_trace.get("title", ""),
                reason=((queue_trace.get("return_match") or {}).get("reason") or ""),
            )

    current_page_role = _infer_current_page_role(current_url, markers)
    ui_context = _build_ui_context(
        markers,
        current_url=current_url,
        page_role=current_page_role,
    )
    detail_marked_image = _build_detail_lightweight_marked_image(
        image_path,
        markers,
        current_url,
        page_role=current_page_role,
    )
    if detail_marked_image:
        marked_image = detail_marked_image
    detail_ocr_buffer = _update_detail_ocr_buffer(
        state.get("detail_ocr_buffer", {}),
        markers,
        current_url,
        image_path,
        page_role=current_page_role,
        detail_key=_detail_key_from_state(state),
    )
    page_policy_msg, page_policy_trace = _detail_page_policy_message(
        current_url,
        markers,
        detail_ocr_buffer,
        page_role=current_page_role,
        transition_status=transition_status,
    )
    if page_policy_msg:
        logger.info(
            "Detail page policy prepared",
            policy=page_policy_trace.get("policy", ""),
            screen_count=page_policy_trace.get("screen_count", 0),
            added_lines=page_policy_trace.get("added_lines_last_screen", 0),
            total_lines=page_policy_trace.get("total_lines", 0),
        )
    
    elapsed = time.perf_counter() - started_monotonic
    logger.info("Perception Node completed", duration_sec=round(elapsed, 6))
    result = {
        "recent_images": [image_path],
        "marked_image": marked_image,
        "current_markers": markers,
        "ui_context": ui_context,
        "screen_signature": screen_signature,
        "current_url": current_url,
        "current_page_role": current_page_role,
        "current_url_stale": current_url_stale,
        "pending_transition": pending_transition,
        "transition_status": transition_status,
        "transition_outcome": transition_outcome,
        "transition_source": transition_source,
        "transition_observations": transition_observations,
        "reflex_blocked_recipe_keys": reflex_blocked_recipe_keys,
        "queue_replay_hit": bool(queue_msg),
        "queue_replay_trace": queue_trace if queue_msg else {},
        "page_policy_hit": bool(page_policy_msg) and not bool(queue_msg),
        "page_policy_trace": page_policy_trace if page_policy_msg and not queue_msg else {},
        "detail_ocr_buffer": detail_ocr_buffer,
        "low_information_screen": False,
        "low_information_retry_count": 0,
        "step_durations": [
            {
                "node": "perception",
                "duration": elapsed,
                "ocr_skipped": False,
                "marker_count": len(markers),
                "queue_replay_hit": bool(queue_msg),
                "page_policy_hit": bool(page_policy_msg) and not bool(queue_msg),
                "analysis_mode": analysis_mode,
            }
        ]
    }
    if queue_msg:
        result["last_action_result"] = queue_msg
    elif page_policy_msg:
        result["last_action_result"] = page_policy_msg
    return result


def _is_repeating(history: list, n: int) -> bool:
    """최근 n개 액션이 모두 동일한지 검사합니다."""
    if len(history) < n:
        return False
    last_n = history[-n:]
    actions = set(
        (a.get("action"), json.dumps(a.get("args", {}), sort_keys=True))
        for a in last_n if isinstance(a, dict)
    )
    return len(actions) == 1


def _has_job_url(job: dict) -> bool:
    return bool((job.get("url") or job.get("URL") or job.get("공고url") or "").strip())


def _should_skip_job_update_without_detail_url(new_data: dict, current_url: str) -> bool:
    policy = _persistence_policy_for_url(current_url)
    if not policy.get("require_detail_url_for_job_update") or _looks_like_job_detail_url(current_url):
        return False

    incoming_jobs = _job_list_value(new_data)
    if isinstance(incoming_jobs, dict):
        incoming_jobs = [incoming_jobs]
    if not isinstance(incoming_jobs, list):
        return False

    return any(isinstance(job, dict) and not _has_job_url(job) for job in incoming_jobs)


def _job_identity(job: dict) -> tuple:
    url = (job.get("url") or job.get("URL") or job.get("공고url") or "").strip()
    company = (job.get("회사명") or job.get("company_name") or "").strip()
    position = (job.get("직무명") or job.get("position") or "").strip()
    return (url, company, position)


def _merge_value(old: Any, new: Any) -> Any:
    if new in (None, "", [], {}):
        return old
    if isinstance(old, list) or isinstance(new, list):
        old_items = old if isinstance(old, list) else ([old] if old not in (None, "") else [])
        new_items = new if isinstance(new, list) else [new]
        merged = []
        seen = set()
        for item in old_items + new_items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
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


def _merge_extracted_info(current_jd: dict, new_data: dict, current_url: str = "") -> tuple[dict, dict]:
    merged = dict(current_jd)
    summary = {"incoming_jobs": 0, "total_jobs": 0, "fields": []}

    incoming_jobs = _job_list_value(new_data)
    if isinstance(incoming_jobs, dict):
        incoming_jobs = [incoming_jobs]

    if isinstance(incoming_jobs, list):
        existing_jobs = _job_list_value(merged)
        if not isinstance(existing_jobs, list):
            existing_jobs = []

        for incoming in incoming_jobs:
            if not isinstance(incoming, dict):
                continue
            job = dict(incoming)
            if _looks_like_job_detail_url(current_url) and not (job.get("url") or job.get("URL") or job.get("공고url")):
                job["url"] = current_url

            summary["incoming_jobs"] += 1
            summary["fields"].extend(job.keys())
            identity = _job_identity(job)
            match_index = None
            for idx, existing in enumerate(existing_jobs):
                if not isinstance(existing, dict):
                    continue
                if _job_identity(existing) == identity and any(identity):
                    match_index = idx
                    break
                if identity[0] and identity[0] in _job_identity(existing):
                    match_index = idx
                    break
                if identity[1:] == _job_identity(existing)[1:] and all(identity[1:]):
                    match_index = idx
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
    existing_jobs = _job_list_value(merged)
    if not summary["total_jobs"] and isinstance(existing_jobs, list):
        summary["total_jobs"] = len(existing_jobs)
    return merged, summary


def _extracted_job_count(extracted_jd: dict) -> int:
    jobs = _job_list_value(extracted_jd)
    if isinstance(jobs, list):
        return len([job for job in jobs if isinstance(job, dict) and job])
    if isinstance(jobs, dict):
        return 1
    return 1 if extracted_jd else 0


def _target_count_from_state(state: GraphState) -> int:
    params = state.get("recipe_params", {}) or {}
    try:
        return max(0, int(params.get("target_count") or 0))
    except (TypeError, ValueError):
        return 0


def _count_mode_from_state(state: GraphState) -> str:
    params = state.get("recipe_params", {}) or {}
    raw = params.get("count_mode") or ""
    return str(getattr(raw, "value", raw)).strip().lower()


def _auto_finish_on_target_enabled() -> bool:
    raw = os.getenv("VISION_AUTO_FINISH_ON_TARGET", "1")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _is_detail_update(args: dict[str, Any]) -> bool:
    role = str(args.get("page_role") or "").strip().lower()
    return role in {"job_detail", "detail", "posting_detail"}


def _infer_current_page_role(current_url: str, markers: list[dict[str, Any]]) -> str:
    """현재 화면의 replay 적용 범위를 보수적으로 분류한다."""

    return _infer_site_page_role(
        current_url,
        [
            marker.get("text")
            for marker in markers or []
            if isinstance(marker, dict)
        ],
    )


def _detail_key_from_state(state: GraphState) -> str:
    """같은 URL의 패널형 상세 화면도 공고별로 OCR 버퍼를 분리한다."""

    card = dict(state.get("active_result_card", {}) or {})
    queue_id = str(card.get("queue_id") or "").strip()
    if queue_id:
        return queue_id
    company = str(card.get("company") or "").strip()
    title = str(card.get("title") or "").strip()
    return "|".join(part for part in (company, title) if part)


def _sensitive_action_reason(state: GraphState, action_name: str, args: dict[str, Any]) -> str:
    if action_name in {"close_browser", "close_current_tab", "switch_tab", "go_back", "scroll"}:
        return ""
    if args.get("needs_user_confirmation") is True:
        return "tool_args_requested_user_confirmation"
    if str(args.get("risk_level") or "").strip().lower() == "sensitive":
        return "tool_args_marked_sensitive"
    return ""


def _compact_action_args(action_name: str, args: dict) -> dict:
    if action_name == "finish_detail_reading":
        return {
            "page_role": args.get("page_role", "job_detail"),
            "detail_complete": args.get("detail_complete", True),
            "reason": _clip_prompt_text(args.get("reason", ""), 120),
        }
    if action_name == "set_result_card_queue":
        cards = _result_card_entries_from_args(args if isinstance(args, dict) else {})
        titles = []
        for card in cards:
            label = _queue_card_label(card)
            if label:
                titles.append(label)
        return {"cards": len(cards), "titles": titles[:5]}
    if action_name != "update_extracted_info":
        return {
            key: value
            for key, value in args.items()
            if not str(key).startswith("_")
        }
    try:
        data = json.loads(args.get("data_json", "{}"))
    except Exception:
        return {"data_json": "<invalid json>"}
    jobs = _job_list_value(data)
    if isinstance(jobs, dict):
        jobs = [jobs]
    fields = []
    if isinstance(jobs, list):
        for job in jobs:
            if isinstance(job, dict):
                fields.extend(job.keys())
    fields.extend(k for k in data.keys() if k not in JOB_LIST_KEYS)
    return {
        "incoming_jobs": len(jobs) if isinstance(jobs, list) else 0,
        "fields": sorted({str(field) for field in fields}),
        "payload_chars": len(args.get("data_json", "")),
    }




def _action_target_metadata(state: GraphState, action_name: str, args: dict) -> dict | None:
    if action_name not in {"click_marker", "type_in_marker", "scroll"} or args.get("marker_id") is None:
        return None
    marker = _marker_by_id(state.get("current_markers", []), args.get("marker_id"))
    if not marker:
        return {"marker_id": args.get("marker_id"), "missing": True}
    bbox = marker.get("bbox", [])
    center = None
    if isinstance(bbox, list) and len(bbox) == 4:
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
    metadata = {
        "marker_id": marker.get("id"),
        "text": marker.get("text", ""),
        "bbox": bbox,
        "center": center,
    }
    signature = dict(state.get("screen_signature", {}) or {})
    size = signature.get("size") or []
    if isinstance(size, list) and len(size) == 2 and isinstance(bbox, list) and len(bbox) == 4:
        try:
            metadata["bbox_ratio"] = bbox_to_ratio(bbox, size)
            metadata["center_ratio"] = center_ratio_from_bbox(bbox, size)
        except Exception:
            pass
    target_label = args.get("target_label") or args.get("semantic_label")
    if target_label:
        metadata["target_label"] = target_label
    return metadata

def _state_snapshot_for_action(state: GraphState, current_url: str) -> dict:
    recent_images = state.get("recent_images", []) or []
    screenshot = str(recent_images[-1]) if recent_images else ""
    return {
        "url": current_url or state.get("current_url", "") or "",
        "screenshot": screenshot,
        "marked_image": state.get("marked_image", "") or "",
        "screen_signature": dict(state.get("screen_signature", {}) or {}),
    }


def _is_open_browser_noop(action: dict) -> bool:
    if action.get("action") != "open_browser":
        return False
    result = action.get("result")
    return isinstance(result, dict) and result.get("opened") is False


def _latest_no_effect_transition(state: GraphState) -> dict[str, Any]:
    """현재 화면에서 효과가 없다고 확인된 가장 최근 물리 행동을 반환합니다."""
    observations = state.get("transition_observations", []) or []
    if not observations:
        return {}
    latest = observations[-1]
    if not isinstance(latest, dict) or latest.get("status") != "unknown":
        return {}
    if latest.get("reason") not in {"reflex_no_screen_change", "no_screen_change"}:
        return {}
    recent_images = state.get("recent_images", []) or []
    latest_screen = str(recent_images[-1]) if recent_images else ""
    observed_screen = str(latest.get("screenshot") or "")
    if latest_screen and observed_screen and latest_screen != observed_screen:
        return {}
    return latest


def _recent_forbidden_actions(action_history: list[dict], limit: int = 6) -> list[dict]:
    forbidden = []
    seen = set()

    for action in reversed(action_history or []):
        if not isinstance(action, dict):
            continue

        reason = action.get("reason", "") or ""
        forbidden_reason = ""
        if reason in {
            "unsafe_ui_action_chain",
            "same_screen_no_effect_action_blocked",
            IMPLAUSIBLE_TEXT_INPUT_TARGET,
        }:
            forbidden_reason = reason
        elif _is_open_browser_noop(action):
            forbidden_reason = action.get("result", {}).get("reason", "open_browser_no_screen_change")
        else:
            continue

        action_name = action.get("action", "")
        args = action.get("args", {}) or {}
        key = (
            action_name,
            json.dumps(args, ensure_ascii=False, sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        forbidden.append({
            "action": action_name,
            "args": args,
            "reason": forbidden_reason,
        })
        if len(forbidden) >= limit:
            break

    return forbidden


def _build_forbidden_action_context(action_history: list[dict]) -> str:
    forbidden = _recent_forbidden_actions(action_history)
    if not forbidden:
        return ""

    lines = [
        "[Execution constraints for the current screen]",
        "Do not call these exact tool+args again; the executor recently skipped them:",
    ]
    for item in forbidden:
        lines.append(
            "- "
            + item["action"]
            + " "
            + json.dumps(item["args"], ensure_ascii=False, sort_keys=True)
            + f" ({item['reason']})"
        )
    lines.append(
        "Choose a different visible marker or a different atomic navigation tool instead. "
        "If go_back had no effect on a detail page opened from results, consider close_current_tab."
    )
    return "\n".join(lines)

def _chain_boundary_reached(action_name: str) -> bool:
    return action_name in {
        "click_marker",
        "scroll",
        "press_key",
        "open_browser",
        "close_browser",
        "close_current_tab",
        "switch_tab",
        "go_back",
    }


def _is_allowed_same_screen_ui_chain(previous_ui_action: str | None, action_name: str) -> bool:
    return previous_ui_action == "type_in_marker" and action_name == "press_key"










def _build_ui_context(
    markers: list[dict],
    current_url: str = "",
    page_role: str = "",
) -> str:
    marker_texts = [marker.get("text") for marker in markers if isinstance(marker, dict)]
    if (
        current_url
        and _is_job_detail_context(
            current_url,
            page_role=page_role,
            marker_texts=marker_texts,
        )
        and _env_enabled("VISION_DETAIL_SECTION_CONTEXT_ENABLED", True)
    ):
        section_context = _build_detail_section_context(markers)
        if section_context:
            return section_context

    try:
        text_limit = int(os.getenv("VISION_UI_TEXT_MARKER_LIMIT", "90"))
        icon_limit = int(os.getenv("VISION_UI_ICON_MARKER_LIMIT", "45"))
    except ValueError:
        text_limit = 90
        icon_limit = 45
    text_markers = []
    icon_markers = []
    for marker in markers:
        if _is_icon_marker(marker):
            icon_markers.append(marker)
        else:
            text_markers.append(marker)

    text_markers = sorted(text_markers, key=_marker_prompt_rank)
    icon_markers = sorted(icon_markers, key=_marker_prompt_rank)
    shown_text_markers = text_markers[:text_limit]
    shown_icon_markers = icon_markers[:icon_limit]

    parts = []
    if shown_text_markers:
        parts.append(
            "식별된 텍스트 요소:\n"
            + "\n".join(f"[id: {m['id']}] {m.get('text', '')}" for m in shown_text_markers)
        )
    if shown_icon_markers:
        parts.append(f"기타 아이콘/버튼 마커 ID 목록: {[m['id'] for m in shown_icon_markers]}")
    omitted_text = max(0, len(text_markers) - len(shown_text_markers))
    omitted_icon = max(0, len(icon_markers) - len(shown_icon_markers))
    if omitted_text or omitted_icon:
        parts.append(f"프롬프트 경량화를 위해 생략된 마커: 텍스트 {omitted_text}개, 아이콘 {omitted_icon}개")
    return "\n".join(parts) if parts else "발견된 UI 마커 없음"


def _safety_page_role_contract() -> str:
    return (
        "\n\n[Safety and page-role contract]\n"
        "- For every UI tool call, include page_role when you can infer it: home, search, list, detail, form, popup, error, or unknown.\n"
        "- Include risk_level: safe_read, safe_navigation, or sensitive.\n"
        "- Set needs_user_confirmation=true before login, password/authentication, personal data, agreement/terms, application/submission, payment, transfer, account, finance, or legal-effect steps. The executor will stop and ask the user.\n"
        "- For public job collection, do not attempt login, signup, authentication, or account switching unless the user explicitly asked for it. If such a screen appears, leave that flow and return to a public search/list/home surface. Use neutral action reasons such as 'return to public search surface' instead of describing a login/signup action.\n"
        "- Unknown or newly released tasks should be researched and narrowed before execution. Do not try random branches first.\n"
        "- On detail pages, your main judgment is whether enough information has been read. If detail OCR buffering is active, do not call update_extracted_info for intermediate extraction; scroll, click a clearly relevant reveal/details control, or call finish_detail_reading(page_role=\"job_detail\", detail_complete=true) when the current posting is sufficiently read.\n"
    )


def _collected_job_count(extracted_jd: Any) -> int:
    """현재 누적 데이터에서 수집된 공고 개수를 계산한다."""
    if not isinstance(extracted_jd, dict) or not extracted_jd:
        return 0
    for value in extracted_jd.values():
        if isinstance(value, list) and any(isinstance(item, dict) and item for item in value):
            return sum(1 for item in value if isinstance(item, dict) and item)
    return 1


def _clip_prompt_text(value: Any, max_chars: int = 160) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _first_nonempty_field(data: dict, aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _compact_prompt_value(value: Any, max_chars: int = 140) -> Any:
    if isinstance(value, list):
        compacted = []
        for item in value:
            if item in (None, "", [], {}):
                continue
            compacted.append(_compact_prompt_value(item, max_chars=100))
            if len(compacted) >= 3:
                break
        return compacted
    if isinstance(value, dict):
        compacted = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 4:
                break
            if item in (None, "", [], {}):
                continue
            compacted[str(key)] = _compact_prompt_value(item, max_chars=80)
        return compacted
    return _clip_prompt_text(value, max_chars=max_chars)


_JOB_FIELD_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("회사명", ("회사명", "company_name", "company")),
    ("직무명", ("직무명", "position", "title", "job_title")),
    ("url", ("url", "공고URL", "link")),
    ("주요업무", ("주요업무", "main_tasks", "responsibilities")),
    ("자격요건", ("자격요건", "requirements", "qualifications")),
    ("우대사항", ("우대사항", "preferred", "preferred_qualifications")),
    ("혜택", ("혜택", "혜택 및 복지", "복리후생", "benefits")),
)


def _job_display_label(job: dict) -> str:
    company = _first_nonempty_field(job, ("회사명", "company_name", "company"))
    position = _first_nonempty_field(job, ("직무명", "position", "title", "job_title"))
    if company and position:
        return _clip_prompt_text(f"{company} - {position}", 120)
    return _clip_prompt_text(position or company or job.get("url") or "", 120)


def _job_summary_for_prompt(job: dict) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    present_fields: list[str] = []
    missing_fields: list[str] = []
    for label, aliases in _JOB_FIELD_ALIASES:
        value = _first_nonempty_field(job, aliases)
        if value in (None, "", [], {}):
            missing_fields.append(label)
            continue
        present_fields.append(label)
        if label in {"회사명", "직무명", "url"}:
            summary[label] = _compact_prompt_value(value, max_chars=140)
        elif label in {"주요업무", "자격요건", "우대사항", "혜택"}:
            summary[label] = _compact_prompt_value(value, max_chars=120)
    if present_fields:
        summary["채워진필드"] = present_fields
    if missing_fields:
        summary["누락필드"] = missing_fields
    return summary


def _job_items_for_prompt(extracted_jd: Any) -> list[dict]:
    if not isinstance(extracted_jd, dict) or not extracted_jd:
        return []
    jobs = _job_list_value(extracted_jd)
    if isinstance(jobs, dict):
        jobs = [jobs]
    if isinstance(jobs, list):
        return [job for job in jobs if isinstance(job, dict) and job]
    return [extracted_jd] if extracted_jd else []


def _current_job_for_prompt(jobs: list[dict], current_url: str) -> dict | None:
    current_url = str(current_url or "").strip()
    if current_url:
        for job in reversed(jobs):
            if str(job.get("url") or "").strip() == current_url:
                return job
    return jobs[-1] if jobs else None


def _compact_extracted_context(extracted_jd: Any, current_url: str) -> str:
    jobs = _job_items_for_prompt(extracted_jd)
    if not jobs:
        return "수집 데이터 요약:\n- 수집된 공고 없음\n\n"

    current_job = _current_job_for_prompt(jobs, current_url)
    recent_labels = [_job_display_label(job) for job in jobs[-3:]]
    recent_labels = [label for label in recent_labels if label]
    lines = [
        "수집 데이터 요약:",
        f"- 수집 공고 수: {len(jobs)}",
    ]
    if recent_labels:
        lines.append(f"- 최근 공고: {json.dumps(recent_labels, ensure_ascii=False, separators=(',', ':'))}")
    if current_job:
        summary = _job_summary_for_prompt(current_job)
        lines.append(
            "- 현재/최근 공고 핵심 필드: "
            + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n".join(lines) + "\n\n"


def _compact_plan_context(plan: list, current_plan_step: int) -> str:
    if not plan:
        return ""
    safe_plan = [str(step) for step in plan]
    current_idx = min(max(int(current_plan_step or 0), 0), len(safe_plan) - 1)
    lines = [
        "계획 요약:",
        f"- 전체 단계 수: {len(safe_plan)}",
        f"- 현재 단계({current_idx + 1}): {_clip_prompt_text(safe_plan[current_idx], 180)}",
    ]
    if current_idx + 1 < len(safe_plan):
        lines.append(f"- 다음 단계({current_idx + 2}): {_clip_prompt_text(safe_plan[current_idx + 1], 180)}")
    return "\n".join(lines) + "\n\n"


def _compact_recent_action(action: dict) -> dict[str, Any]:
    action_name = str(action.get("action") or "")
    args = action.get("args") or {}
    compact_args = _compact_action_args(action_name, args) if isinstance(args, dict) else {}
    keep_keys = (
        "marker_id",
        "target_label",
        "target_component",
        "target_role",
        "text",
        "key",
        "direction",
        "url",
        "page_role",
    )
    if action_name == "update_extracted_info":
        shown_args = compact_args
    else:
        shown_args = {key: compact_args.get(key) for key in keep_keys if compact_args.get(key) not in (None, "", [], {})}
    item: dict[str, Any] = {
        "action": action_name,
        "status": action.get("status", ""),
        "args": shown_args,
    }
    reason = action.get("reason")
    if reason and action.get("status") != "success":
        item["reason"] = _clip_prompt_text(reason, 120)
    return item


def _compact_recent_actions_context(action_history: list[dict]) -> str:
    try:
        limit = max(1, int(os.getenv("VISION_REASONING_ACTION_HISTORY_LIMIT", "2")))
    except ValueError:
        limit = 2
    recent = [
        _compact_recent_action(action)
        for action in (action_history or [])[-limit:]
        if isinstance(action, dict)
    ]
    if not recent:
        return "최근 행동 요약: []\n\n"
    return (
        "최근 행동 요약:\n"
        + json.dumps(recent, ensure_ascii=False, separators=(",", ":"))
        + "\n\n"
    )


def _compact_result_card_queue_context(state: GraphState) -> str:
    queue = [item for item in (state.get("result_card_queue", []) or []) if isinstance(item, dict)]
    if not queue:
        return "공고 카드 큐: []\n\n"
    compact = [
        {
            "queue_id": item.get("queue_id", ""),
            "status": item.get("status", "pending"),
            "title": item.get("title", ""),
            "company": item.get("company", ""),
        }
        for item in queue
    ]
    pending_count = len(_pending_result_cards(queue))
    return (
        "공고 카드 큐:\n"
        f"- pending_count: {pending_count}\n"
        f"- cards: {json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}\n"
        "- 큐가 있으면 상세 수집 완료 후 다음 카드 선택은 executor가 처리합니다. 같은 목록에서 다음 카드를 다시 고르지 마십시오.\n\n"
    )


def _compact_result_availability_context(state: GraphState) -> str:
    availability = dict(state.get("result_availability", {}) or {})
    if not availability:
        return "검색 결과 개수 힌트: 없음\n\n"
    return (
        "검색 결과 개수 힌트:\n"
        f"- 현재 검색 조건의 전체 결과 수: {availability.get('available_result_count')}\n"
        f"- 화면 근거: {availability.get('count_evidence') or '(없음)'}\n"
        f"- 판단 신뢰도: {availability.get('count_confidence', 0)}\n"
        "- 이 숫자는 현재 검색어와 필터 조건의 결과 수이지 사이트 전체의 최대치가 아닙니다.\n"
        "- 현재 조건의 결과를 모두 수집했으면 같은 목록을 더 스크롤하지 마십시오. 목표 수가 남았다면 사용자 의도를 "
        "유지하는 범위에서 검색어 또는 필터를 넓힐지 판단하고, 적절한 확장 방법이 없으면 수집 건수와 부족분을 밝히며 "
        "finish_task로 부분 완료하십시오.\n\n"
    )


def _reasoning_image_base64(state: GraphState) -> str:
    marked_image_path = state.get("marked_image")
    if not marked_image_path or not os.path.exists(marked_image_path):
        return ""
    try:
        from pathlib import Path

        from agent.utils.image_utils import image_to_base64_jpeg

        try:
            max_dim = int(os.getenv("VISION_REASONING_IMAGE_MAX_DIM", "768"))
            quality = int(os.getenv("VISION_REASONING_IMAGE_QUALITY", "60"))
        except ValueError:
            max_dim = 768
            quality = 60
        return image_to_base64_jpeg(
            Path(marked_image_path),
            max_dim=max_dim,
            quality=quality,
            fast=True,
        )
    except Exception as img_err:
        logger.warning("Failed to read/resize marked_image for reasoning node", error=str(img_err))
        return ""


def _build_reasoning_messages(
    state: GraphState,
    loop_warning: str,
    selector_trace: dict[str, Any] | None = None,
) -> list:
    """
    reasoning_node용 LLM 메시지 리스트를 조립합니다.
    마킹 이미지가 있으면 멀티모달, 없으면 텍스트 전용 메시지를 반환합니다.
    """
    plan = state.get("plan", [])
    current_plan_step = state.get("current_plan_step", 0)
    plan_context = _compact_plan_context(plan, current_plan_step)

    system_prompt_text = COMMANDER_SYSTEM_PROMPT.format(goal=state.get("goal", "")) + _safety_page_role_contract()
    extracted_jd = state.get("extracted_jd", {})
    ui_context = state.get("ui_context", "")
    current_url = state.get("current_url", "")
    action_history = state.get("action_history", [])
    recipe_params = dict(state.get("recipe_params", {}) or {})
    target_count = int(recipe_params.get("target_count") or 0)
    collected_count = _collected_job_count(extracted_jd)
    visited_cards: list[str] = []
    for action in action_history:
        if not isinstance(action, dict) or action.get("status") != "success":
            continue
        args = action.get("args") or {}
        target = action.get("target") or {}
        component = args.get("target_component") or target.get("component") or ""
        if component != "job_card_title":
            continue
        label = args.get("target_label") or target.get("target_label") or target.get("text") or ""
        label = str(label).strip()
        if label and label not in visited_cards:
            visited_cards.append(label)
    collection_context = (
        "수집 순회 상태:\n"
        f"- 목표 공고 수: {target_count if target_count > 0 else '(지정 안 됨)'}\n"
        f"- 현재 수집 공고 수: {collected_count}\n"
        f"- 이미 방문한 공고 카드: {json.dumps(visited_cards, ensure_ascii=False)}\n"
        "- 검색 결과의 공고 제목은 실행마다 달라지는 동적 대상입니다. 기록된 과거 공고명을 재사용하지 말고, "
        "현재 화면에서 보이는 미방문 공고 제목을 선택하십시오.\n"
        "- 목표 수를 채웠으면 목록으로 돌아가거나 같은 카드를 다시 열지 말고 finish_task를 호출하십시오.\n\n"
    )
    transition_context = ""
    if state.get("transition_status"):
        latest_transition = _latest_no_effect_transition(state)
        transition_context = (
            "직전 화면 전환 검증:\n"
            f"- status: {state.get('transition_status')}\n"
            f"- outcome: {state.get('transition_outcome') or '(없음)'}\n"
            f"- source: {state.get('transition_source') or '(없음)'}\n"
        )
        if latest_transition:
            transition_context += (
                f"- 효과가 없었던 행동: {latest_transition.get('action') or '(없음)'}\n"
                f"- 판정 이유: {latest_transition.get('reason') or '(없음)'}\n"
                "- 같은 행동을 반복하지 마십시오. 상세 공고가 별도 탭에 열렸을 가능성이 있으면 "
                "close_current_tab을 사용하고, 이전 탭을 유지해야 하면 switch_tab을 사용하십시오.\n"
            )
        transition_context += "\n"
    result_refinement_context = ""
    selector_trace = selector_trace or {}
    if selector_trace.get("reason") == "result_refinement_needed":
        refinement_reason = str(selector_trace.get("refinement_reason") or "").strip()
        result_refinement_context = (
            "검색 결과 정제 필요:\n"
            "- 현재 화면에서 검색어와 직접 일치하는 공고가 목표 수보다 부족합니다. 비슷한 직무로 개수를 채우지 마십시오.\n"
            "- 검색어를 더 정확하게 표현하는 화면 필터가 있으면 적용하고, 없으면 다음 정확한 후보를 찾도록 스크롤하십시오.\n"
            f"- 카드 선택기 판단: {refinement_reason or '(구체적 이유 없음)'}\n\n"
        )
    forbidden_action_context = _build_forbidden_action_context(action_history)
    if forbidden_action_context:
        forbidden_action_context += "\n\n"

    human_prompt_text = (
        f"{plan_context}"
        f"{_compact_extracted_context(extracted_jd, current_url)}"
        f"현재 브라우저 URL:\n{current_url or '(확인 안 됨)'}\n\n"
        f"{_site_runtime_guidance(current_url, state.get('current_page_role', ''))}"
        f"{collection_context}"
        f"{_compact_result_availability_context(state)}"
        f"{_compact_result_card_queue_context(state)}"
        f"{_compact_detail_ocr_buffer_context(state, current_url)}"
        f"{transition_context}"
        f"{result_refinement_context}"
        f"현재 화면 상태 (UI 마커):\n{ui_context + loop_warning}\n\n"
        f"{forbidden_action_context}"
        f"{_compact_recent_actions_context(action_history)}"
        f"다음 행동을 결정하세요. 상세 페이지에서 OCR 버퍼가 활성화되어 있으면 중간 정보 추출 대신 finish_detail_reading으로 읽기 종료를 알리고, "
        f"그 외 화면에서 새로운 정보가 식별되었다면 update_extracted_info를 먼저 부르고, "
        f"계획 단계 전환이 일어났다면 update_plan_progress를 함께 체이닝 호출하여 계획 진행률을 반영하십시오."
    )

    # 마킹 이미지가 있으면 멀티모달 메시지
    base64_image = _reasoning_image_base64(state)

    if base64_image:
        logger.info("Invoking reasoning node with multimodal SoM marked image...")
        return [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=[
                {"type": "text", "text": human_prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ])
        ]
    else:
        logger.info("Invoking reasoning node with text-only prompts...")
        return [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=human_prompt_text)
        ]


def reasoning_node(state: GraphState) -> Dict[str, Any]:
    """Gemini Flash를 호출하여 다음 행동을 결정합니다."""
    from agent.application.run_context import raise_if_cancelled

    raise_if_cancelled()
    start_time = time.perf_counter()
    logger.info("Executing Reasoning Node")

    # 루프 감지
    action_history = state.get("action_history", [])
    loop_warning = ""
    error_increment = 0

    if _is_repeating(action_history, 3):
        repeated = action_history[-1]
        logger.warning(f"Loop detected! Repeated action: {repeated.get('action')} with args: {repeated.get('args')}")
        loop_warning = (
            f"\n\n[경고: 무한 루프 감지됨] 당신은 직전 3회 동안 동일한 행동"
            f"({repeated.get('action')}: {repeated.get('args')})을 반복했습니다. "
            f"절대 동일한 행동(동일 마커 클릭 등)을 다시 수행하지 마십시오. "
            f"새로운 마커를 클릭하거나, 스크롤을 하거나, 다른 방식으로 목표를 해결해야 합니다."
        )

    if _is_repeating(action_history, 4):
        logger.error("Persistent loop detected. Increasing error count to terminate.")
        error_increment = 1

    selector_response, selector_trace = _select_result_cards(state)
    if selector_response is not None:
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Reasoning Node completed",
            component="reasoning",
            duration_sec=round(elapsed, 6),
            reasoning_mode="card_selection",
        )
        result = {
            "last_action_result": selector_response,
            "result_card_selector_trace": selector_trace,
            "reflex_hit": False,
            "reflex_trace": {"hit": False, "source": "card_selector"},
            "reflex_transition_contracts": {},
            "step_durations": [
                {"node": "reasoning", "duration": elapsed, "reasoning_mode": "card_selection"}
            ],
        }
        if error_increment > 0:
            result["error_count"] = state.get("error_count", 0) + error_increment
        return result

    # 메시지 조립 + LLM 호출
    from agent.application.run_context import invoke_with_metrics

    reasoning_mode = "general_after_card_selector" if selector_trace.get("attempted") else "general"
    response = invoke_with_metrics(
        _get_ui_llm_with_tools(_allowed_tool_names_for_state(state)),
        _build_reasoning_messages(state, loop_warning, selector_trace),
        "vision_reasoning",
    )

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Reasoning Node completed",
        component="reasoning",
        duration_sec=round(elapsed, 6),
        reasoning_mode=reasoning_mode,
    )

    result = {
        "last_action_result": response,
        "result_card_selector_trace": selector_trace,
        "reflex_hit": False,
        "reflex_trace": {"hit": False, "source": "reasoning"},
        "reflex_transition_contracts": {},
        "step_durations": [
            {"node": "reasoning", "duration": elapsed, "reasoning_mode": reasoning_mode}
        ]
    }
    if error_increment > 0:
        result["error_count"] = state.get("error_count", 0) + error_increment

    return result




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
    current_plan: list,
    current_plan_step: int,
    current_url: str = "",
    state: GraphState | None = None,
) -> Tuple[dict, dict, list, int]:
    """그래프 상태 변경 도구를 실행하고 (result, jd, plan, step)을 반환합니다."""
    if action_name == "update_plan_progress":
        current_plan_step = args["current_step"]
        if args.get("plan") is not None:
            current_plan = args["plan"]
        result = {
            "action": "update_plan_progress",
            "status": "success",
            "result": f"Plan progress updated. Current step index: {current_plan_step}",
        }
    elif action_name == "update_extracted_info":
        try:
            new_data = json.loads(args["data_json"])
            if _should_skip_job_update_without_detail_url(new_data, current_url):
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
            else:
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
                }
        except Exception as e:
            result = {
                "action": "finish_detail_reading",
                "status": "error",
                "result": f"Failed to extract detail OCR buffer: {e}",
            }
    elif action_name == "set_result_card_queue":
        queue, memory = _normalize_result_card_queue(args, state or {}, current_url)
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
            "_result_card_queue": queue,
            "_result_page_memory": memory,
            "_result_availability": availability,
        }
    else:
        raise ValueError(f"Unknown state action: {action_name}")
    return result, current_jd, current_plan, current_plan_step


def action_node(state: GraphState) -> Dict[str, Any]:
    """Reasoning Node가 선택한 도구(들)를 순차적으로 실행(Action Chaining)합니다."""
    from agent.application.run_context import raise_if_cancelled

    raise_if_cancelled()
    started_monotonic = time.perf_counter()
    logger.info("Executing Action Node (with potential Action Chaining)")

    try:
        from agent.recipe.record import record_ui_step, commit_if_finished
    except Exception:
        record_ui_step = commit_if_finished = None
    try:
        from agent.recipe.feedback import record_action_episode
    except Exception:
        record_action_episode = None
    recorded_steps: list = []
    feedback_episodes: list = []
    prior_recorded_steps = list(state.get("recorded_steps", []) or [])

    ai_msg: AIMessage = state.get("last_action_result")

    if ai_msg and hasattr(ai_msg, "content") and ai_msg.content:
        logger.info(f"LLM Thoughts: {ai_msg.content}")

    if not ai_msg or not hasattr(ai_msg, "tool_calls") or not ai_msg.tool_calls:
        logger.warning("LLM did not return a tool call.")
        elapsed = time.perf_counter() - started_monotonic
        return {
            "action_history": [{"action": "none", "status": "error", "error": "No tool call", "args": {}}],
            "step_durations": [{"node": "action", "duration": elapsed}]
        }

    prior_actions = list(state.get("action_history", []) or [])
    new_actions = []
    current_jd        = dict(state.get("extracted_jd", {}))
    is_finished       = state.get("is_finished", False)
    collected_data    = list(state.get("collected_data", []))
    error_count       = state.get("error_count", 0)
    step_durations    = []
    current_plan_step = state.get("current_plan_step", 0)
    current_plan      = list(state.get("plan", []))
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
    screen_changed    = False
    chain_boundary    = False
    previous_ui_action: str | None = None
    pending_transition: dict[str, Any] = {}
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
            "args": _compact_action_args(action_name, args),
            "page_role": args.get("page_role") or state.get("current_page_role", ""),
            "target_role": args.get("target_role") or args.get("target_role_candidate") or "",
            "component": args.get("target_component") or args.get("component_candidate") or "",
            "expected_after": args.get("expected_after") or "",
        }
        if tool_call_id:
            step["tool_call_id"] = tool_call_id
        if state.get("reflex_hit"):
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
    ) -> dict:
        result["args"] = _compact_action_args(requested_action, action_args)
        result["before_url"] = before_snapshot.get("url", "")
        result["before_screenshot"] = before_snapshot.get("screenshot", "")
        result["before_marked_image"] = before_snapshot.get("marked_image", "")
        result["screen_change_expected"] = screen_change_expected
        target = _action_target_metadata(state, requested_action, action_args)
        if target:
            result["target"] = target
        if state.get("reflex_hit"):
            trace = dict(state.get("reflex_trace", {}) or {})
            if trace:
                result["reflex_hit"] = True
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
        if record_action_episode:
            record_action_episode(
                feedback_episodes,
                state,
                ai_msg,
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
        step_durations.append({"node": f"action ({action_name})", "duration": step_elapsed})
        logger.warning(message, action=action_name, reason=reason)

    def append_policy_ui_action(action_name: str, args: dict, reason: str) -> dict:
        nonlocal current_url_stale, screen_changed, pending_transition
        step_start = time.perf_counter()
        before_snapshot = _state_snapshot_for_action(state, current_url)
        action_seq = next_action_seq()
        result = _dispatch_ui(action_name, args, get_bbox)
        action_changed_screen = action_name in SCREEN_CHANGING_ACTIONS
        current_url_stale = current_url_stale or action_name in URL_STALE_ACTIONS
        screen_changed = screen_changed or action_changed_screen
        if action_changed_screen:
            set_pending_transition(action_seq, action_name, args, None, "page_policy")
        enriched = enrich_result(result, action_name, args, before_snapshot, action_changed_screen)
        enriched["policy_action"] = True
        enriched["policy_reason"] = reason
        new_actions.append(enriched)
        if record_ui_step:
            record_ui_step(recorded_steps, state, action_name, args, action_seq)
        if record_action_episode:
            record_action_episode(
                feedback_episodes,
                state,
                ai_msg,
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
            )
        step_elapsed = time.perf_counter() - step_start
        step_durations.append({"node": f"action ({action_name})", "duration": step_elapsed})
        logger.info("Page policy action executed", action=action_name, reason=reason, duration=f"{step_elapsed:.2f}s")
        return enriched

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
    STATE_ACTIONS = {"update_plan_progress", "update_extracted_info", "finish_detail_reading", "set_result_card_queue"}

    for idx, tool_call in enumerate(ai_msg.tool_calls):
        action_name = tool_call["name"]
        args        = tool_call["args"]
        if action_name == "finish_detail_reading":
            args.setdefault("page_role", "job_detail")
            args.setdefault("detail_complete", True)
        compact_args = _compact_action_args(action_name, args)

        logger.info(
            f"LLM decided to call (chained {idx+1}/{len(ai_msg.tool_calls)}): "
            f"{action_name} with args: {compact_args}"
        )
        step_start = time.perf_counter()
        before_snapshot = _state_snapshot_for_action(state, current_url)
        action_seq = next_action_seq()
        policy_ui_action: tuple[str, dict, str] | None = None

        try:
            if chain_boundary and action_name in UI_ACTIONS:
                append_guard_result(
                    action_name,
                    args,
                    before_snapshot,
                    "skipped",
                    "chain_boundary_after_screen_change",
                    "Skipped chained UI tool after a screen-changing action; next perception is required.",
                    step_start,
                )
                break
            if action_name in UI_ACTIONS:
                if (
                    action_name in {"click_marker", "type_in_marker"}
                    and not state.get("reflex_hit")
                    and not state.get("queue_replay_hit")
                    and not state.get("page_policy_hit")
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
                if no_effect_transition.get("action") == action_name:
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
                    contract = reflex_transition_contracts.get(str(tool_call.get("id") or ""))
                    transition_source = (
                        "card_queue"
                        if state.get("queue_replay_hit")
                        else ("reflex" if state.get("reflex_hit") else "autonomous")
                    )
                    transition_source = str(args.get("_transition_source") or transition_source)
                    set_pending_transition(
                        action_seq,
                        action_name,
                        args,
                        contract,
                        transition_source,
                        str(tool_call.get("id") or ""),
                    )
                if action_changed_screen and _chain_boundary_reached(action_name):
                    chain_boundary = True
                if record_ui_step:
                    record_ui_step(recorded_steps, state, action_name, args, action_seq)
                if (
                    result.get("status") == "success"
                    and action_name == "click_marker"
                    and _result_card_click_matches_queue(result_card_queue, args)
                ):
                    result_card_queue, active_result_card = _mark_result_card_active(result_card_queue, args)

            elif action_name in STATE_ACTIONS:
                result, current_jd, current_plan, current_plan_step = _dispatch_state(
                    action_name,
                    args,
                    current_jd,
                    current_plan,
                    current_plan_step,
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
                            policy_ui_action = (
                                "click_marker",
                                {
                                    "marker_id": marker_id,
                                    "queue_id": first_card.get("queue_id", ""),
                                    "target_label": first_card.get("title", ""),
                                    "target_role": "job_card",
                                    "target_component": "job_card_title",
                                    "page_role": "search",
                                    "reason": "result card queue stored; open the first pending card",
                                    "expected_after": "selected job detail page is visible",
                                },
                                "result_card_queue_first_item",
                            )
                if action_name == "finish_detail_reading":
                    detail_ocr_buffer = dict(result.pop("_detail_ocr_buffer", detail_ocr_buffer) or {})
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
                            no_effect_return = _latest_no_effect_transition(state)
                            if no_effect_return.get("action") in {
                                "go_back",
                                "close_current_tab",
                                "switch_tab",
                            }:
                                # 실패가 확인된 복귀 방식을 자동 반복하지 않고 LLM이 다른 원자 도구를 고르게 합니다.
                                result["detail_policy"] = "return_requires_reasoning"
                                result["failed_return_action"] = no_effect_return.get("action")
                            else:
                                policy_ui_action = (
                                    "go_back",
                                    {
                                        "reason": "detail page complete and more result cards remain",
                                        "expected_after": "job result list is visible",
                                    },
                                    "detail_complete_more_items",
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
                str(tool_call.get("id") or ""),
            )
            new_actions.append(enriched)
            if record_action_episode:
                record_action_episode(
                    feedback_episodes,
                    state,
                    ai_msg,
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
                )

            step_elapsed = time.perf_counter() - step_start
            step_durations.append({"node": f"action ({action_name})", "duration": step_elapsed})
            logger.info(f"Action Node [{action_name}] completed in {step_elapsed:.2f} seconds")

            if policy_ui_action and not is_finished:
                policy_action, policy_args, policy_reason = policy_ui_action
                policy_result = append_policy_ui_action(policy_action, policy_args, policy_reason)
                if (
                    policy_result.get("status") == "success"
                    and policy_action == "click_marker"
                    and _result_card_click_matches_queue(result_card_queue, policy_args)
                ):
                    result_card_queue, active_result_card = _mark_result_card_active(
                        result_card_queue,
                        policy_args,
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
                str(tool_call.get("id") or ""),
            )
            new_actions.append(enriched)
            if record_action_episode:
                record_action_episode(
                    feedback_episodes,
                    state,
                    ai_msg,
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
            step_durations.append({"node": f"action ({action_name})", "duration": step_elapsed})
            break  # 에러 발생 시 체인 중단

    if is_finished and commit_if_finished:
        commit_if_finished(prior_recorded_steps + recorded_steps, state, current_url)

    total_elapsed = time.perf_counter() - started_monotonic
    logger.info(
        "Action Node completed all chained tools",
        duration_sec=round(total_elapsed, 6),
    )

    return {
        "action_history":    new_actions,
        "extracted_jd":      current_jd,
        "is_finished":       is_finished,
        "collected_data":    collected_data,
        "error_count":       error_count,
        "step_durations":    step_durations,
        "plan":              current_plan,
        "current_plan_step": current_plan_step,
        "current_url":       current_url,
        "current_url_stale": current_url_stale,
        "current_markers":   latest_markers,
        "ui_context":        latest_ui_context,
        "marked_image":      latest_marked_image,
        "screen_signature":  dict(state.get("screen_signature", {}) or {}),
        "recent_images":     latest_recent_images,
        "last_action_screen_changed": screen_changed,
        "pending_transition": pending_transition,
        "transition_status": "",
        "transition_outcome": "",
        "transition_source": "",
        "result_card_queue": result_card_queue,
        "result_page_memory": result_page_memory,
        "result_availability": result_availability,
        "active_result_card": active_result_card,
        "queue_replay_hit": False,
        "queue_replay_trace": {},
        "page_policy_hit": False,
        "page_policy_trace": {},
        "detail_ocr_buffer": detail_ocr_buffer,
        "recorded_steps":    recorded_steps,
        "feedback_episodes": feedback_episodes,
        "pending_human_approval": pending_human_approval,
        "human_approval_request": human_approval_request,
    }

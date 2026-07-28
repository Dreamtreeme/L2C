"""작업자 화면 문맥을 구성하고 LLM 행동 요청을 선택한다."""

import json
import os
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from agent.config import get_settings
from agent.graph.action_request import (
    ActionRequest,
    action_request_from_model_response,
)
from agent.graph.state import GraphState
from agent.graph.tool_schema import ACTION_TOOL_SCHEMAS as _ACTION_TOOL_SCHEMAS
from agent.graph.worker_execution_policy import compact_action_args as _compact_action_args
from agent.graph.worker_state import (
    job_detail_key_from_state as _job_detail_key_from_state,
    return_to_job_results_for_url as _return_to_job_results_for_url,
)
from agent.prompts.commander import COMMANDER_SYSTEM_PROMPT
from agent.runtime.action_validation import IMPLAUSIBLE_TEXT_INPUT_TARGET
from agent.runtime.detail_runtime import (
    compact_job_detail_buffer_context as _compact_job_detail_buffer_context,
)
from agent.runtime.job_collection import (
    job_count as _job_count,
    job_items as _job_items,
)
from agent.runtime.job_card_queue import pending_job_cards as _pending_job_cards
from agent.runtime.job_card_selector import select_job_cards as _select_job_cards
from agent.runtime.site_context import site_runtime_guidance as _site_runtime_guidance
from agent.runtime.transition_runtime import (
    detect_two_screen_transition_cycle as _detect_two_screen_transition_cycle,
    latest_no_effect_transition as _latest_no_effect_transition,
)
from agent.utils.logger import logger

def _get_ui_llm_with_tools(allowed_tool_names: tuple[str, ...] | None = None):
    """선택된 사이트가 허용한 도구만 바인딩한 모델을 재사용한다."""
    names = tuple(
        name
        for name in (allowed_tool_names or tuple(_ACTION_TOOL_SCHEMAS))
        if name in _ACTION_TOOL_SCHEMAS
    )
    if not names:
        names = tuple(_ACTION_TOOL_SCHEMAS)
    from agent.runtime.vision_worker_runtime import current_vision_worker_runtime

    return current_vision_worker_runtime().get_ui_model_with_tools(
        names,
        _ACTION_TOOL_SCHEMAS,
    )


def _allowed_tool_names_for_state(state: GraphState) -> tuple[str, ...]:
    """현재 사이트 프로필의 허용 도구 목록을 반환한다."""

    from agent.runtime.site_context import site_profile_for_url

    profile = site_profile_for_url(str(state.get("current_url") or ""))
    configured = profile.tools.allowed_tools if profile else ()
    names = tuple(str(name) for name in configured if str(name) in _ACTION_TOOL_SCHEMAS)
    return names or tuple(_ACTION_TOOL_SCHEMAS)


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


def _is_open_browser_noop(action: dict) -> bool:
    if action.get("action") != "open_browser":
        return False
    result = action.get("result")
    return isinstance(result, dict) and result.get("opened") is False


def _repeats_no_effect_target(
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
        return (
            previous_marker is not None
            and current_marker is not None
            and previous_marker == current_marker
        )
    if action_name == "press_key":
        return str(previous_args.get("key") or "") == str(args.get("key") or "")
    if action_name == "switch_tab":
        return str(previous_args.get("direction") or "") == str(args.get("direction") or "")
    if action_name == "open_browser":
        previous_target = previous_args.get("url") or previous_args.get("site")
        current_target = args.get("url") or args.get("site")
        return bool(previous_target and previous_target == current_target)
    return action_name in {"go_back", "close_current_tab", "close_browser"}


def _recent_forbidden_actions(action_history: list[dict], limit: int = 6) -> list[dict]:
    forbidden = []
    seen = set()

    for action in reversed(action_history or []):
        if not isinstance(action, dict):
            continue

        reason = action.get("reason", "") or ""
        forbidden_reason = ""
        if reason in {
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

def _safety_page_role_contract() -> str:
    return (
        "\n\n[Safety and page-role contract]\n"
        "- For every UI tool call, include page_role when you can infer it: home, search, list, detail, form, popup, error, or unknown.\n"
        "- Include risk_level: safe_read, safe_navigation, or sensitive.\n"
        "- Set needs_user_confirmation=true before login, password/authentication, personal data, agreement/terms, application/submission, payment, transfer, account, finance, or legal-effect steps. The executor will stop and ask the user.\n"
        "- For public job collection, do not attempt login, signup, authentication, or account switching unless the user explicitly asked for it. If such a screen appears, leave that flow and return to a public search/list/home surface. Use neutral action reasons such as 'return to public search surface' instead of describing a login/signup action.\n"
        "- A marker whose OCR text is only a generic icon label has no known semantic identity. Infer it only from a clearly visible symbol; otherwise choose a nearby labeled text marker or another visible navigation path instead of inventing what its ID means.\n"
        "- Unknown or newly released tasks should be researched and narrowed before execution. Do not try random branches first.\n"
        "- On detail pages, your main judgment is whether enough information has been read. If detail OCR buffering is active, do not call update_extracted_info for intermediate extraction; scroll, click a clearly relevant reveal/details control, or call finish_detail_reading(page_role=\"job_detail\", detail_complete=true) when the current posting is sufficiently read.\n"
        "- A title and company without actual duties or qualifications may be an intermediary page. Follow a visible original-source or content-reveal control before finishing; never guess a destination URL.\n"
        "- If finish_detail_reading was rejected with detail_content_incomplete, inspect accumulated OCR and the visible page, then navigate to the original posting or reveal its content instead of repeating finish.\n"
    )


def _collected_job_count(extracted_jd: Any) -> int:
    """현재 누적 데이터에서 수집된 공고 개수를 계산한다."""
    return _job_count(extracted_jd)


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
    return _job_items(extracted_jd)


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
    limit = get_settings().vision.reasoning_action_history_limit
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


def _compact_job_card_queue_context(state: GraphState) -> str:
    queue = [item for item in (state.get("job_card_queue", []) or []) if isinstance(item, dict)]
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
    pending_count = len(_pending_job_cards(queue))
    return (
        "공고 카드 큐:\n"
        f"- pending_count: {pending_count}\n"
        f"- cards: {json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}\n"
        "- 큐가 있으면 상세 수집 완료 후 다음 카드 선택은 executor가 처리합니다. 같은 목록에서 다음 카드를 다시 고르지 마십시오.\n\n"
    )


def _compact_detail_return_context(state: GraphState, current_url: str) -> str:
    pending = _return_to_job_results_for_url(state, current_url)
    if not pending:
        return ""
    return (
        "상세 수집 완료 후 목록 복귀 대기:\n"
        "- 현재 공고의 상세 OCR 정제와 큐 완료 처리는 이미 성공했습니다.\n"
        "- 같은 공고에서 finish_detail_reading, scroll, 본문 펼치기, 정보 추출을 반복하지 마십시오.\n"
        "- 현재 탭 상태를 보고 go_back, close_current_tab, switch_tab 중 맞는 원자 행동 하나로 "
        "검색 결과 화면에 복귀하십시오.\n"
        f"- 남은 목표/대기 카드 수: {pending.get('pending_count', 0)}\n\n"
    )


def _compact_job_results_availability_context(state: GraphState) -> str:
    availability = dict(state.get("job_results_availability", {}) or {})
    if not availability:
        return "검색 결과 개수 힌트: 없음\n\n"
    return (
        "검색 결과 개수 힌트:\n"
        f"- 현재 검색 조건의 전체 결과 수: {availability.get('available_job_count')}\n"
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

        settings = get_settings().vision
        max_dim = settings.reasoning_image_max_dim
        quality = settings.reasoning_image_quality
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
    transition_result = dict(state.get("transition_result", {}) or {})
    if transition_result.get("status") not in {"", "idle", None}:
        latest_transition = _latest_no_effect_transition(state)
        transition_context = (
            "직전 화면 전환 검증:\n"
            f"- status: {transition_result.get('status')}\n"
            f"- outcome: {transition_result.get('outcome') or '(없음)'}\n"
            f"- source: {transition_result.get('source') or '(없음)'}\n"
        )
        if latest_transition:
            transition_context += (
                f"- 효과가 없었던 행동: {latest_transition.get('action') or '(없음)'}\n"
                f"- 판정 이유: {latest_transition.get('reason') or '(없음)'}\n"
                "- 같은 행동을 반복하지 마십시오. 상세 공고가 별도 탭에 열렸을 가능성이 있으면 "
                "close_current_tab을 사용하고, 이전 탭을 유지해야 하면 switch_tab을 사용하십시오.\n"
            )
        transition_context += "\n"
    job_results_refinement_context = ""
    selector_trace = selector_trace or {}
    if selector_trace.get("reason") == "job_results_refinement_needed":
        refinement_reason = str(selector_trace.get("refinement_reason") or "").strip()
        job_results_refinement_context = (
            "검색 결과 정제 필요:\n"
            "- 현재 화면에서 검색어와 직접 일치하는 공고가 목표 수보다 부족합니다. 비슷한 직무로 개수를 채우지 마십시오.\n"
            "- 검색어를 더 정확하게 표현하는 화면 필터가 있으면 적용하고, 없으면 다음 정확한 후보를 찾도록 스크롤하십시오.\n"
            f"- 카드 선택기 판단: {refinement_reason or '(구체적 이유 없음)'}\n\n"
        )
    forbidden_action_context = _build_forbidden_action_context(action_history)
    if forbidden_action_context:
        forbidden_action_context += "\n\n"

    human_prompt_text = (
        f"{_compact_extracted_context(extracted_jd, current_url)}"
        f"현재 브라우저 URL:\n{current_url or '(확인 안 됨)'}\n\n"
        f"{_site_runtime_guidance(current_url, state.get('current_page_role', ''))}"
        f"{collection_context}"
        f"{_compact_job_results_availability_context(state)}"
        f"{_compact_job_card_queue_context(state)}"
        f"{_compact_detail_return_context(state, current_url)}"
        f"{_compact_job_detail_buffer_context(state, current_url, _job_detail_key_from_state(state))}"
        f"{transition_context}"
        f"{job_results_refinement_context}"
        f"현재 화면 상태 (UI 마커):\n{ui_context + loop_warning}\n\n"
        f"{forbidden_action_context}"
        f"{_compact_recent_actions_context(action_history)}"
        f"다음 행동을 결정하세요. 상세 페이지에서 OCR 버퍼가 활성화되어 있으면 중간 정보 추출 대신 finish_detail_reading으로 읽기 종료를 알리고, "
        f"그 외 화면에서 새로운 정보가 식별되었다면 update_extracted_info를 먼저 부르십시오."
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

    transition_cycle = _detect_two_screen_transition_cycle(
        list(state.get("transition_records", []) or [])
    )
    if transition_cycle.get("detected"):
        logger.warning(
            "Two-screen transition cycle detected",
            action_cycle=transition_cycle.get("action_cycle", []),
            same_screen_distances=transition_cycle.get("same_screen_distances", []),
        )
        loop_warning += (
            "\n\n[경고: 두 화면 왕복 반복 감지] 최근 화면이 A-B-A-B 순서로 반복됐습니다. "
            f"반복된 전환 행동: {transition_cycle.get('action_cycle', [])}. "
            "이전 입력 대상이나 이동 경로가 목표에 맞지 않을 가능성이 높습니다. "
            "같은 행동 순서를 다시 실행하지 말고 현재 화면에서 의미가 다른 입력창, 버튼 또는 이동 경로를 선택하십시오."
        )

    if _is_repeating(action_history, 4):
        logger.error("Persistent loop detected. Increasing error count to terminate.")
        error_increment = 1

    selector_request, selector_trace = _select_job_cards(state)
    if selector_trace.get("reason") == "screen_loading":
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Reasoning Node skipped while result screen is loading",
            component="reasoning",
            duration_sec=round(elapsed, 6),
            reasoning_mode="loading_retry",
        )
        return {
            "pending_action": None,
            "job_card_selection_trace": selector_trace,
            "reflex_trace": {"hit": False, "source": "screen_loading"},
            "reflex_transition_contracts": {},
        }
    if selector_request is not None:
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Reasoning Node completed",
            component="reasoning",
            duration_sec=round(elapsed, 6),
            reasoning_mode="card_selection",
        )
        result = {
            "pending_action": selector_request,
            "job_card_selection_trace": selector_trace,
            "reflex_trace": {"hit": False, "source": "card_selector"},
            "reflex_transition_contracts": {},
        }
        if error_increment > 0:
            result["error_count"] = state.get("error_count", 0) + error_increment
        return result

    # 메시지 조립 + LLM 호출
    from agent.application.run_context import invoke_with_metrics

    reasoning_mode = "general_after_card_selector" if selector_trace.get("attempted") else "general"
    allowed_tool_names = _allowed_tool_names_for_state(state)
    response = invoke_with_metrics(
        _get_ui_llm_with_tools(allowed_tool_names),
        _build_reasoning_messages(state, loop_warning, selector_trace),
        "vision_reasoning",
    )
    try:
        pending_action = action_request_from_model_response(
            response,
            allowed_tool_names=allowed_tool_names,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Model action request rejected", error=str(exc))
        pending_action = ActionRequest(
            source="llm",
            summary="모델이 유효하지 않은 도구 호출을 반환했습니다.",
            metadata={"validation_error": str(exc)},
        )
        error_increment += 1

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Reasoning Node completed",
        component="reasoning",
        duration_sec=round(elapsed, 6),
        reasoning_mode=reasoning_mode,
    )

    result = {
        "pending_action": pending_action,
        "job_card_selection_trace": selector_trace,
        "reflex_trace": {"hit": False, "source": "reasoning"},
        "reflex_transition_contracts": {},
    }
    if error_increment > 0:
        result["error_count"] = state.get("error_count", 0) + error_increment

    return result

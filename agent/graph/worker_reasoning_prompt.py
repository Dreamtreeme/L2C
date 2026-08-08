"""비전 작업자 추론에 전달할 화면·수집 문맥을 구성한다."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.config import get_settings
from agent.runtime.worker_contracts import WorkerState, action_event_results
from agent.graph.worker_execution_policy import compact_action_args
from agent.runtime.worker_state import (
    job_detail_key_from_state,
    return_to_job_results_for_url,
)
from agent.prompts.commander import COMMANDER_SYSTEM_PROMPT
from agent.runtime.action_validation import IMPLAUSIBLE_TEXT_INPUT_TARGET
from agent.runtime.detail_runtime import compact_job_detail_buffer_context
from agent.runtime.job_card_queue import pending_job_cards
from agent.runtime.job_collection import job_count, job_items
from agent.runtime.site_context import site_runtime_guidance
from agent.runtime.transition_runtime import latest_no_effect_transition
from agent.utils.job_fields import job_field_value
from agent.utils.logger import logger
from shared.schema.agent_contract import JOB_COLLECTION_FIELD_LABELS


def _is_open_browser_noop(action: dict[str, Any]) -> bool:
    if action.get("action") != "open_browser":
        return False
    result = action.get("result")
    return isinstance(result, dict) and result.get("opened") is False


def _recent_forbidden_actions(
    action_history: list[dict[str, Any]],
    limit: int = 6,
) -> list[dict[str, Any]]:
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
            forbidden_reason = action.get("result", {}).get(
                "reason",
                "open_browser_no_screen_change",
            )
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
        forbidden.append(
            {
                "action": action_name,
                "args": args,
                "reason": forbidden_reason,
            }
        )
        if len(forbidden) >= limit:
            break

    return forbidden


def _build_forbidden_action_context(
    action_history: list[dict[str, Any]],
) -> str:
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
            + json.dumps(
                item["args"],
                ensure_ascii=False,
                sort_keys=True,
            )
            + f" ({item['reason']})"
        )
    lines.append(
        "Choose a different visible marker or a different atomic navigation tool instead. "
        "If go_back had no effect on a detail page opened from results, consider close_current_tab."
    )
    return "\n".join(lines)


def _safety_page_role_contract() -> str:
    from agent.prompts.trust_boundary import external_content_contract_en

    return (
        "\n\n"
        + external_content_contract_en()
        + "\n[Safety and page-role contract]\n"
        "- For every UI tool call, include page_role when you can infer it: home, search, list, detail, form, popup, error, or unknown.\n"
        "- Include risk_level: safe_read, safe_navigation, or sensitive.\n"
        "- Set needs_user_confirmation=true before login, password/authentication, personal data, agreement/terms, application/submission, payment, transfer, account, finance, or legal-effect steps. The executor will stop and ask the user.\n"
        "- Set needs_user_confirmation=false for safe UI actions. For type_in_marker, set slot_name to the matching task input key such as query or keyword; type only values supplied by the task contract.\n"
        "- For public job collection, do not attempt login, signup, authentication, or account switching unless the user explicitly asked for it. If such a screen appears, leave that flow and return to a public search/list/home surface. Use neutral action reasons such as 'return to public search surface' instead of describing a login/signup action.\n"
        "- A marker whose OCR text is only a generic icon label has no known semantic identity. Infer it only from a clearly visible symbol; otherwise choose a nearby labeled text marker or another visible navigation path instead of inventing what its ID means.\n"
        "- Unknown or newly released tasks should be researched and narrowed before execution. Do not try random branches first.\n"
        "- On detail pages, report fields visibly confirmed on the current screen in observed_fields whenever you scroll, click a reveal control, or finish. The state keeps this evidence across screens without another extraction call.\n"
        "- If detail OCR buffering is active, do not call update_extracted_info for intermediate extraction. Call finish_detail_reading only after every field in the current required-field contract is confirmed. A field that the posting does not provide may be listed in unavailable_fields only after page_exhausted=true.\n"
        "- A title and company without actual duties or qualifications may be an intermediary page. Follow a visible original-source or content-reveal control before finishing; never guess a destination URL.\n"
        "- If finish_detail_reading was rejected with required_field_evidence_incomplete, use the returned missing_fields to choose one visible reveal, navigation, or scroll action instead of repeating finish.\n"
    )


def _clip_prompt_text(value: Any, max_chars: int = 160) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


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
        for index, (key, item) in enumerate(value.items()):
            if index >= 4:
                break
            if item in (None, "", [], {}):
                continue
            compacted[str(key)] = _compact_prompt_value(
                item,
                max_chars=80,
            )
        return compacted
    return _clip_prompt_text(value, max_chars=max_chars)


_JOB_SUMMARY_FIELDS: tuple[str, ...] = (
    "company_name",
    "position",
    "url",
    "main_tasks",
    "requirements",
    "preferred",
    "benefits",
)


def _job_display_label(job: dict[str, Any]) -> str:
    company = job_field_value(job, "company_name")
    position = job_field_value(job, "position")
    if company and position:
        return _clip_prompt_text(f"{company} - {position}", 120)
    return _clip_prompt_text(
        position or company or job.get("url") or "",
        120,
    )


def _job_summary_for_prompt(job: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    present_fields: list[str] = []
    missing_fields: list[str] = []
    for field in _JOB_SUMMARY_FIELDS:
        label = JOB_COLLECTION_FIELD_LABELS.get(field, field)
        value = job_field_value(job, field)
        if value in (None, "", [], {}):
            missing_fields.append(label)
            continue
        present_fields.append(label)
        if field in {"company_name", "position", "url"}:
            summary[label] = _compact_prompt_value(
                value,
                max_chars=140,
            )
        elif field in {"main_tasks", "requirements", "preferred", "benefits"}:
            summary[label] = _compact_prompt_value(
                value,
                max_chars=120,
            )
    if present_fields:
        summary["채워진필드"] = present_fields
    if missing_fields:
        summary["누락필드"] = missing_fields
    return summary


def _current_job_for_prompt(
    jobs: list[dict[str, Any]],
    current_url: str,
) -> dict[str, Any] | None:
    current_url = str(current_url or "").strip()
    if current_url:
        for job in reversed(jobs):
            if str(job.get("url") or "").strip() == current_url:
                return job
    return jobs[-1] if jobs else None


def _compact_extracted_context(
    extracted_jd: Any,
    current_url: str,
) -> str:
    jobs = job_items(extracted_jd)
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
        lines.append(
            "- 최근 공고: "
            + json.dumps(
                recent_labels,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    if current_job:
        summary = _job_summary_for_prompt(current_job)
        lines.append(
            "- 현재/최근 공고 핵심 필드: "
            + json.dumps(
                summary,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return "\n".join(lines) + "\n\n"


def _compact_recent_action(action: dict[str, Any]) -> dict[str, Any]:
    action_name = str(action.get("action") or "")
    args = action.get("args") or {}
    compact_args = (
        compact_action_args(action_name, args)
        if isinstance(args, dict)
        else {}
    )
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
        shown_args = {
            key: compact_args.get(key)
            for key in keep_keys
            if compact_args.get(key) not in (None, "", [], {})
        }
    item: dict[str, Any] = {
        "action": action_name,
        "status": action.get("status", ""),
        "args": shown_args,
    }
    reason = action.get("reason")
    if reason and action.get("status") != "success":
        item["reason"] = _clip_prompt_text(reason, 120)
    return item


def _compact_recent_actions_context(
    action_history: list[dict[str, Any]],
) -> str:
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
        + json.dumps(
            recent,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n\n"
    )


def _compact_job_card_queue_context(state: WorkerState) -> str:
    queue = [
        item
        for item in (state.get("job_card_queue", []) or [])
        if isinstance(item, dict)
    ]
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
    pending_count = len(pending_job_cards(queue))
    return (
        "공고 카드 큐:\n"
        f"- pending_count: {pending_count}\n"
        f"- cards: {json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}\n"
        "- 큐가 있으면 상세 수집 완료 후 다음 카드 선택은 executor가 처리합니다. 같은 목록에서 다음 카드를 다시 고르지 마십시오.\n\n"
    )


def _compact_detail_return_context(
    state: WorkerState,
    current_url: str,
) -> str:
    pending = return_to_job_results_for_url(state, current_url)
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


def _compact_job_results_availability_context(
    state: WorkerState,
) -> str:
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


def _reasoning_image_base64(state: WorkerState) -> str:
    marked_image_path = state.get("marked_image")
    if not marked_image_path or not os.path.exists(marked_image_path):
        return ""
    try:
        from pathlib import Path

        from agent.utils.image_utils import image_to_base64_jpeg

        settings = get_settings().vision
        return image_to_base64_jpeg(
            Path(marked_image_path),
            max_dim=settings.reasoning_image_max_dim,
            quality=settings.reasoning_image_quality,
            fast=True,
        )
    except Exception as exc:
        logger.warning(
            "Failed to read/resize marked_image for reasoning node",
            error=str(exc),
        )
        return ""


def build_reasoning_messages(
    state: WorkerState,
    loop_warning: str,
    selector_trace: dict[str, Any] | None = None,
) -> list:
    """현재 작업 상태를 텍스트 또는 멀티모달 추론 메시지로 만든다."""

    system_prompt_text = (
        COMMANDER_SYSTEM_PROMPT.format(goal=state.get("goal", ""))
        + _safety_page_role_contract()
    )
    extracted_jd = state.get("extracted_jd", {})
    ui_context = state.get("ui_context", "")
    current_url = state.get("current_url", "")
    action_history = action_event_results(
        state.get("action_events", []) or []
    )
    recipe_params = dict(state.get("recipe_params", {}) or {})
    target_count = int(recipe_params.get("target_count") or 0)
    collected_count = job_count(extracted_jd)
    visited_cards: list[str] = []
    for action in action_history:
        if not isinstance(action, dict) or action.get("status") != "success":
            continue
        args = action.get("args") or {}
        target = action.get("target") or {}
        component = (
            args.get("target_component")
            or target.get("component")
            or ""
        )
        if component != "job_card_title":
            continue
        label = (
            args.get("target_label")
            or target.get("target_label")
            or target.get("text")
            or ""
        )
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
        latest_transition = latest_no_effect_transition(state)
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
        refinement_reason = str(
            selector_trace.get("refinement_reason") or ""
        ).strip()
        job_results_refinement_context = (
            "검색 결과 정제 필요:\n"
            "- 현재 화면에서 검색어와 직접 일치하는 공고가 목표 수보다 부족합니다. 비슷한 직무로 개수를 채우지 마십시오.\n"
            "- 검색어를 더 정확하게 표현하는 화면 필터가 있으면 적용하고, 없으면 다음 정확한 후보를 찾도록 스크롤하십시오.\n"
            f"- 카드 선택기 판단: {refinement_reason or '(구체적 이유 없음)'}\n\n"
        )
    forbidden_action_context = _build_forbidden_action_context(
        action_history
    )
    if forbidden_action_context:
        forbidden_action_context += "\n\n"

    human_prompt_text = (
        f"{_compact_extracted_context(extracted_jd, current_url)}"
        f"현재 브라우저 URL:\n{current_url or '(확인 안 됨)'}\n\n"
        f"{site_runtime_guidance(current_url, state.get('current_page_role', ''))}"
        f"{collection_context}"
        f"{_compact_job_results_availability_context(state)}"
        f"{_compact_job_card_queue_context(state)}"
        f"{_compact_detail_return_context(state, current_url)}"
        f"{compact_job_detail_buffer_context(state, current_url, job_detail_key_from_state(state))}"
        f"{transition_context}"
        f"{job_results_refinement_context}"
        f"현재 화면 상태 (UI 마커):\n{ui_context + loop_warning}\n\n"
        f"{forbidden_action_context}"
        f"{_compact_recent_actions_context(action_history)}"
        "다음 행동을 결정하세요. 상세 페이지에서 OCR 버퍼가 활성화되어 있으면 중간 정보 추출 대신 "
        "finish_detail_reading으로 읽기 종료를 알리고, 그 외 화면에서 새로운 정보가 식별되었다면 "
        "update_extracted_info를 먼저 부르십시오."
    )

    base64_image = _reasoning_image_base64(state)
    if base64_image:
        logger.info(
            "Invoking reasoning node with multimodal SoM marked image..."
        )
        return [
            SystemMessage(content=system_prompt_text),
            HumanMessage(
                content=[
                    {"type": "text", "text": human_prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ]
            ),
        ]

    logger.info("Invoking reasoning node with text-only prompts...")
    return [
        SystemMessage(content=system_prompt_text),
        HumanMessage(content=human_prompt_text),
    ]


__all__ = ["build_reasoning_messages"]

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from agent.graph.state import GraphState
from agent.prompts.commander import COMMANDER_SYSTEM_PROMPT, QA_COMMANDER_SYSTEM_PROMPT
from agent.utils.logger import logger
from agent.tools.sqlite_query import sqlite_query
from agent.tools.realtime_scraping import realtime_scraping
from agent.tools.site_registry import list_collection_sites, get_collection_site_profile

_perception = None
_action_tools = None
_ui_llm_with_tools = None
_qa_llm_with_tools = None

# --- LLM 도구 정의용 Pydantic 모델 ---
class click_marker(BaseModel):
    """화면의 특정 ID 마커를 클릭합니다."""
    marker_id: int = Field(..., description="클릭할 마커의 ID")
    target_label: Optional[str] = Field(None, description="Visible title or label of the selected card/list item, when the marker is only part of a larger target")

class type_in_marker(BaseModel):
    """특정 id의 마커를 클릭한 후 텍스트를 입력합니다."""
    marker_id: int = Field(..., description="텍스트를 입력할 마커의 ID")
    text: str = Field(..., description="입력할 텍스트")

class scroll(BaseModel):
    """화면을 스크롤합니다."""
    direction: str = Field("down", description="스크롤 방향 ('down' 또는 'up')")

class press_key(BaseModel):
    """엔터, ESC 등 특수키를 누릅니다."""
    key: str = Field(..., description="누를 특수키 (예: 'enter', 'esc')")

class open_browser(BaseModel):
    """기본 브라우저를 열고 특정 URL에 접속합니다. 목표가 주어지면 가장 먼저 호출해야 할 수 있습니다."""
    url: str = Field(..., description="접속할 URL (예: https://www.wanted.co.kr)")

class close_browser(BaseModel):
    """열려 있는 브라우저 창을 닫습니다."""
    pass

class update_extracted_info(BaseModel):
    """현재 화면에서 식별한 채용 공고 정보를 수집 상태에 병합합니다. 변경된 공고 또는 새 필드만 보내도 됩니다. (예: {'공고목록': [{'회사명': '로이드케이', '직무명': '...', '주요업무': ['A']}]} 형태의 JSON 문자열)"""
    data_json: str = Field(..., description="업데이트할 정보 키-값 딕셔너리의 JSON 문자열")

class go_back(BaseModel):
    """브라우저의 뒤로가기(이전 페이지 이동) 기능을 실행합니다."""
    pass

class update_plan_progress(BaseModel):
    """현재 실행 중인 계획 단계를 업데이트하거나 필요시 계획을 수정합니다."""
    current_step: int = Field(..., description="수행 중인 계획 단계 인덱스 (0-indexed)")
    plan: Optional[List[str]] = Field(None, description="수정된 계획 단계 목록 (필요한 경우)")

class finish_task(BaseModel):
    """작업을 완료하고 최종 데이터를 반환합니다."""
    result: str = Field(..., description="최종 완료 요약 또는 결과 데이터")


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


def _get_ui_llm_with_tools():
    """브라우저 자동화용 LLM은 import 시점이 아니라 호출 시점에 준비합니다."""
    global _ui_llm_with_tools
    if _ui_llm_with_tools is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)
        _ui_llm_with_tools = llm.bind_tools([
            click_marker,
            type_in_marker,
            scroll,
            press_key,
            open_browser,
            close_browser,
            update_extracted_info,
            go_back,
            update_plan_progress,
            finish_task,
        ])
    return _ui_llm_with_tools


def _get_qa_llm_with_tools():
    """QA 지휘자용 LLM은 서버 import와 분리해 필요한 순간에만 초기화합니다."""
    global _qa_llm_with_tools
    if _qa_llm_with_tools is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        qa_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
        _qa_llm_with_tools = qa_llm.bind_tools([
            sqlite_query,
            list_collection_sites,
            get_collection_site_profile,
            realtime_scraping,
        ])
    return _qa_llm_with_tools


def perception_node(state: GraphState) -> Dict[str, Any]:
    """화면을 캡처하고 마커를 파싱하여 상태를 업데이트합니다."""
    start_time = time.time()
    logger.info("Executing Perception Node")
    perception = _get_perception()
    
    # 화면 캡처
    image_path = perception.capture_screen()

    current_url = state.get("current_url", "")
    current_url_stale = state.get("current_url_stale", True)
    if current_url_stale or not current_url:
        fetched_url = perception.get_current_url()
        if fetched_url:
            current_url = fetched_url
        current_url_stale = False

    analysis = perception.analyze_ui(image_path)
    markers = analysis.get("markers", [])
    marked_image = analysis.get("marked_image", "")
    
    # 우측 스크롤바 영역 마커 필터링 (우측 끝 35픽셀 이내 제거)
    from PIL import Image
    try:
        with Image.open(image_path) as img:
            img_width, _ = img.size
    except Exception as e:
        logger.error(f"Failed to open screenshot to get dimensions: {e}")
        img_width = 1929
        
    filtered_markers = []
    for m in markers:
        bbox = m.get("bbox", [0, 0, 0, 0])
        x_center = (bbox[0] + bbox[2]) // 2
        if x_center >= img_width - 65:
            logger.info(f"Filtering out scrollbar marker: ID {m.get('id')}, bbox {bbox}, text {m.get('text')}")
            continue
        filtered_markers.append(m)
    markers = filtered_markers
    
    ui_context = _build_ui_context(markers)
    ocr_texts = []
    ocr_delta_added = []
    ocr_delta_removed = []
    reflex_validation_status = ""
    try:
        from agent.recipe.ocr_delta import diff_marker_texts
        from agent.recipe.state_key import compute_state_key

        reflex_state_key = compute_state_key(current_url, markers)
        marker_delta = diff_marker_texts(state.get("ocr_texts", []), markers)
        ocr_texts = marker_delta["current"]
        ocr_delta_added = marker_delta["added"]
        ocr_delta_removed = marker_delta["removed"]
        if state.get("reflex_pending_validation"):
            expected_next = state.get("reflex_expected_next_state", "")
            if expected_next and reflex_state_key == expected_next:
                reflex_validation_status = "matched"
            elif ocr_delta_added or ocr_delta_removed:
                reflex_validation_status = "changed_unexpected"
            else:
                reflex_validation_status = "unchanged"
    except Exception as e:
        logger.debug("reflex state_key computation skipped", error=str(e))
        reflex_state_key = state.get("reflex_state_key", "")
    
    elapsed = time.time() - start_time
    logger.info(f"Perception Node completed in {elapsed:.2f} seconds")
    return {
        "recent_images": [image_path],
        "marked_image": marked_image,
        "current_markers": markers,
        "ui_context": ui_context,
        "current_url": current_url,
        "current_url_stale": current_url_stale,
        "reflex_state_key": reflex_state_key,
        "ocr_texts": ocr_texts,
        "ocr_delta_added": ocr_delta_added,
        "ocr_delta_removed": ocr_delta_removed,
        "reflex_validation_status": reflex_validation_status,
        "step_durations": [{"node": "perception", "duration": elapsed}]
    }


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


def _looks_like_job_detail_url(url: str) -> bool:
    return bool(url and re.search(r"/wd/\d+", url))


def _is_browser_back_marker_bbox(bbox: list) -> bool:
    """브라우저 툴바의 뒤로가기 버튼 마커를 일반 좌표 클릭 대신 go_back으로 처리합니다."""
    if len(bbox) != 4:
        return False
    x_center = (bbox[0] + bbox[2]) // 2
    y_center = (bbox[1] + bbox[3]) // 2
    return x_center <= 90 and 60 <= y_center <= 180


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

    incoming_jobs = new_data.get("공고목록")
    if isinstance(incoming_jobs, dict):
        incoming_jobs = [incoming_jobs]

    if isinstance(incoming_jobs, list):
        existing_jobs = merged.get("공고목록")
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
        if key == "공고목록":
            continue
        summary["fields"].append(key)
        merged[key] = _merge_value(merged.get(key), value)

    summary["fields"] = sorted({str(field) for field in summary["fields"]})
    if not summary["total_jobs"] and isinstance(merged.get("공고목록"), list):
        summary["total_jobs"] = len(merged["공고목록"])
    return merged, summary


def _compact_action_args(action_name: str, args: dict) -> dict:
    if action_name != "update_extracted_info":
        return args
    try:
        data = json.loads(args.get("data_json", "{}"))
    except Exception:
        return {"data_json": "<invalid json>"}
    jobs = data.get("공고목록")
    if isinstance(jobs, dict):
        jobs = [jobs]
    fields = []
    if isinstance(jobs, list):
        for job in jobs:
            if isinstance(job, dict):
                fields.extend(job.keys())
    fields.extend(k for k in data.keys() if k != "공고목록")
    return {
        "incoming_jobs": len(jobs) if isinstance(jobs, list) else 0,
        "fields": sorted({str(field) for field in fields}),
        "payload_chars": len(args.get("data_json", "")),
    }


def _marker_by_id(markers: list[dict], marker_id: int | None) -> dict | None:
    for marker in markers or []:
        if isinstance(marker, dict) and marker.get("id") == marker_id:
            return marker
    return None


def _action_target_metadata(state: GraphState, action_name: str, args: dict) -> dict | None:
    if action_name not in {"click_marker", "type_in_marker"}:
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
    target_label = args.get("target_label") or args.get("semantic_label")
    if target_label:
        metadata["target_label"] = target_label
    return metadata

def _state_snapshot_for_action(state: GraphState, current_url: str) -> dict:
    recent_images = state.get("recent_images", []) or []
    screenshot = str(recent_images[-1]) if recent_images else ""
    return {
        "state_key": state.get("reflex_state_key", "") or "",
        "url": current_url or state.get("current_url", "") or "",
        "screenshot": screenshot,
        "marked_image": state.get("marked_image", "") or "",
    }


def _repeat_target_text(target: dict | None) -> str:
    if not isinstance(target, dict):
        return ""
    text = target.get("target_label") or target.get("semantic_label") or target.get("text", "")
    try:
        from agent.recipe.state_key import normalize_text

        text = normalize_text(text)
    except Exception:
        text = str(text or "").strip()
    return text.lower().replace(" ", "")


def _action_repeat_keys(
    action_name: str,
    args: dict,
    state_key: str,
    target: dict | None = None,
) -> set[tuple[str, str, str]]:
    compact_args = _compact_action_args(action_name, args)
    keys = {
        (
            state_key or "",
            action_name,
            json.dumps(compact_args, ensure_ascii=False, sort_keys=True),
        )
    }
    if action_name in {"click_marker", "type_in_marker"}:
        target_text = _repeat_target_text(target)
        if target_text:
            semantic_args = dict(compact_args)
            semantic_args.pop("marker_id", None)
            semantic_args["marker_text"] = target_text
            keys.add(
                (
                    state_key or "",
                    action_name,
                    json.dumps(semantic_args, ensure_ascii=False, sort_keys=True),
                )
            )
    return keys


def _same_state_action_seen(state: GraphState, action_name: str, args: dict, state_key: str) -> bool:
    if not state_key:
        return False
    current_keys = _action_repeat_keys(
        action_name,
        args,
        state_key,
        _action_target_metadata(state, action_name, args),
    )
    for previous in reversed(state.get("action_history", []) or []):
        if not isinstance(previous, dict):
            continue
        if previous.get("status") not in {"success", "skipped", "error"}:
            continue
        previous_state_key = previous.get("state_key", "") or previous.get("before_state_key", "")
        if previous_state_key != state_key:
            continue
        previous_keys = _action_repeat_keys(
            previous.get("action", ""),
            previous.get("args", {}) or {},
            previous_state_key,
            previous.get("target"),
        )
        if current_keys.intersection(previous_keys):
            return True
    return False

def _is_open_browser_noop(action: dict) -> bool:
    if action.get("action") != "open_browser":
        return False
    result = action.get("result")
    return isinstance(result, dict) and result.get("opened") is False


def _recent_forbidden_actions(action_history: list[dict], limit: int = 6) -> list[dict]:
    forbidden = []
    seen = set()
    current_state_key = ""

    for action in reversed(action_history or []):
        if not isinstance(action, dict):
            continue

        action_state_key = action.get("state_key", "") or action.get("before_state_key", "")
        if not current_state_key and action_state_key:
            current_state_key = action_state_key
        if current_state_key and action_state_key and action_state_key != current_state_key:
            break

        reason = action.get("reason", "") or ""
        forbidden_reason = ""
        if reason == "same_state_repeat_blocked":
            forbidden_reason = reason
        elif reason == "unsafe_ui_action_chain":
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
        "Do not call these exact tool+args again in the current screen state; the executor will skip them:",
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
        "Choose a different visible marker, scroll, go back, extract information, or finish the task instead."
    )
    return "\n".join(lines)

def _chain_boundary_reached(action_name: str) -> bool:
    return action_name in {"click_marker", "scroll", "press_key", "open_browser", "close_browser", "go_back"}


def _is_allowed_same_screen_ui_chain(previous_ui_action: str | None, action_name: str) -> bool:
    return previous_ui_action == "type_in_marker" and action_name == "press_key"


def _marker_prompt_rank(marker: dict) -> tuple[int, int, int]:
    text = marker.get("text", "")
    bbox = marker.get("bbox", [0, 0, 0, 0])
    y = int(bbox[1]) if len(bbox) == 4 else 0
    x = int(bbox[0]) if len(bbox) == 4 else 0
    lowered = text.lower()
    important_terms = (
        "검색", "채용", "포지션", "데이터", "엔지니어", "개발", "로그인",
        "닫기", "x", "원티드", "wanted", "지원", "상세", "회사",
    )
    priority = 0 if any(term in lowered for term in important_terms) else 1
    return (priority, y, x)


def _build_ui_context(markers: list[dict]) -> str:
    try:
        text_limit = int(os.getenv("VISION_UI_TEXT_MARKER_LIMIT", "90"))
        icon_limit = int(os.getenv("VISION_UI_ICON_MARKER_LIMIT", "45"))
    except ValueError:
        text_limit = 90
        icon_limit = 45
    text_markers = []
    icon_markers = []
    for marker in markers:
        text = marker.get("text", "")
        if text.startswith("상호작용 가능한 요소 (") or text == "상호작용 가능한 요소":
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


def _build_reasoning_messages(state: GraphState, loop_warning: str) -> list:
    """
    reasoning_node용 LLM 메시지 리스트를 조립합니다.
    마킹 이미지가 있으면 멀티모달, 없으면 텍스트 전용 메시지를 반환합니다.
    """
    plan = state.get("plan", [])
    current_plan_step = state.get("current_plan_step", 0)
    plan_context = ""
    if plan:
        plan_context = "현재 수립된 세부 계획 단계:\n"
        for i, step in enumerate(plan):
            marker = "➡️" if i == current_plan_step else " "
            plan_context += f"  {marker} {i+1}. {step}\n"
        plan_context += f"(현재 단계: {current_plan_step + 1}번째 소목표 실행 중)\n\n"

    system_prompt_text = COMMANDER_SYSTEM_PROMPT.format(goal=state.get("goal", ""))
    extracted_jd = state.get("extracted_jd", {})
    ui_context = state.get("ui_context", "")
    current_url = state.get("current_url", "")
    action_history = state.get("action_history", [])
    forbidden_action_context = _build_forbidden_action_context(action_history)
    if forbidden_action_context:
        forbidden_action_context += "\n\n"

    human_prompt_text = (
        f"{plan_context}"
        f"현재까지 누적 수집된 정보:\n{json.dumps(extracted_jd, ensure_ascii=False, indent=2)}\n\n"
        f"현재 브라우저 URL:\n{current_url or '(확인 안 됨)'}\n\n"
        f"현재 화면 상태 (UI 마커):\n{ui_context + loop_warning}\n\n"
        f"{forbidden_action_context}"
        f"이전 행동 내역:\n{json.dumps(action_history[-5:], ensure_ascii=False, indent=2)}\n\n"
        f"다음 행동을 결정하세요. 새로운 정보가 식별되었다면 update_extracted_info를 먼저 부르고, "
        f"계획 단계 전환이 일어났다면 update_plan_progress를 함께 체이닝 호출하여 계획 진행률을 반영하십시오."
    )

    # 마킹 이미지가 있으면 멀티모달 메시지
    marked_image_path = state.get("marked_image")
    base64_image = ""
    if marked_image_path and os.path.exists(marked_image_path):
        try:
            from agent.utils.image_utils import image_to_base64_jpeg
            from pathlib import Path
            try:
                max_dim = int(os.getenv("VISION_REASONING_IMAGE_MAX_DIM", "768"))
                quality = int(os.getenv("VISION_REASONING_IMAGE_QUALITY", "60"))
            except ValueError:
                max_dim = 768
                quality = 60
            base64_image = image_to_base64_jpeg(Path(marked_image_path), max_dim=max_dim, quality=quality, fast=True)
        except Exception as img_err:
            logger.warning("Failed to read/resize marked_image for reasoning node", error=str(img_err))

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
    start_time = time.time()
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

    # 메시지 조립 + LLM 호출
    messages = _build_reasoning_messages(state, loop_warning)
    response = _get_ui_llm_with_tools().invoke(messages)

    elapsed = time.time() - start_time
    logger.info(f"Reasoning Node completed in {elapsed:.2f} seconds")

    result = {
        "last_action_result": response,
        "reflex_hit": False,
        "reflex_expected_next_state": "",
        "reflex_pending_validation": False,
        "step_durations": [{"node": "reasoning", "duration": elapsed}]
    }
    if error_increment > 0:
        result["error_count"] = state.get("error_count", 0) + error_increment

    return result


def _reflex_action_args(step: dict, marker_id: int | None) -> dict | None:
    """저장된 RecipeStep dict를 action_node tool args 형태로 변환한다."""
    action = step.get("action")
    param = dict(step.get("param") or {})
    value = step.get("value")

    if action == "click_marker":
        if marker_id is None:
            return None
        return {"marker_id": marker_id}
    if action == "type_in_marker":
        if marker_id is None:
            return None
        text = param.get("text") or value
        if not text:
            return None
        return {"marker_id": marker_id, "text": text}
    if action == "scroll":
        return {"direction": param.get("direction") or value or "down"}
    if action == "press_key":
        key = param.get("key") or value
        return {"key": key} if key else None
    if action == "go_back":
        return {}
    return None


def reflex_node(state: GraphState) -> Dict[str, Any]:
    """캐시된 Reflex Recipe가 있으면 reasoning을 우회해 같은 tool_call 형태를 만든다."""
    start_time = time.time()
    logger.info("Executing Reflex Node")

    try:
        from agent.recipe.matcher import is_replayable_step, match_marker
        from agent.recipe.state_key import compute_state_key
        from agent.recipe.store import RecipeStore

        markers = state.get("current_markers", []) or []
        state_key = state.get("reflex_state_key") or compute_state_key(state.get("current_url", ""), markers)
        recipe = RecipeStore().get_recipe(state_key)
        if not recipe or not recipe.steps:
            elapsed = time.time() - start_time
            logger.info("Reflex miss: no recipe", state_key=state_key[:24])
            return {
                "reflex_state_key": state_key,
                "reflex_hit": False,
                "reflex_expected_next_state": "",
                "reflex_pending_validation": False,
                "step_durations": [{"node": "reflex", "duration": elapsed}],
            }

        tool_calls = []
        expected_next = ""
        for idx, recipe_step in enumerate(recipe.steps):
            step = recipe_step.model_dump() if hasattr(recipe_step, "model_dump") else recipe_step.dict()
            action = step.get("action")
            marker_id = None
            params = {"goal": state.get("goal", "")}
            if not is_replayable_step(step, params=params):
                elapsed = time.time() - start_time
                logger.info("Reflex miss: cached step is not replayable", state_key=state_key[:24], action=action)
                return {
                    "reflex_state_key": state_key,
                    "reflex_hit": False,
                    "reflex_expected_next_state": "",
                    "reflex_pending_validation": False,
                    "step_durations": [{"node": "reflex", "duration": elapsed}],
                }
            if action in {"click_marker", "type_in_marker"}:
                marker_id = match_marker(step, markers, params=params)
                if marker_id is None:
                    elapsed = time.time() - start_time
                    logger.info("Reflex miss: marker did not match", state_key=state_key[:24], action=action)
                    return {
                        "reflex_state_key": state_key,
                        "reflex_hit": False,
                        "reflex_expected_next_state": "",
                        "reflex_pending_validation": False,
                        "step_durations": [{"node": "reflex", "duration": elapsed}],
                    }
            args = _reflex_action_args(step, marker_id)
            if args is None:
                elapsed = time.time() - start_time
                logger.info("Reflex miss: unsupported or incomplete action", state_key=state_key[:24], action=action)
                return {
                    "reflex_state_key": state_key,
                    "reflex_hit": False,
                    "reflex_expected_next_state": "",
                    "reflex_pending_validation": False,
                    "step_durations": [{"node": "reflex", "duration": elapsed}],
                }
            expected_next = step.get("expected_next_state") or expected_next
            tool_calls.append({"name": action, "args": args, "id": f"reflex_{abs(hash(state_key))}_{idx}"})

        msg = AIMessage(
            content=f"[reflex] cached {len(tool_calls)} action(s)",
            tool_calls=tool_calls,
        )
        elapsed = time.time() - start_time
        logger.info(
            "Reflex hit",
            state_key=state_key[:24],
            actions=[call["name"] for call in tool_calls],
            expected_next=expected_next[:24] if expected_next else "",
            duration=f"{elapsed:.3f}s",
        )
        return {
            "last_action_result": msg,
            "reflex_state_key": state_key,
            "reflex_hit": True,
            "reflex_expected_next_state": expected_next,
            "reflex_pending_validation": bool(expected_next),
            "step_durations": [{"node": "reflex", "duration": elapsed}],
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.debug("reflex node skipped", error=str(e))
        return {
            "reflex_hit": False,
            "reflex_expected_next_state": "",
            "reflex_pending_validation": False,
            "step_durations": [{"node": "reflex", "duration": elapsed}],
        }


def _dispatch_ui(action_name: str, args: dict, get_bbox, current_url: str = "") -> dict:
    """마우스/키보드 물리 조작 도구를 실행합니다."""
    action_tools = _get_action_tools()
    if action_name == "click_marker":
        bbox = get_bbox(args["marker_id"])
        if _is_browser_back_marker_bbox(bbox):
            logger.info(f"Redirecting browser toolbar back marker click to go_back: marker_id={args['marker_id']}, bbox={bbox}")
            return action_tools.go_back()
        return action_tools.click_marker(bbox)
    elif action_name == "type_in_marker":
        return action_tools.type_in_marker(get_bbox(args["marker_id"]), args["text"])
    elif action_name == "scroll":
        return action_tools.scroll(direction=args.get("direction", "down"))
    elif action_name == "press_key":
        return action_tools.press_key(args["key"])
    elif action_name == "open_browser":
        return action_tools.open_browser(args["url"], current_url=current_url)
    elif action_name == "close_browser":
        return action_tools.close_browser()
    elif action_name == "go_back":
        return action_tools.go_back()
    raise ValueError(f"Unknown UI action: {action_name}")


def _dispatch_state(
    action_name: str, args: dict,
    current_jd: dict, current_plan: list, current_plan_step: int, current_url: str = ""
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
            current_jd, summary = _merge_extracted_info(current_jd, new_data, current_url=current_url)
            result_str = (
                "Extracted data merged "
                f"(incoming_jobs={summary['incoming_jobs']}, total_jobs={summary['total_jobs']}, "
                f"fields={summary['fields']})"
            )
            status = "success"
        except Exception as e:
            result_str = f"Failed to parse data_json: {e}"
            status = "error"
        result = {"action": "update_extracted_info", "status": status, "result": result_str}
    else:
        raise ValueError(f"Unknown state action: {action_name}")
    return result, current_jd, current_plan, current_plan_step


def action_node(state: GraphState) -> Dict[str, Any]:
    """Reasoning Node가 선택한 도구(들)를 순차적으로 실행(Action Chaining)합니다."""
    start_time = time.time()
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
    prior_feedback_episodes = list(state.get("feedback_episodes", []) or [])

    ai_msg: AIMessage = state.get("last_action_result")

    if ai_msg and hasattr(ai_msg, "content") and ai_msg.content:
        logger.info(f"LLM Thoughts: {ai_msg.content}")

    if not ai_msg or not hasattr(ai_msg, "tool_calls") or not ai_msg.tool_calls:
        logger.warning("LLM did not return a tool call.")
        elapsed = time.time() - start_time
        return {
            "action_history": [{"action": "none", "status": "error", "error": "No tool call", "args": {}}],
            "step_durations": [{"node": "action", "duration": elapsed}]
        }

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
    screen_changed    = False
    chain_boundary    = False
    previous_ui_action: str | None = None

    # marker_id → bbox 변환 헬퍼
    def get_bbox(marker_id: int):
        marker = _marker_by_id(state.get("current_markers", []), marker_id)
        if marker:
            return marker["bbox"]
        raise ValueError(f"Marker ID {marker_id} not found in current screen.")

    def enrich_result(
        result: dict,
        requested_action: str,
        action_args: dict,
        before_snapshot: dict,
        screen_change_expected: bool = False,
    ) -> dict:
        result["args"] = _compact_action_args(requested_action, action_args)
        result["state_key"] = before_snapshot.get("state_key", "")
        result["before_state_key"] = before_snapshot.get("state_key", "")
        result["before_url"] = before_snapshot.get("url", "")
        result["before_screenshot"] = before_snapshot.get("screenshot", "")
        result["before_marked_image"] = before_snapshot.get("marked_image", "")
        result["screen_change_expected"] = screen_change_expected
        target = _action_target_metadata(state, requested_action, action_args)
        if target:
            result["target"] = target
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
    ) -> None:
        nonlocal error_count
        result = {
            "status": status,
            "action": action_name,
            "result": message if status != "error" else None,
            "error": message if status == "error" else None,
            "reason": reason,
        }
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
                    "screen_changed": False,
                    "extracted_jd": current_jd,
                    "is_finished": is_finished,
                },
                len(prior_feedback_episodes) + len(feedback_episodes),
            )
        if increments_error:
            error_count += 1
        step_elapsed = time.time() - step_start
        step_durations.append({"node": f"action ({action_name})", "duration": step_elapsed})
        logger.warning(message, action=action_name, reason=reason)

    # 도구 카테고리 라우팅 테이블
    UI_ACTIONS    = {"click_marker", "type_in_marker", "scroll", "press_key", "open_browser", "close_browser", "go_back"}
    SCREEN_CHANGING_ACTIONS = {"click_marker", "type_in_marker", "scroll", "press_key", "open_browser", "close_browser", "go_back"}
    URL_STALE_ACTIONS = {"click_marker", "press_key", "open_browser", "close_browser", "go_back"}
    STATE_ACTIONS = {"update_plan_progress", "update_extracted_info"}

    for idx, tool_call in enumerate(ai_msg.tool_calls):
        action_name = tool_call["name"]
        args        = tool_call["args"]
        compact_args = _compact_action_args(action_name, args)

        logger.info(
            f"LLM decided to call (chained {idx+1}/{len(ai_msg.tool_calls)}): "
            f"{action_name} with args: {compact_args}"
        )
        step_start = time.time()
        before_snapshot = _state_snapshot_for_action(state, current_url)
        before_state_key = before_snapshot.get("state_key", "")

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

                if _same_state_action_seen(state, action_name, args, before_state_key):
                    append_guard_result(
                        action_name,
                        args,
                        before_snapshot,
                        "skipped",
                        "same_state_repeat_blocked",
                        "Blocked repeated UI action in the same screen state.",
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
                if action_changed_screen and _chain_boundary_reached(action_name):
                    chain_boundary = True
                if record_ui_step:
                    record_ui_step(recorded_steps, state, action_name, args, len(prior_recorded_steps) + idx)

            elif action_name in STATE_ACTIONS:
                result, current_jd, current_plan, current_plan_step = _dispatch_state(
                    action_name, args, current_jd, current_plan, current_plan_step, current_url=current_url
                )
                action_changed_screen = False

            elif action_name == "finish_task":
                action_tools = _get_action_tools()
                result = action_tools.finish_task(args["result"])
                is_finished = True
                collected_data.append(args["result"])
                action_changed_screen = False

            else:
                raise ValueError(f"Unknown tool: {action_name}")

            enriched = enrich_result(result, action_name, args, before_snapshot, action_changed_screen)
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
                    len(prior_feedback_episodes) + len(feedback_episodes),
                )

            step_elapsed = time.time() - step_start
            step_durations.append({"node": f"action ({action_name})", "duration": step_elapsed})
            logger.info(f"Action Node [{action_name}] completed in {step_elapsed:.2f} seconds")

            if is_finished:
                break

        except Exception as e:
            logger.error(f"Failed to execute action {action_name}", error=str(e))
            step_elapsed = time.time() - step_start
            before_snapshot = _state_snapshot_for_action(state, current_url)
            result = {"action": action_name, "status": "error", "error": str(e)}
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
                        "screen_changed": False,
                        "extracted_jd": current_jd,
                        "is_finished": is_finished,
                    },
                    len(prior_feedback_episodes) + len(feedback_episodes),
                )
            error_count += 1
            step_durations.append({"node": f"action ({action_name})", "duration": step_elapsed})
            break  # 에러 발생 시 체인 중단

    if is_finished and commit_if_finished:
        commit_if_finished(prior_recorded_steps + recorded_steps, state, current_url)

    total_elapsed = time.time() - start_time
    logger.info(f"Action Node completed all chained tools in {total_elapsed:.2f} seconds")

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
        "last_action_screen_changed": screen_changed,
        "recorded_steps":    recorded_steps,
        "feedback_episodes": feedback_episodes,
    }


def validate_citations(answer: str, valid_ids: List[int]) -> str:
    """답변 내의 인용 ID를 검증하고, 유효하지 않은 경우 [출처 확인 불가]로 치환합니다."""
    import re
    valid_ids_str = {str(i) for i in valid_ids}
    
    def repl(match):
        jid = match.group(1)
        if jid in valid_ids_str:
            return match.group(0)
        else:
            return "[출처 확인 불가]"
            
    return re.sub(r"\[job_id:(\d+)\]", repl, answer)


def qa_reasoning_node(state: GraphState) -> Dict[str, Any]:
    """
    지휘자 모델(Gemini 3.5 Flash)이 사용자 질문을 받고
    SQLite 검색 도구 및 실시간 크롤링 도구를 직접 도구 호출(Tool Calling)을 통해 조율하며 최종 답변을 반환합니다.
    """
    from shared.config import DB_PATH
    import time
    
    start_time = time.time()
    logger.info("Executing Commander QA Reasoning Node (Agent Tool Calling Loop)")
    
    query = state.get("goal") or ""
    if not query:
        return {
            "last_action_result": "질문이 비어있습니다.",
            "is_finished": True,
            "step_durations": [{"node": "qa_reasoning", "duration": time.time() - start_time}]
        }

    # 메시지 리스트 초기화 (모듈 레벨 싱글톤 qa_llm_with_tools 사용)
    messages = [
        SystemMessage(content=QA_COMMANDER_SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]

    # 5개 사이트를 순차 수집할 수 있도록 여유를 두되 무한 루프는 방지
    max_turns = 14
    valid_ids = []
    
    for turn in range(max_turns):
        logger.info(f"Commander Agent Loop: Turn {turn + 1}")
        
        # 지휘자 LLM 호출
        response = _get_qa_llm_with_tools().invoke(messages)
        messages.append(response)
        
        # 1. 도구 호출이 있는 경우
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                logger.info(f"Commander decided to call tool: {tool_name} with args: {tool_args}")
                
                # 도구 매핑 및 실행
                if tool_name == "sqlite_query":
                    # DB SQLite 쿼리 실행
                    result_str = sqlite_query.invoke(tool_args)
                    
                    # XML 문서에서 id를 파싱하여 인용 검증용 valid_ids 채우기
                    import re
                    doc_ids = re.findall(r'<document id="(\d+)">', result_str)
                    for d_id in doc_ids:
                        try:
                            valid_ids.append(int(d_id))
                        except ValueError:
                            pass
                            
                elif tool_name == "list_collection_sites":
                    # 지휘자용 사이트 레지스트리 조회
                    result_str = list_collection_sites.invoke(tool_args)
                elif tool_name == "get_collection_site_profile":
                    # 사이트별 수집 지침 조회
                    result_str = get_collection_site_profile.invoke(tool_args)
                elif tool_name == "realtime_scraping":
                    # Playwright 실시간 수집 실행
                    result_str = realtime_scraping.invoke(tool_args)
                else:
                    result_str = f"알 수 없는 도구: {tool_name}"
                
                logger.info(f"Tool {tool_name} execution completed. Result summary: {result_str[:100]}...")
                
                # 도구 실행 결과를 지휘자 컨텍스트에 피드백
                messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))
                
        # 2. 도구 호출 없이 최종 답변을 도출한 경우
        else:
            logger.info("Commander formulated the final answer.")
            full_answer = response.content
            if isinstance(full_answer, list):
                full_answer = "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in full_answer])
            elif not isinstance(full_answer, str):
                full_answer = str(full_answer)
            
            # 인용 교정 적용 (validate_citations)
            final_answer = validate_citations(full_answer, list(set(valid_ids)))
            
            elapsed = time.time() - start_time
            logger.info(f"Commander Agent Loop finished successfully in {elapsed:.2f}s")
            
            return {
                "last_action_result": final_answer,
                "is_finished": True,
                "step_durations": [{"node": "qa_reasoning", "duration": elapsed}]
            }

    # 루프 초과 시 강제 거절 폴백
    elapsed = time.time() - start_time
    logger.error("Commander Agent Loop exceeded max_turns limit.")
    return {
        "last_action_result": "답변 생성 실패: 최대 추론 횟수를 초과하였습니다.",
        "is_finished": True,
        "step_durations": [{"node": "qa_reasoning", "duration": elapsed}]
    }

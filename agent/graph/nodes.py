import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from agent.graph.state import GraphState
from agent.prompts.commander import COMMANDER_SYSTEM_PROMPT, QA_COMMANDER_SYSTEM_PROMPT
from agent.utils.logger import logger
from agent.utils.model_dump import dump_model
from agent.vision.marker_geometry import (
    bbox_from_ratio,
    bbox_to_ratio,
    center_ratio_from_bbox,
    marker_bbox,
    screen_size_from_signature,
)
from agent.tools.sqlite_query import sqlite_query
from agent.tools.realtime_scraping import realtime_scraping
from agent.tools.site_registry import list_collection_sites, get_collection_site_profile
from agent.tools.recipe_learning import review_recipe_candidates

_perception = None
_action_tools = None
_ui_llm_with_tools = None
_qa_llm_with_tools = None
_detail_extraction_llm = None
_detail_extraction_llm_key = None

# --- LLM 도구 정의용 Pydantic 모델 ---
class click_marker(BaseModel):
    """화면의 특정 ID 마커를 클릭합니다."""
    marker_id: int = Field(..., description="클릭할 마커의 ID")
    target_label: Optional[str] = Field(None, description="선택한 항목의 보이는 제목(target_label)")
    target_role: Optional[str] = Field(None, description="목표 기준 대상 역할(target_role)")
    target_component: Optional[str] = Field(None, description="화면 구성요소(target_component)")
    reason: Optional[str] = Field(None, description="이 대상을 선택한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="클릭 후 정상이라면 보여야 할 화면 변화(expected_after)")

    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")

class type_in_marker(BaseModel):
    """특정 id의 마커를 클릭한 후 텍스트를 입력합니다."""
    marker_id: int = Field(..., description="텍스트를 입력할 마커의 ID")
    text: str = Field(..., description="입력할 텍스트")

    slot_name: Optional[str] = Field(None, description="실행마다 바뀌는 입력 슬롯 이름(slot_name)")
    target_role: Optional[str] = Field(None, description="목표 기준 대상 역할(target_role)")
    target_component: Optional[str] = Field(None, description="화면 구성요소(target_component)")
    reason: Optional[str] = Field(None, description="이 입력을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="입력 후 정상이라면 보여야 할 화면 변화(expected_after)")

    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")

class scroll(BaseModel):
    """화면을 스크롤합니다."""
    direction: str = Field("down", description="스크롤 방향 ('down' 또는 'up')")
    reason: Optional[str] = Field(None, description="스크롤을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="스크롤 후 정상이라면 보여야 할 화면 변화(expected_after)")

    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")

class press_key(BaseModel):
    """엔터, ESC 등 특수키를 누릅니다."""
    key: str = Field(..., description="누를 특수키 (예: 'enter', 'esc')")
    reason: Optional[str] = Field(None, description="키 입력을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="키 입력 후 정상이라면 보여야 할 화면 변화(expected_after)")

    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")

class open_browser(BaseModel):
    """기본 브라우저를 열고 특정 URL에 접속합니다. 목표가 주어지면 가장 먼저 호출해야 할 수 있습니다."""
    url: str = Field(..., description="접속할 URL (예: https://www.wanted.co.kr)")
    reason: Optional[str] = Field(None, description="이 URL을 여는 이유(reason)")
    expected_after: Optional[str] = Field(None, description="브라우저 이동 후 정상이라면 보여야 할 화면 변화(expected_after)")

    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")

class close_browser(BaseModel):
    """열려 있는 브라우저 창을 닫습니다."""
    reason: Optional[str] = Field(None, description="브라우저를 닫는 이유(reason)")
    expected_after: Optional[str] = Field(None, description="브라우저 종료 후 기대 상태(expected_after)")

    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")

class update_extracted_info(BaseModel):
    """현재 화면에서 식별한 채용 공고 정보를 수집 상태에 병합합니다. 변경된 공고 또는 새 필드만 보내도 됩니다. (예: {'공고목록': [{'회사명': '로이드케이', '직무명': '...', '주요업무': ['A']}]} 형태의 JSON 문자열)"""
    data_json: str = Field(..., description="업데이트할 정보 키-값 딕셔너리의 JSON 문자열")
    page_role: Optional[str] = Field(None, description="현재 정보를 읽은 페이지 역할(page_role). 상세 공고면 job_detail.")
    detail_complete: Optional[bool] = Field(None, description="상세 공고 본문 정보가 충분히 수집되었는지 여부(detail_complete).")

    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")

class finish_detail_reading(BaseModel):
    """상세 페이지 OCR 누적을 종료하고, 누적 본문을 한 번만 정제하여 수집 상태에 병합합니다."""
    reason: Optional[str] = Field(None, description="상세 페이지 읽기를 종료하는 이유(reason)")
    detail_complete: Optional[bool] = Field(True, description="상세 공고 본문 정보가 충분히 수집되었는지 여부(detail_complete).")
    expected_after: Optional[str] = Field(None, description="정제 후 정상이라면 다음에 기대되는 상태(expected_after)")

    page_role: Optional[str] = Field("job_detail", description="Current page role.")
    risk_level: Optional[str] = Field("safe_read", description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")

class go_back(BaseModel):
    """브라우저의 뒤로가기(이전 페이지 이동) 기능을 실행합니다."""
    reason: Optional[str] = Field(None, description="뒤로가기를 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="뒤로가기 후 정상이라면 보여야 할 화면 변화(expected_after)")

    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")

class update_plan_progress(BaseModel):
    """현재 실행 중인 계획 단계를 업데이트하거나 필요시 계획을 수정합니다."""
    current_step: int = Field(..., description="수행 중인 계획 단계 인덱스 (0-indexed)")
    plan: Optional[List[str]] = Field(None, description="수정된 계획 단계 목록 (필요한 경우)")

class set_result_card_queue(BaseModel):
    """검색 결과 목록에서 현재 화면에 보이는 수집 대상 공고 카드들을 런타임 작업 큐에 저장합니다."""
    cards: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "수집할 공고 카드 목록. 각 항목에는 marker_id와 title/target_label, company를 가능한 만큼 넣으십시오. "
            "현재 화면에 보이는 카드만 넣어야 합니다."
        ),
    )
    titles: Optional[List[str]] = Field(
        None,
        description=(
            "cards 객체를 만들기 어려울 때만 사용합니다. 현재 화면에 보이는 수집 대상 공고 제목 목록입니다. "
            "executor는 OCR 마커 텍스트와 정확히 대응되는 제목만 큐에 저장합니다."
        ),
    )
    companies: Optional[List[str]] = Field(
        None,
        description="titles와 같은 순서로 대응되는 회사명 목록입니다. 모르면 생략하십시오.",
    )
    reason: Optional[str] = Field(None, description="이 카드들을 큐에 넣은 이유")

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
            finish_detail_reading,
            go_back,
            update_plan_progress,
            set_result_card_queue,
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
            review_recipe_candidates,
        ])
    return _qa_llm_with_tools


class _OllamaDetailExtractionLLM:
    """Ollama JSON 모드로 상세 OCR 누적 본문을 JobPosting으로 정제합니다."""

    def __init__(self, model_name: str):
        import ollama
        from shared.config import OLLAMA_HOST

        self.model_name = model_name
        self.client = ollama.Client(host=os.getenv("OLLAMA_HOST", str(OLLAMA_HOST)))

    @staticmethod
    def _message_content(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            return "\n".join(part for part in parts if part)
        return str(content or "")

    @staticmethod
    def _response_content(response: Any) -> str:
        if isinstance(response, dict):
            message = response.get("message") or {}
            return str(message.get("content") or "")
        message = getattr(response, "message", None)
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(getattr(message, "content", "") or "")

    def invoke(self, messages: list[Any]) -> Any:
        from json_repair import repair_json
        from shared.schema.jd_schema import JobPosting

        system_text = "\n".join(
            self._message_content(message)
            for message in messages
            if isinstance(message, SystemMessage)
        )
        user_text = "\n\n".join(
            self._message_content(message)
            for message in messages
            if not isinstance(message, SystemMessage)
        )
        schema = JobPosting.model_json_schema()
        prompt = (
            f"{system_text}\n\n"
            "반드시 아래 JSON 스키마와 호환되는 JSON 객체 하나만 출력하십시오.\n"
            f"{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"{user_text}"
        )
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={
                "temperature": 0,
                "num_predict": int(os.getenv("VISION_DETAIL_OLLAMA_NUM_PREDICT", "2048")),
            },
        )
        content = self._response_content(response)
        parsed = repair_json(content, return_objects=True)
        if not isinstance(parsed, dict):
            parsed = {}
        return JobPosting.model_validate(parsed)


class _OpenAIDetailExtractionLLM:
    """OpenAI Responses API로 상세 OCR 누적 본문을 JobPosting으로 정제합니다."""

    def __init__(self, model_name: str):
        import requests
        from shared.config import BASE_DIR  # noqa: F401 - .env 로드를 보장합니다.

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model_name = model_name
        self.api_key = api_key
        self.requests = requests

    @staticmethod
    def _message_content(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            return "\n".join(part for part in parts if part)
        return str(content or "")

    @staticmethod
    def _output_text(response_json: dict[str, Any]) -> str:
        if isinstance(response_json.get("output_text"), str):
            return response_json["output_text"]
        parts: list[str] = []
        for item in response_json.get("output") or []:
            for content in item.get("content") or []:
                if content.get("type") == "output_text":
                    parts.append(str(content.get("text") or ""))
        return "\n".join(parts)

    @staticmethod
    def _job_posting_schema() -> dict[str, Any]:
        from shared.schema.jd_schema import JobPosting

        schema = JobPosting.model_json_schema()
        properties = dict(schema.get("properties") or {})
        for noisy_field in ("raw_ocr_text", "content_hash"):
            properties.pop(noisy_field, None)
        schema["properties"] = properties
        if isinstance(schema.get("required"), list):
            schema["required"] = [key for key in schema["required"] if key in properties]
        schema["additionalProperties"] = False
        return schema

    def invoke(self, messages: list[Any]) -> Any:
        from json_repair import repair_json
        from shared.schema.jd_schema import JobPosting

        system_text = "\n".join(
            self._message_content(message)
            for message in messages
            if isinstance(message, SystemMessage)
        )
        user_text = "\n\n".join(
            self._message_content(message)
            for message in messages
            if not isinstance(message, SystemMessage)
        )
        system_prompt = (
            f"{system_text} "
            "북마크, 브라우저 메뉴, 보상 배지, 추천인 현금, 로그인 문구 같은 주변 UI 노이즈는 무시하십시오. "
            "채용 도메인에서 명확한 OCR 혼동은 문맥으로 보정하십시오. 예를 들어 Swift, Xcode, 앱 개발, 모바일 문맥에서 "
            "'ios', 'i0S', 'j0s', '10s'처럼 보이는 토큰은 직무명과 기술스택에서 'iOS'로 정규화하십시오. "
            "대괄호로 나뉜 직무명 조각은 한 줄 직무명으로 합치고, 직무명에는 브라우저/광고/보상 문구를 넣지 마십시오. "
            "회사명은 로고, 영문 브랜드, 회사소개 문장 주변의 반복 토큰을 우선하고, 깨진 한글 OCR만으로 확정하지 마십시오. "
            "기술스택은 실제 업무에 쓰는 기술만 넣고, 면접 질문 예시나 CS 개념 목록은 requirements에 요약하십시오. "
            "salary, deadline, location, benefits는 서로 섞지 말고 해당 필드에만 넣으십시오. "
            "목록 필드는 핵심 항목만 간결하게 유지하십시오. "
            "raw_ocr_text와 content_hash는 출력하지 마십시오. JSON 객체 하나만 출력하십시오."
        )
        payload = {
            "model": self.model_name,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "max_output_tokens": int(os.getenv("VISION_DETAIL_OPENAI_MAX_OUTPUT_TOKENS", "2048")),
            "temperature": 0,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "job_posting",
                    "schema": self._job_posting_schema(),
                    "strict": False,
                }
            },
            "store": False,
        }
        response = self.requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=int(os.getenv("VISION_DETAIL_OPENAI_TIMEOUT", "120")),
        )
        try:
            response_json = response.json()
        except ValueError:
            response_json = {"raw_text": response.text}
        if response.status_code >= 400:
            raise RuntimeError(json.dumps(response_json, ensure_ascii=False)[:2000])
        parsed = repair_json(self._output_text(response_json), return_objects=True)
        if not isinstance(parsed, dict):
            parsed = {}
        return JobPosting.model_validate(parsed)


def _detail_extraction_model_spec() -> str:
    return os.getenv("VISION_DETAIL_FINAL_EXTRACTION_MODEL", "openai:gpt-5.4-mini").strip()


def _get_detail_extraction_llm():
    """상세 OCR 누적 본문을 최종 채용공고 스키마로 정제하는 전용 LLM입니다."""
    global _detail_extraction_llm
    global _detail_extraction_llm_key
    model_spec = _detail_extraction_model_spec()
    if _detail_extraction_llm is None or _detail_extraction_llm_key != model_spec:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from shared.schema.jd_schema import JobPosting

        if model_spec.startswith("ollama:"):
            model_name = model_spec.removeprefix("ollama:").strip()
            _detail_extraction_llm = _OllamaDetailExtractionLLM(model_name)
        elif model_spec.startswith("openai:"):
            model_name = model_spec.removeprefix("openai:").strip()
            _detail_extraction_llm = _OpenAIDetailExtractionLLM(model_name)
        else:
            _detail_extraction_llm = ChatGoogleGenerativeAI(
                model=model_spec,
                temperature=0.0,
            ).with_structured_output(JobPosting)
        _detail_extraction_llm_key = model_spec
    return _detail_extraction_llm


def perception_node(state: GraphState) -> Dict[str, Any]:
    """화면을 캡처하고 마커를 파싱하여 상태를 업데이트합니다."""
    start_time = time.time()
    logger.info("Executing Perception Node")
    perception = _get_perception()
    
    # 화면 캡처
    capture = getattr(perception, "capture_usable_screen", perception.capture_screen)
    image_path = capture()

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
    
    screen_signature = {}
    try:
        from agent.vision.screen_signature import compute_screen_signature

        screen_signature = compute_screen_signature(image_path, markers)
    except Exception as e:
        logger.debug("screen signature skipped", error=str(e))
    transition_observations = []
    transition_status = ""
    transition_outcome = ""
    transition_source = ""
    pending_transition = dict(state.get("pending_transition", {}) or {})
    observed_transition = dict(pending_transition)
    try:
        from agent.recipe.state_key import compute_state_key
        from agent.recipe.transition import evaluate_transition, marker_texts

        reflex_state_key = compute_state_key(current_url, markers)
        if pending_transition:
            started_at = float(pending_transition.get("started_at") or start_time)
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
            attempt = int(pending_transition.get("attempts") or 0) + 1
            transition_observations.append(
                {
                    "action_seq": pending_transition.get("action_seq"),
                    "action": pending_transition.get("action", ""),
                    "expected_after": pending_transition.get("expected_after", ""),
                    "source": transition_source,
                    "attempt": attempt,
                    "elapsed_sec": round(elapsed_sec, 3),
                    "status": transition_status,
                    "outcome": transition_outcome,
                    "reason": evaluation.get("reason", ""),
                    "state_key": reflex_state_key,
                    "marker_count": len(markers),
                    "marker_texts": marker_texts(markers),
                    "screenshot": str(image_path),
                    "marked_image": str(marked_image or ""),
                }
            )
            if transition_status == "pending":
                pending_transition["attempts"] = attempt
            else:
                pending_transition = {}
    except Exception as e:
        logger.debug("transition observation skipped", error=str(e))
        reflex_state_key = state.get("reflex_state_key", "")

    queue_msg = None
    queue_trace: dict[str, Any] = {}
    if observed_transition:
        queue_msg, markers, queue_trace = _queue_replay_after_return(
            state,
            observed_transition,
            current_url,
            reflex_state_key,
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

    ui_context = _build_ui_context(markers, current_url=current_url)
    detail_marked_image = _build_detail_lightweight_marked_image(image_path, markers, current_url)
    if detail_marked_image:
        marked_image = detail_marked_image
    detail_ocr_buffer = _update_detail_ocr_buffer(
        state.get("detail_ocr_buffer", {}),
        markers,
        current_url,
        image_path,
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Perception Node completed in {elapsed:.2f} seconds")
    result = {
        "recent_images": [image_path],
        "marked_image": marked_image,
        "current_markers": markers,
        "ui_context": ui_context,
        "screen_signature": screen_signature,
        "current_url": current_url,
        "current_url_stale": current_url_stale,
        "reflex_state_key": reflex_state_key,
        "pending_transition": pending_transition,
        "transition_status": transition_status,
        "transition_outcome": transition_outcome,
        "transition_source": transition_source,
        "transition_observations": transition_observations,
        "queue_replay_hit": bool(queue_msg),
        "queue_replay_trace": queue_trace if queue_msg else {},
        "detail_ocr_buffer": detail_ocr_buffer,
        "step_durations": [{"node": "perception", "duration": elapsed}]
    }
    if queue_msg:
        result["last_action_result"] = queue_msg
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


def _site_profile_for_url(url: str) -> dict:
    try:
        host = (urlparse(url or "").netloc or "").lower()
    except Exception:
        return {}
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return {}
    try:
        from agent.sites import list_supported_sites, load_site_profile

        for entry in list_supported_sites(enabled_only=False):
            domains = [str(domain or "").lower() for domain in entry.get("domains", [])]
            if any(host == domain or host.endswith("." + domain) for domain in domains):
                return load_site_profile(str(entry.get("slug") or ""))
    except Exception:
        return {}
    return {}


def _persistence_policy_for_url(url: str) -> dict:
    profile = _site_profile_for_url(url)
    manual = profile.get("manual", {}) if isinstance(profile, dict) else {}
    policy = manual.get("persistence_policy", {}) if isinstance(manual, dict) else {}
    return policy if isinstance(policy, dict) else {}


def _looks_like_job_detail_url(url: str) -> bool:
    pattern = str(_persistence_policy_for_url(url).get("detail_url_pattern") or "").strip()
    return bool(url and pattern and re.search(pattern, url))


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


JOB_LIST_KEYS = ("\uacf5\uace0\ubaa9\ub85d", "jobs", "job_list")


def _job_list_value(data: dict) -> Any:
    for key in JOB_LIST_KEYS:
        if key in data:
            return data.get(key)
    return None


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


def _auto_finish_on_target_enabled() -> bool:
    raw = os.getenv("VISION_AUTO_FINISH_ON_TARGET", "1")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _is_detail_update(args: dict[str, Any]) -> bool:
    role = str(args.get("page_role") or "").strip().lower()
    return role in {"job_detail", "detail", "posting_detail"}


def _sensitive_action_reason(state: GraphState, action_name: str, args: dict[str, Any]) -> str:
    if action_name in {"close_browser", "go_back", "scroll"}:
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
        return args
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


def _marker_by_id(markers: list[dict], marker_id: int | None) -> dict | None:
    for marker in markers or []:
        if isinstance(marker, dict) and marker.get("id") == marker_id:
            return marker
    return None


def _card_queue_enabled() -> bool:
    raw = os.getenv("VISION_RESULT_CARD_QUEUE_ENABLED", "1")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _queue_card_label(card: dict) -> str:
    for key in ("title", "target_label", "position", "text", "label"):
        value = card.get(key)
        if value not in (None, "", [], {}):
            return str(value).strip()
    return ""


def _text_list_arg(args: dict, key: str) -> list[str]:
    value = args.get(key)
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _result_card_entries_from_args(args: dict) -> list[dict]:
    """도구 입력을 카드 후보 목록으로 통일한다.

    모델이 cards 객체 목록을 정상 제공하면 그대로 쓰고, titles 목록만 준 경우에는
    제목 기반 후보로 변환한다. 실제 좌표 검증은 큐 정규화 단계에서만 수행한다.
    """
    cards = args.get("cards")
    if isinstance(cards, list):
        entries: list[dict] = []
        for raw in cards:
            if isinstance(raw, dict):
                entries.append(dict(raw))
            elif isinstance(raw, str) and raw.strip():
                entries.append({"title": raw.strip()})
        if entries:
            return entries

    titles = _text_list_arg(args, "titles")
    if not titles:
        titles = _text_list_arg(args, "target_labels")
    companies = _text_list_arg(args, "companies")
    entries = []
    for index, title in enumerate(titles):
        entry = {"title": title}
        if index < len(companies):
            entry["company"] = companies[index]
        entries.append(entry)
    return entries


def _queue_match_text(value: Any) -> str:
    try:
        from agent.recipe.state_key import normalize_text

        text = normalize_text(value)
    except Exception:
        text = str(value or "").strip()
    return text.casefold().replace(" ", "")


def _marker_by_label(markers: list[dict], label: str, used_marker_ids: set[int]) -> dict | None:
    label_key = _queue_match_text(label)
    if not label_key:
        return None
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        try:
            marker_id = int(marker.get("id"))
        except (TypeError, ValueError):
            marker_id = -1
        if marker_id in used_marker_ids:
            continue
        if _queue_match_text(marker.get("text")) == label_key:
            return marker
    return None


def _normalize_result_card_queue(args: dict, state: GraphState, current_url: str) -> tuple[list[dict], dict]:
    """LLM이 고른 현재 화면의 공고 카드를 런타임 큐 항목으로 정규화한다."""
    cards = _result_card_entries_from_args(args)
    markers = list(state.get("current_markers", []) or [])
    signature = dict(state.get("screen_signature", {}) or {})
    size = screen_size_from_signature(signature)
    queue: list[dict] = []
    used_marker_ids: set[int] = set()
    target_count = _target_count_from_state(state)
    remaining = target_count - _collected_job_count(state.get("extracted_jd", {}) or {}) if target_count > 0 else len(cards)
    limit = max(0, remaining) if target_count > 0 else len(cards)

    for raw in cards[:limit]:
        if not isinstance(raw, dict):
            continue
        marker_id = raw.get("marker_id", raw.get("id"))
        try:
            marker_id = int(marker_id) if marker_id is not None else None
        except (TypeError, ValueError):
            marker_id = None
        marker = _marker_by_id(markers, marker_id)
        label = _queue_card_label(raw) or (str(marker.get("text") or "").strip() if marker else "")
        if marker is None and label:
            marker = _marker_by_label(markers, label, used_marker_ids)
            if marker:
                try:
                    marker_id = int(marker.get("id"))
                except (TypeError, ValueError):
                    marker_id = None
        if not label:
            continue
        if marker_id is not None:
            used_marker_ids.add(marker_id)

        bbox = marker.get("bbox") if marker else raw.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = []
        bbox_ratio = raw.get("bbox_ratio")
        center_ratio = raw.get("center_ratio")
        if marker and size and not bbox_ratio:
            try:
                bbox_ratio = bbox_to_ratio(bbox, size)
                center_ratio = center_ratio_from_bbox(bbox, size)
            except Exception:
                bbox_ratio = []
                center_ratio = []
        if (not bbox_ratio or len(bbox_ratio) != 4) and raw.get("bbox_ratio"):
            bbox_ratio = raw.get("bbox_ratio")
        if (not center_ratio or len(center_ratio) != 2) and raw.get("center_ratio"):
            center_ratio = raw.get("center_ratio")
        if not bbox_ratio and not marker:
            continue

        queue_id = str(raw.get("queue_id") or raw.get("id") or f"card-{len(queue) + 1}")
        evidence_texts = raw.get("evidence_texts") if isinstance(raw.get("evidence_texts"), list) else []
        company = str(raw.get("company") or raw.get("company_name") or "").strip()
        if company and company not in evidence_texts:
            evidence_texts = [company, *evidence_texts]
        item = {
            "queue_id": queue_id,
            "status": "pending",
            "title": label,
            "company": company,
            "source_marker_id": marker_id,
            "bbox_ratio": bbox_ratio or [],
            "center_ratio": center_ratio or [],
            "evidence_texts": evidence_texts[:6],
            "target": {
                "text": str(marker.get("text") or label) if marker else label,
                "semantic_label": label,
                "bbox_ratio": bbox_ratio or [],
                "center_ratio": center_ratio or [],
                "evidence_texts": evidence_texts[:6],
            },
        }
        queue.append(item)

    memory = {
        "state_key": state.get("reflex_state_key", "") or "",
        "url": current_url or state.get("current_url", "") or "",
        "screen_signature": signature,
        "screenshot": str((state.get("recent_images") or [""])[-1] or "") if state.get("recent_images") else "",
        "marked_image": state.get("marked_image", "") or "",
    }
    return queue, memory


def _pending_result_cards(queue: list[dict]) -> list[dict]:
    return [
        dict(item)
        for item in queue or []
        if isinstance(item, dict) and str(item.get("status") or "pending") == "pending"
    ]


def _same_queue_card(item: dict, args: dict) -> bool:
    if args.get("queue_id") and str(args.get("queue_id")) == str(item.get("queue_id")):
        return True
    label = str(args.get("target_label") or "").strip()
    if label and label == str(item.get("title") or "").strip():
        return True
    marker_id = args.get("marker_id")
    return marker_id is not None and str(marker_id) == str(item.get("source_marker_id"))


def _mark_result_card_active(queue: list[dict], args: dict) -> tuple[list[dict], dict]:
    updated = []
    active: dict = {}
    for raw in queue or []:
        item = dict(raw)
        if not active and item.get("status") == "pending" and _same_queue_card(item, args):
            item["status"] = "active"
            active = dict(item)
        updated.append(item)
    return updated, active


def _result_card_click_matches_queue(queue: list[dict], args: dict) -> bool:
    """LLM이 공고 카드 클릭을 넓게 표현해도 큐 항목 제목과 맞으면 카드 클릭으로 인정한다."""
    if not queue:
        return False
    if args.get("queue_id"):
        return True

    component = str(args.get("target_component") or "")
    role = str(args.get("target_role") or "")
    if component in {"job_card", "job_card_title"} or role in {"job_card", "job_card_title"}:
        return True

    label = str(args.get("target_label") or "").strip()
    if not label:
        return False
    return any(label == str(item.get("title") or "").strip() for item in queue if isinstance(item, dict))


def _complete_active_result_card(queue: list[dict], active_card: dict) -> tuple[list[dict], dict]:
    if not active_card:
        return queue, {}
    active_id = str(active_card.get("queue_id") or "")
    updated = []
    for raw in queue or []:
        item = dict(raw)
        if active_id and str(item.get("queue_id") or "") == active_id:
            item["status"] = "done"
        updated.append(item)
    return updated, {}


def _queue_return_screen_matches(memory: dict, current_url: str, state_key: str, current_signature: dict) -> tuple[bool, dict]:
    if not memory:
        return False, {"reason": "queue_memory_missing"}
    if _looks_like_job_detail_url(current_url):
        return False, {"reason": "still_on_detail_url", "url": current_url}
    saved_state_key = str(memory.get("state_key") or "")
    if saved_state_key and state_key and saved_state_key == state_key:
        return True, {"reason": "state_key_match", "state_key": state_key}

    saved_signature = dict(memory.get("screen_signature") or {})
    saved_phash = str(saved_signature.get("phash") or "")
    current_phash = str((current_signature or {}).get("phash") or "")
    if not saved_phash or not current_phash:
        return False, {"reason": "phash_missing", "saved_phash": bool(saved_phash), "current_phash": bool(current_phash)}
    try:
        from agent.recipe.phash_replay import anchor_overlap
        from agent.vision.screen_signature import hamming_distance

        distance = hamming_distance(saved_phash, current_phash)
        overlap = anchor_overlap(saved_signature.get("anchors") or [], (current_signature or {}).get("anchors") or [])
    except Exception as exc:
        return False, {"reason": "phash_compare_failed", "error": str(exc)}

    try:
        max_distance = int(os.getenv("VISION_CARD_QUEUE_RETURN_PHASH_MAX_DISTANCE", "16"))
        min_overlap = float(os.getenv("VISION_CARD_QUEUE_RETURN_MIN_ANCHOR_OVERLAP", "0.20"))
    except ValueError:
        max_distance = 16
        min_overlap = 0.20
    matched = distance is not None and distance <= max_distance and overlap >= min_overlap
    return matched, {
        "reason": "phash_anchor_match" if matched else "phash_anchor_mismatch",
        "distance": distance,
        "max_distance": max_distance,
        "anchor_overlap": overlap,
        "min_anchor_overlap": min_overlap,
    }


def _queue_marker_for_item(item: dict, markers: list[dict], signature: dict) -> tuple[int | None, list[dict], dict]:
    target = dict(item.get("target") or {})
    target.setdefault("text", item.get("title", ""))
    target.setdefault("semantic_label", item.get("title", ""))
    target.setdefault("bbox_ratio", item.get("bbox_ratio") or [])
    target.setdefault("center_ratio", item.get("center_ratio") or [])
    target.setdefault("evidence_texts", item.get("evidence_texts") or [])
    try:
        from agent.recipe.phash_replay import match_target_by_ratio

        marker_id = match_target_by_ratio(target, markers, screen_size_from_signature(signature))
        if marker_id is not None:
            return marker_id, markers, {"reason": "current_marker_ratio_match"}
    except Exception as exc:
        ratio_error = str(exc)
    else:
        ratio_error = ""

    size = screen_size_from_signature(signature)
    bbox = bbox_from_ratio(item.get("bbox_ratio") or [], size)
    if bbox == [0, 0, 0, 0]:
        return None, markers, {"reason": "cached_bbox_missing", "ratio_error": ratio_error}
    next_id = max([int(marker.get("id") or 0) for marker in markers or [] if isinstance(marker, dict)] + [-1]) + 1
    synthetic = {
        "id": next_id,
        "bbox": bbox,
        "text": item.get("title") or "queued result card",
        "type": "queue_cached_card",
    }
    return next_id, [*markers, synthetic], {"reason": "synthetic_marker_from_cached_bbox", "bbox": bbox, "ratio_error": ratio_error}


def _queue_replay_after_return(
    state: GraphState,
    observed_transition: dict,
    current_url: str,
    state_key: str,
    markers: list[dict],
    screen_signature: dict,
) -> tuple[AIMessage | None, list[dict], dict]:
    """상세 페이지에서 목록으로 돌아온 직후 pending 카드가 있으면 다음 카드 클릭을 준비한다."""
    if not _card_queue_enabled():
        return None, markers, {"reason": "queue_disabled"}
    if str(observed_transition.get("action") or "") != "go_back":
        return None, markers, {"reason": "last_transition_not_go_back"}
    queue = [dict(item) for item in (state.get("result_card_queue", []) or []) if isinstance(item, dict)]
    pending = _pending_result_cards(queue)
    if not pending:
        return None, markers, {"reason": "queue_empty"}
    active_card = dict(state.get("active_result_card", {}) or {})
    if active_card:
        return None, markers, {
            "reason": "active_card_not_completed",
            "queue_id": active_card.get("queue_id", ""),
            "title": active_card.get("title", ""),
        }

    matched, match_trace = _queue_return_screen_matches(
        dict(state.get("result_page_memory", {}) or {}),
        current_url,
        state_key,
        screen_signature,
    )
    if not matched:
        return None, markers, match_trace

    item = pending[0]
    marker_id, next_markers, marker_trace = _queue_marker_for_item(item, markers, screen_signature)
    if marker_id is None:
        trace = dict(match_trace)
        trace.update(marker_trace)
        return None, markers, trace
    args = {
        "marker_id": marker_id,
        "queue_id": item.get("queue_id", ""),
        "target_label": item.get("title", ""),
        "target_role": "job_card",
        "target_component": "job_card_title",
        "reason": "검색 결과 카드 큐에서 다음 미방문 공고를 선택합니다.",
        "expected_after": "선택한 공고의 상세 페이지가 열린다.",
    }
    msg = AIMessage(
        content="[card_queue] cached next result card",
        tool_calls=[{"name": "click_marker", "args": args, "id": f"card_queue_{item.get('queue_id', 'next')}"}],
    )
    trace = {
        "hit": True,
        "queue_id": item.get("queue_id", ""),
        "title": item.get("title", ""),
        "return_match": match_trace,
        "marker": marker_trace,
    }
    return msg, next_markers, trace


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
        "state_key": state.get("reflex_state_key", "") or "",
        "url": current_url or state.get("current_url", "") or "",
        "screenshot": screenshot,
        "marked_image": state.get("marked_image", "") or "",
        "screen_signature": dict(state.get("screen_signature", {}) or {}),
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


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_icon_marker(marker: dict) -> bool:
    text = str(marker.get("text") or "")
    marker_type = str(marker.get("type") or "").strip().lower()
    return (
        marker_type == "icon"
        or text == "icon"
        or text.startswith("상호작용 가능한 요소 (")
        or text == "상호작용 가능한 요소"
    )


def _line_bbox(markers: list[dict]) -> list[int]:
    boxes = [marker_bbox(marker) for marker in markers]
    boxes = [box for box in boxes if box != [0, 0, 0, 0]]
    if not boxes:
        return [0, 0, 0, 0]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _join_line_marker_text(markers: list[dict]) -> str:
    ordered = sorted(markers, key=lambda marker: (marker_bbox(marker)[0], marker_bbox(marker)[1]))
    pieces: list[str] = []
    for marker in ordered:
        text = str(marker.get("text") or "").strip()
        if not text:
            continue
        pieces.append(text)
    joined = " ".join(pieces)
    joined = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", joined)
    joined = re.sub(r"([(\[\{])\s+", r"\1", joined)
    return re.sub(r"\s+", " ", joined).strip()


def _group_text_markers_into_lines(markers: list[dict]) -> list[dict]:
    text_markers = [
        marker
        for marker in markers
        if str(marker.get("text") or "").strip() and not _is_icon_marker(marker)
    ]
    if not text_markers:
        return []
    heights = sorted(max(1, marker_bbox(marker)[3] - marker_bbox(marker)[1]) for marker in text_markers)
    median_height = heights[len(heights) // 2] if heights else 16
    tolerance = max(8, min(24, int(median_height * 0.7)))
    lines: list[dict] = []
    for marker in sorted(text_markers, key=lambda item: ((marker_bbox(item)[1] + marker_bbox(item)[3]) / 2, marker_bbox(item)[0])):
        bbox = marker_bbox(marker)
        center_y = (bbox[1] + bbox[3]) / 2
        matched = None
        for line in lines:
            if abs(center_y - line["center_y"]) <= tolerance:
                matched = line
                break
        if matched is None:
            lines.append({"center_y": center_y, "markers": [marker]})
        else:
            matched["markers"].append(marker)
            count = len(matched["markers"])
            matched["center_y"] = ((matched["center_y"] * (count - 1)) + center_y) / count

    compacted: list[dict] = []
    for line in lines:
        ordered = sorted(line["markers"], key=lambda marker: marker_bbox(marker)[0])
        segments: list[list[dict]] = []
        current_segment: list[dict] = []
        previous_right: int | None = None
        max_inline_gap = max(160, int(median_height * 8))
        for marker in ordered:
            bbox = marker_bbox(marker)
            if current_segment and previous_right is not None and bbox[0] - previous_right > max_inline_gap:
                segments.append(current_segment)
                current_segment = [marker]
            else:
                current_segment.append(marker)
            previous_right = max(previous_right or bbox[2], bbox[2])
        if current_segment:
            segments.append(current_segment)

        for segment in segments:
            ids = [marker.get("id") for marker in segment if marker.get("id") is not None]
            text = _join_line_marker_text(segment)
            if text:
                compacted.append({"text": text, "ids": ids, "bbox": _line_bbox(segment)})
    return sorted(compacted, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _is_probable_detail_noise_line(line: dict) -> bool:
    bbox = line.get("bbox") or [0, 0, 0, 0]
    text = str(line.get("text") or "").strip()
    if not text:
        return True
    collapsed = re.sub(r"\s+", "", text)
    lowered = collapsed.lower()
    # 브라우저/사이트 헤더는 상세 본문 추출에는 잡음이지만 클릭 마커 목록에는 별도로 남긴다.
    if len(bbox) == 4 and bbox[1] < 120:
        return True
    browser_terms = ("youtube", "github", "gmail", "naver", "chzzk", "모든북마크")
    if sum(1 for term in browser_terms if term in lowered) >= 2:
        return True
    site_nav_terms = ("wanted", "채용", "이력서", "교육이벤트", "콘텐츠", "소셜", "프리랜서", "회원가입", "기업서비스")
    if ("wanted" in lowered or "원티드" in collapsed) and sum(1 for term in site_nav_terms if term in lowered) >= 2:
        return True
    if any(term in collapsed for term in ("상세정보더보기", "지원하기", "합격확률확인하기", "북마크", "공유하기")):
        return True
    if any(term in collapsed for term in ("회원가입/로그인", "회원가입로그인", "기업서비스", "합격확률", "이포지션나의합격확률은")):
        return True
    if len(bbox) == 4 and bbox[1] < 180 and any(
        term in collapsed
        for term in ("wanted", "원티드", "채용", "이력서", "교육이벤트", "콘텐츠", "소셜", "프리랜서", "회원가입", "기업서비스")
    ):
        return True
    if text in {"wanted", "원티드", "채용", "이력서", "교육·이벤트", "콘텐츠", "소셜", "프리랜서", "더보기"}:
        return True
    return False


def _append_limited_ocr_line(parts: list[str], index: int, line: dict, max_line_chars: int) -> None:
    text = str(line.get("text") or "").strip()
    if len(text) > max_line_chars:
        text = text[: max_line_chars - 1].rstrip() + "…"
    parts.append(f"{index}. {text}")


def _detail_action_marker_candidates(markers: list[dict], limit: int) -> list[dict]:
    primary_terms = (
        "상세 정보 더 보기",
        "더 보기",
        "더보기",
        "상세정보더보기",
    )
    primary: list[dict] = []
    seen: set[int] = set()
    for marker in sorted(markers, key=_marker_prompt_rank):
        marker_id = marker.get("id")
        if marker_id in seen:
            continue
        text = str(marker.get("text") or "").strip()
        collapsed = re.sub(r"\s+", "", text)
        bbox = marker_bbox(marker)
        if any(term.replace(" ", "") in collapsed for term in primary_terms):
            if "상세" in collapsed or (len(bbox) == 4 and bbox[1] > 240):
                primary.append(marker)
                seen.add(marker_id)
        if len(primary) >= limit:
            break
    return primary


def _detail_lightweight_marked_image_enabled() -> bool:
    return _env_enabled("VISION_DETAIL_LIGHTWEIGHT_MARKED_IMAGE_ENABLED", True)


def _draw_detail_lightweight_marker(draw: Any, marker: dict, color: tuple[int, int, int], font: Any) -> None:
    bbox = marker_bbox(marker)
    if bbox == [0, 0, 0, 0]:
        return
    x1, y1, x2, y2 = bbox
    pad = 4
    draw.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], outline=color, width=4)
    label = f"[{marker.get('id')}]"
    label_box = [x1 - pad, max(0, y1 - 30), x1 + 70, max(24, y1 - 4)]
    draw.rectangle(label_box, fill=color)
    draw.text((label_box[0] + 4, label_box[1] + 2), label, fill=(255, 255, 255), font=font)


def _build_detail_lightweight_marked_image(image_path: Any, markers: list[dict], current_url: str) -> str:
    """상세 페이지 reasoning에는 클릭 후보만 표시한 가벼운 화면 이미지를 사용한다."""
    if not current_url or not _looks_like_job_detail_url(current_url):
        return ""
    if not _detail_lightweight_marked_image_enabled():
        return ""
    try:
        from pathlib import Path
        from PIL import Image, ImageDraw, ImageFont

        source_path = Path(image_path)
        if not source_path.exists():
            return ""
        try:
            action_limit = int(os.getenv("VISION_DETAIL_ACTION_MARKER_LIMIT", "35"))
        except ValueError:
            action_limit = 35
        candidates = _detail_action_marker_candidates(markers, action_limit)

        image = Image.open(source_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        for marker in candidates:
            _draw_detail_lightweight_marker(draw, marker, (0, 120, 255), font)

        output_path = source_path.with_name(f"light_marked_{source_path.stem}.jpg")
        image.save(output_path, "JPEG", quality=88)
        logger.info(
            "Detail lightweight marked image prepared",
            markers_count=len(markers),
            highlighted_markers=len(candidates),
            output_path=str(output_path),
        )
        return str(output_path)
    except Exception as exc:
        logger.debug("detail lightweight marked image skipped", error=str(exc))
        return ""


def _build_detail_section_context(markers: list[dict]) -> str:
    try:
        min_text_markers = int(os.getenv("VISION_DETAIL_SECTION_MIN_TEXT_MARKERS", "120"))
        max_lines = int(os.getenv("VISION_DETAIL_OCR_MAX_LINES", "90"))
        max_line_chars = int(os.getenv("VISION_DETAIL_SECTION_MAX_LINE_CHARS", "180"))
        action_limit = int(os.getenv("VISION_DETAIL_ACTION_MARKER_LIMIT", "35"))
    except ValueError:
        min_text_markers = 120
        max_lines = 90
        max_line_chars = 180
        action_limit = 35

    text_marker_count = sum(1 for marker in markers if str(marker.get("text") or "").strip() and not _is_icon_marker(marker))
    if text_marker_count < min_text_markers:
        return ""

    lines = [line for line in _group_text_markers_into_lines(markers) if not _is_probable_detail_noise_line(line)]
    if not lines:
        return ""

    parts = ["상세 페이지 OCR 본문(읽기용, 위에서 아래 순서. 원본 마커는 클릭/좌표용으로 유지됨):"]
    shown_lines = lines[:max_lines]
    for index, line in enumerate(shown_lines, start=1):
        _append_limited_ocr_line(parts, index, line, max_line_chars)
    omitted_lines = max(0, len(lines) - len(shown_lines))
    if omitted_lines:
        parts.append(f"본문 압축으로 생략된 줄: {omitted_lines}개")

    action_markers = _detail_action_marker_candidates(markers, action_limit)
    if action_markers:
        parts.append("수집 진행용 클릭 후보:")
        for marker in action_markers:
            parts.append(f"[id: {marker.get('id')}] {marker.get('text', '')}")

    parts.append(f"원본 텍스트 마커 {text_marker_count}개를 읽기용 줄 {len(lines)}개로 압축")
    return "\n".join(parts)


def _detail_ocr_buffer_enabled() -> bool:
    return _env_enabled("VISION_DETAIL_OCR_BUFFER_ENABLED", True)


def _detail_buffer_line_key(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return text


def _detail_lines_for_buffer(markers: list[dict]) -> list[dict]:
    return [
        line
        for line in _group_text_markers_into_lines(markers)
        if not _is_probable_detail_noise_line(line)
    ]


def _new_detail_ocr_buffer(current_url: str) -> dict[str, Any]:
    return {
        "url": current_url,
        "lines": [],
        "seen_keys": [],
        "screens": [],
        "stats": {
            "screen_count": 0,
            "added_lines_last_screen": 0,
            "duplicate_lines_last_screen": 0,
            "total_lines": 0,
        },
    }


def _update_detail_ocr_buffer(
    existing: dict[str, Any] | None,
    markers: list[dict],
    current_url: str,
    image_path: Any = "",
) -> dict[str, Any]:
    """상세 페이지 OCR 본문 줄을 URL 단위 버퍼에 누적한다."""
    if not _detail_ocr_buffer_enabled():
        return dict(existing or {})
    if not current_url or not _looks_like_job_detail_url(current_url):
        return dict(existing or {})

    try:
        max_lines = int(os.getenv("VISION_DETAIL_OCR_BUFFER_MAX_LINES", "260"))
        max_line_chars = int(os.getenv("VISION_DETAIL_OCR_BUFFER_MAX_LINE_CHARS", "220"))
    except ValueError:
        max_lines = 260
        max_line_chars = 220

    buffer = dict(existing or {})
    if buffer.get("url") != current_url:
        buffer = _new_detail_ocr_buffer(current_url)
    lines = [dict(item) for item in (buffer.get("lines") or []) if isinstance(item, dict)]
    seen_keys = [str(item) for item in (buffer.get("seen_keys") or []) if str(item)]
    seen = set(seen_keys)
    added = 0
    duplicate = 0
    screen_name = ""
    try:
        from pathlib import Path

        screen_name = Path(image_path).name if image_path else ""
    except Exception:
        screen_name = str(image_path or "")

    for line in _detail_lines_for_buffer(markers):
        text = str(line.get("text") or "").strip()
        if len(text) < 2:
            continue
        if len(text) > max_line_chars:
            text = text[: max_line_chars - 1].rstrip() + "…"
        key = _detail_buffer_line_key(text)
        if not key:
            continue
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        seen_keys.append(key)
        lines.append(
            {
                "text": text,
                "bbox": line.get("bbox") or [0, 0, 0, 0],
                "first_screen": screen_name,
            }
        )
        added += 1
        if len(lines) >= max_lines:
            break

    screens = [str(item) for item in (buffer.get("screens") or []) if str(item)]
    if screen_name and (not screens or screens[-1] != screen_name):
        screens.append(screen_name)
    stats = dict(buffer.get("stats") or {})
    stats["screen_count"] = int(stats.get("screen_count") or 0) + 1
    stats["added_lines_last_screen"] = added
    stats["duplicate_lines_last_screen"] = duplicate
    stats["total_lines"] = len(lines)
    buffer.update(
        {
            "url": current_url,
            "lines": lines[:max_lines],
            "seen_keys": seen_keys[:max_lines],
            "screens": screens[-20:],
            "stats": stats,
        }
    )
    logger.info(
        "Detail OCR buffer updated",
        url=current_url,
        added_lines=added,
        duplicate_lines=duplicate,
        total_lines=len(lines[:max_lines]),
        screen_count=stats["screen_count"],
    )
    return buffer


def _compact_detail_ocr_buffer_context(state: GraphState, current_url: str) -> str:
    if not _detail_ocr_buffer_enabled() or not current_url or not _looks_like_job_detail_url(current_url):
        return ""
    buffer = dict(state.get("detail_ocr_buffer", {}) or {})
    if buffer.get("url") != current_url:
        return ""
    stats = dict(buffer.get("stats") or {})
    lines = [item for item in (buffer.get("lines") or []) if isinstance(item, dict)]
    preview = [str(item.get("text") or "").strip() for item in lines[-8:]]
    preview = [line for line in preview if line]
    parts = [
        "상세 OCR 누적 상태:",
        f"- 누적 본문 줄 수: {len(lines)}",
        f"- 이번 화면 새 줄 수: {stats.get('added_lines_last_screen', 0)}",
        f"- 이번 화면 중복 줄 수: {stats.get('duplicate_lines_last_screen', 0)}",
        f"- 상세 화면 관찰 횟수: {stats.get('screen_count', 0)}",
        "- 상세 페이지에서는 중간 DB 추출을 위해 update_extracted_info를 호출하지 마십시오.",
        "- 더 읽어야 하면 scroll 또는 보이는 상세 펼치기 버튼 클릭을 선택하십시오.",
        "- 현재 공고 정보가 충분하면 finish_detail_reading(page_role=\"job_detail\", detail_complete=true)을 호출하십시오.",
    ]
    if preview:
        parts.append("- 최근 누적 본문 미리보기: " + json.dumps(preview, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(parts) + "\n\n"


def _detail_buffer_text(buffer: dict[str, Any]) -> str:
    try:
        max_chars = int(os.getenv("VISION_DETAIL_FINAL_OCR_MAX_CHARS", "16000"))
    except ValueError:
        max_chars = 16000
    lines = [item for item in (buffer.get("lines") or []) if isinstance(item, dict)]
    rendered: list[str] = []
    total = 0
    for index, line in enumerate(lines, start=1):
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        row = f"{index}. {text}"
        if total + len(row) + 1 > max_chars:
            break
        rendered.append(row)
        total += len(row) + 1
    return "\n".join(rendered)


def _extract_job_from_detail_ocr_buffer(state: GraphState, current_url: str) -> dict[str, Any]:
    buffer = dict(state.get("detail_ocr_buffer", {}) or {})
    ocr_text = _detail_buffer_text(buffer)
    if not ocr_text.strip():
        return {}
    active_card = dict(state.get("active_result_card", {}) or {})
    messages = [
        SystemMessage(
            content=(
                "누적 OCR 본문에서 채용공고 1건을 JobPosting 스키마로 정리하십시오. "
                "OCR에 없는 사실은 만들지 말고, 알 수 없는 필드는 비우십시오. "
                "현재 상세 URL은 보존하십시오."
            )
        ),
        HumanMessage(
            content=json.dumps(
                {
                    "current_url": current_url,
                    "active_result_card": {
                        "title": active_card.get("title") or active_card.get("target_label") or "",
                        "company": active_card.get("company") or "",
                    },
                    "ocr_text": ocr_text,
                },
                ensure_ascii=False,
                indent=2,
            )
        ),
    ]
    start = time.time()
    extracted = dump_model(_get_detail_extraction_llm().invoke(messages))
    logger.info(
        "Detail OCR final extraction completed",
        duration=f"{time.time() - start:.2f}s",
        model=_detail_extraction_model_spec(),
        ocr_chars=len(ocr_text),
        ocr_lines=len(buffer.get("lines") or []),
    )
    extracted = {key: value for key, value in extracted.items() if value not in (None, "", [], {})}
    if current_url and not extracted.get("url"):
        extracted["url"] = current_url
    if active_card.get("title") and not (extracted.get("position") or extracted.get("직무명")):
        extracted["position"] = active_card.get("title")
    if active_card.get("company") and not (extracted.get("company_name") or extracted.get("회사명")):
        extracted["company_name"] = active_card.get("company")
    return extracted


def _build_ui_context(markers: list[dict], current_url: str = "") -> str:
    if current_url and _looks_like_job_detail_url(current_url) and _env_enabled("VISION_DETAIL_SECTION_CONTEXT_ENABLED", True):
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


def _build_reasoning_messages(state: GraphState, loop_warning: str) -> list:
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
        transition_context = (
            "직전 화면 전환 검증:\n"
            f"- status: {state.get('transition_status')}\n"
            f"- outcome: {state.get('transition_outcome') or '(없음)'}\n"
            f"- source: {state.get('transition_source') or '(없음)'}\n\n"
        )
    forbidden_action_context = _build_forbidden_action_context(action_history)
    if forbidden_action_context:
        forbidden_action_context += "\n\n"

    human_prompt_text = (
        f"{plan_context}"
        f"{_compact_extracted_context(extracted_jd, current_url)}"
        f"현재 브라우저 URL:\n{current_url or '(확인 안 됨)'}\n\n"
        f"{collection_context}"
        f"{_compact_result_card_queue_context(state)}"
        f"{_compact_detail_ocr_buffer_context(state, current_url)}"
        f"{transition_context}"
        f"현재 화면 상태 (UI 마커):\n{ui_context + loop_warning}\n\n"
        f"{forbidden_action_context}"
        f"{_compact_recent_actions_context(action_history)}"
        f"다음 행동을 결정하세요. 상세 페이지에서 OCR 버퍼가 활성화되어 있으면 중간 정보 추출 대신 finish_detail_reading으로 읽기 종료를 알리고, "
        f"그 외 화면에서 새로운 정보가 식별되었다면 update_extracted_info를 먼저 부르고, "
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
        "reflex_trace": {"hit": False, "source": "reasoning"},
        "reflex_transition_contracts": {},
        "step_durations": [{"node": "reasoning", "duration": elapsed}]
    }
    if error_increment > 0:
        result["error_count"] = state.get("error_count", 0) + error_increment

    return result


def _reflex_action_args(step: dict, marker_id: int | None, params: dict | None = None) -> dict | None:
    """저장된 RecipeStep dict를 action_node tool args 형태로 변환한다."""
    action = step.get("action")
    param = dict(step.get("param") or {})
    value = step.get("value")
    params = dict(params or {})
    trace_args = _reflex_trace_args(step)

    if action == "click_marker":
        if marker_id is None:
            return None
        return {"marker_id": marker_id, **trace_args}
    if action == "type_in_marker":
        if marker_id is None:
            return None
        slot_name = param.get("slot_name") or param.get("slot") or ""
        if not slot_name:
            slot_refs = step.get("slot_refs") or []
            slot_name = slot_refs[0] if slot_refs else ""
        text = params.get(slot_name) if slot_name else None
        text = text or param.get("text") or value
        if not text:
            return None
        args = {"marker_id": marker_id, "text": text, **trace_args}
        if slot_name:
            args["slot_name"] = slot_name
        return args
    if action == "scroll":
        return {"direction": param.get("direction") or value or "down", **trace_args}
    if action == "press_key":
        key = param.get("key") or value
        return {"key": key, **trace_args} if key else None
    if action == "go_back":
        return dict(trace_args)
    return None


def _reflex_trace_args(step: dict) -> dict:
    """재생 액션 실행에는 영향 없는 추적 메타데이터만 도구 인자에 복원한다."""
    out: dict[str, str] = {}
    mapping = {
        "intent": "reason",
        "target_role": "target_role",
        "component": "target_component",
        "expected_after": "expected_after",
    }
    for source_key, arg_key in mapping.items():
        value = step.get(source_key)
        if value:
            out[arg_key] = str(value)
    target = step.get("target") if isinstance(step.get("target"), dict) else {}
    semantic_label = target.get("semantic_label")
    if semantic_label:
        out["target_label"] = str(semantic_label)
    return out


def _missing_required_recipe_inputs(recipe, params: dict) -> list[str]:
    """레시피 메타데이터(skill_metadata)의 필수 입력 누락 여부를 확인한다."""
    metadata = getattr(recipe, "skill_metadata", None)
    inputs = getattr(metadata, "inputs", []) if metadata is not None else []
    missing: list[str] = []
    for item in inputs or []:
        name = getattr(item, "name", "")
        required = bool(getattr(item, "required", False))
        value = params.get(name)
        if required and name and (value is None or value == ""):
            missing.append(name)
    return missing


def _recipe_task_category(recipe: Any) -> str:
    """레시피 메타데이터에 저장된 작업 카테고리를 읽는다."""

    from agent.recipe.task_category import normalize_task_category

    metadata = getattr(recipe, "skill_metadata", None)
    return normalize_task_category(getattr(metadata, "task_category", "") if metadata is not None else "")


def _recipe_matches_task_category(recipe: Any, params: dict) -> bool:
    """요청 작업 카테고리와 레시피 카테고리가 맞는지 확인한다."""

    from agent.recipe.task_category import task_category_matches

    return task_category_matches(params.get("task_category"), _recipe_task_category(recipe))


def reflex_node(state: GraphState) -> Dict[str, Any]:
    """캐시된 Reflex Recipe가 있으면 reasoning을 우회해 같은 tool_call 형태를 만든다."""
    start_time = time.time()
    logger.info("Executing Reflex Node")

    def miss(state_key: str, elapsed: float, reason: str = "", trace: dict | None = None) -> Dict[str, Any]:
        reflex_trace = dict(trace or {})
        reflex_trace.update(
            {
                "hit": False,
                "state_key": state_key,
                "reason": reason or reflex_trace.get("reason", ""),
            }
        )
        return {
            "reflex_state_key": state_key,
            "reflex_hit": False,
            "reflex_trace": reflex_trace,
            "reflex_transition_contracts": {},
            "step_durations": [{"node": "reflex", "duration": elapsed, "hit": False, "reason": reason}],
        }

    try:
        from agent.recipe.matcher import is_replayable_step
        from agent.recipe.phash_replay import match_step_by_screen_signature
        from agent.recipe.state_key import compute_state_key
        from agent.recipe.store import RecipeStore

        markers = state.get("current_markers", []) or []
        state_key = state.get("reflex_state_key") or compute_state_key(state.get("current_url", ""), markers)
        params = dict(state.get("recipe_params", {}) or {})
        params.setdefault("goal", state.get("goal", ""))
        requested_task_category = str(params.get("task_category") or "").strip()
        site = str(params.get("site") or "").strip()
        store = RecipeStore()
        recent_images = state.get("recent_images", []) or []
        current_image_path = str(recent_images[-1]) if recent_images else ""
        recipe_candidates = []
        task_category_skips = 0
        exact_recipe = store.get_recipe(state_key, site=site or None, task_category=requested_task_category or None)
        if exact_recipe and exact_recipe.steps:
            if _recipe_matches_task_category(exact_recipe, params):
                recipe_candidates.append((state_key, exact_recipe, 1.0, "exact"))
            else:
                task_category_skips += 1

        def broad_site_recipe_allowed(recipe: Any) -> bool:
            steps = list(getattr(recipe, "steps", []) or [])
            if not steps:
                return False
            first_step = dump_model(steps[0])
            if first_step.get("action") not in {"click_marker", "type_in_marker"}:
                return False
            return bool(first_step.get("roi_signature"))

        if site:
            for recipe_key, recipe in store.get_site_recipes(site, task_category=requested_task_category or None):
                if not _recipe_matches_task_category(recipe, params):
                    task_category_skips += 1
                    continue
                if recipe_key != state_key and broad_site_recipe_allowed(recipe):
                    recipe_candidates.append((recipe_key, recipe, 0.0, "site"))

        if not recipe_candidates:
            elapsed = time.time() - start_time
            logger.info("Reflex miss: no recipe", state_key=state_key[:24], site=site)
            return miss(
                state_key,
                elapsed,
                "no_recipe",
                {
                    "candidate_count": 0,
                    "site": site,
                    "task_category": requested_task_category,
                    "task_category_skips": task_category_skips,
                    "current_phash": (state.get("screen_signature") or {}).get("phash", ""),
                },
            )

        selected = None
        candidate_traces: list[dict[str, Any]] = []
        for recipe_key, recipe, similarity, lookup in recipe_candidates:
            candidate_trace: dict[str, Any] = {
                "recipe_key": recipe_key,
                "lookup": lookup,
                "similarity": similarity,
                "task_category": _recipe_task_category(recipe),
                "steps": [],
            }
            missing_inputs = _missing_required_recipe_inputs(recipe, params)
            if missing_inputs:
                candidate_trace["accepted"] = False
                candidate_trace["reason"] = "missing_required_inputs"
                candidate_trace["missing_inputs"] = missing_inputs
                candidate_traces.append(candidate_trace)
                logger.info(
                    "Reflex candidate skipped: missing required inputs",
                    recipe_key=recipe_key[:24],
                    missing=missing_inputs,
                )
                continue

            tool_calls = []
            transition_contracts: dict[str, dict] = {}
            tool_call_traces: dict[str, dict[str, Any]] = {}
            candidate_valid = True
            for idx, recipe_step in enumerate(recipe.steps):
                step = dump_model(recipe_step)
                action = step.get("action")
                marker_id = None
                step_trace: dict[str, Any] = {
                    "seq": step.get("seq"),
                    "action": action,
                    "replay_mode": step.get("replay_mode"),
                    "match_mode": "none",
                    "target_text": ((step.get("target") or {}).get("text") if isinstance(step.get("target"), dict) else ""),
                }
                if not is_replayable_step(step, params=params):
                    step_trace["accepted"] = False
                    step_trace["reason"] = "not_replayable"
                    candidate_trace["steps"].append(step_trace)
                    candidate_valid = False
                    break
                if action not in {"click_marker", "type_in_marker"}:
                    step_trace["accepted"] = False
                    step_trace["reason"] = "non_roi_action"
                    candidate_trace["steps"].append(step_trace)
                    candidate_valid = False
                    break
                step_trace["match_mode"] = "roi_phash"
                marker_id, phash_result = match_step_by_screen_signature(
                    step,
                    dict(state.get("screen_signature", {}) or {}),
                    markers,
                    current_image_path=current_image_path,
                )
                step_trace["phash"] = phash_result
                step_trace["match_mode"] = phash_result.get("mode") or step_trace["match_mode"]
                if marker_id is None:
                    logger.info(
                        "Reflex candidate skipped: ROI replay check failed",
                        recipe_key=recipe_key[:24],
                        reason=phash_result.get("reason"),
                        distance=phash_result.get("distance"),
                    )
                    step_trace["accepted"] = False
                    step_trace["reason"] = phash_result.get("reason", "phash_check_failed")
                    candidate_trace["steps"].append(step_trace)
                    candidate_valid = False
                    break
                step_trace["marker_id"] = marker_id
                args = _reflex_action_args(step, marker_id, params=params)
                if args is None:
                    step_trace["accepted"] = False
                    step_trace["reason"] = "args_build_failed"
                    candidate_trace["steps"].append(step_trace)
                    candidate_valid = False
                    break
                call_id = f"reflex_{abs(hash(recipe_key))}_{idx}"
                tool_calls.append({"name": action, "args": args, "id": call_id})
                step_trace["accepted"] = True
                step_trace["tool_call_id"] = call_id
                tool_call_traces[call_id] = dict(step_trace)
                candidate_trace["steps"].append(step_trace)
                contract = step.get("transition_contract")
                if contract:
                    transition_contracts[call_id] = dict(contract)
            if candidate_valid and tool_calls:
                candidate_trace["accepted"] = True
                candidate_trace["reason"] = "matched"
                candidate_traces.append(candidate_trace)
                selected = (recipe_key, recipe, similarity, lookup, tool_calls, transition_contracts, tool_call_traces)
                break
            candidate_trace.setdefault("accepted", False)
            candidate_trace.setdefault("reason", "candidate_invalid")
            candidate_traces.append(candidate_trace)

        if selected is None:
            elapsed = time.time() - start_time
            logger.info(
                "Reflex miss: no candidate passed marker matching",
                state_key=state_key[:24],
                candidates=len(recipe_candidates),
            )
            return miss(
                state_key,
                elapsed,
                "no_candidate_passed",
                {"candidate_count": len(recipe_candidates), "candidates": candidate_traces},
            )

        recipe_key, recipe, similarity, lookup, tool_calls, transition_contracts, tool_call_traces = selected

        msg = AIMessage(
            content=f"[reflex] cached {len(tool_calls)} action(s)",
            tool_calls=tool_calls,
        )
        elapsed = time.time() - start_time
        logger.info(
            "Reflex hit",
            state_key=state_key[:24],
            recipe_key=recipe_key[:24],
            lookup=lookup,
            similarity=f"{similarity:.3f}",
            actions=[call["name"] for call in tool_calls],
            transition_contracts=len(transition_contracts),
            when_to_use=getattr(getattr(recipe, "skill_metadata", None), "when_to_use", "")[:80],
            duration=f"{elapsed:.3f}s",
        )
        return {
            "last_action_result": msg,
            "reflex_state_key": state_key,
            "reflex_hit": True,
            "reflex_trace": {
                "hit": True,
                "state_key": state_key,
                "recipe_key": recipe_key,
                "lookup": lookup,
                "similarity": similarity,
                "candidate_count": len(recipe_candidates),
                "task_category": requested_task_category,
                "actions": [call["name"] for call in tool_calls],
                "tool_calls": tool_call_traces,
                "candidates": candidate_traces,
            },
            "reflex_transition_contracts": transition_contracts,
            "step_durations": [
                {
                    "node": "reflex",
                    "duration": elapsed,
                    "hit": True,
                    "lookup": lookup,
                    "candidate_count": len(recipe_candidates),
                    "actions": [call["name"] for call in tool_calls],
                }
            ],
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.debug("reflex node skipped", error=str(e))
        return miss(state.get("reflex_state_key", ""), elapsed, "exception", {"error": str(e)})


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
        result = {
            "action": "set_result_card_queue",
            "status": "success" if queue else "skipped",
            "result": f"Result card queue stored: {len(queue)} card(s)." if queue else "No valid visible result cards were queued.",
            "queued_count": len(queue),
            "queued_titles": [item.get("title", "") for item in queue],
            "_result_card_queue": queue,
            "_result_page_memory": memory,
        }
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

    def set_pending_transition(
        action_seq: int,
        action_name: str,
        args: dict,
        contract: dict | None,
        source: str,
    ) -> None:
        nonlocal pending_transition
        pending_transition = {
            "action_seq": action_seq,
            "action": action_name,
            "expected_after": str(args.get("expected_after") or ""),
            "source": source,
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
        result["state_key"] = before_snapshot.get("state_key", "")
        result["before_state_key"] = before_snapshot.get("state_key", "")
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
                result["reflex_lookup"] = trace.get("lookup", "")
                result["reflex_similarity"] = trace.get("similarity")
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
        step_elapsed = time.time() - step_start
        step_durations.append({"node": f"action ({action_name})", "duration": step_elapsed})
        logger.warning(message, action=action_name, reason=reason)

    def append_policy_ui_action(action_name: str, args: dict, reason: str) -> dict:
        nonlocal current_url_stale, screen_changed, pending_transition
        step_start = time.time()
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
        step_elapsed = time.time() - step_start
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

    UI_ACTIONS    = {"click_marker", "type_in_marker", "scroll", "press_key", "open_browser", "close_browser", "go_back"}
    SCREEN_CHANGING_ACTIONS = {"click_marker", "type_in_marker", "scroll", "press_key", "open_browser", "close_browser", "go_back"}
    URL_STALE_ACTIONS = {"click_marker", "press_key", "open_browser", "close_browser", "go_back"}
    OBSERVATION_REQUIRED_ACTIONS = {"click_marker", "press_key", "open_browser", "go_back"}
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
        step_start = time.time()
        before_snapshot = _state_snapshot_for_action(state, current_url)
        before_state_key = before_snapshot.get("state_key", "")
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
                sensitive_reason = _sensitive_action_reason(
                    {**state, "current_markers": latest_markers},
                    action_name,
                    args,
                )
                if sensitive_reason:
                    request_human_approval(action_name, args, sensitive_reason, before_snapshot, step_start)
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

                if _same_state_action_seen(state, action_name, args, before_state_key):
                    append_guard_result(
                        action_name,
                        args,
                        before_snapshot,
                        "skipped",
                        "same_state_repeat_blocked",
                        "Blocked repeated UI action in the same screen state.",
                        step_start,
                        observation_required=False,
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
                    set_pending_transition(
                        action_seq,
                        action_name,
                        args,
                        contract,
                        transition_source,
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
                        if target_count > 0 and collected_count < target_count:
                            policy_ui_action = (
                                "go_back",
                                {
                                    "reason": "detail page complete and more result cards remain",
                                    "expected_after": "job result list is visible",
                                },
                                "detail_complete_more_items",
                            )
                if (
                    action_name in {"update_extracted_info", "finish_detail_reading"}
                    and result.get("status") == "success"
                    and _auto_finish_on_target_enabled()
                    and args.get("detail_complete") is not False
                ):
                    target_count = _target_count_from_state(state)
                    collected_count = _extracted_job_count(current_jd)
                    if target_count > 0 and collected_count >= target_count:
                        is_finished = True
                        collected_data.append(
                            f"Auto-finished after collecting target_count={target_count} jobs."
                        )
                        result["auto_finished"] = True
                        result["target_count"] = target_count
                        result["collected_count"] = collected_count

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

            step_elapsed = time.time() - step_start
            step_durations.append({"node": f"action ({action_name})", "duration": step_elapsed})
            logger.info(f"Action Node [{action_name}] completed in {step_elapsed:.2f} seconds")

            if policy_ui_action and not is_finished:
                policy_action, policy_args, policy_reason = policy_ui_action
                append_policy_ui_action(policy_action, policy_args, policy_reason)
                break

            if is_finished:
                break

        except Exception as e:
            logger.error(f"Failed to execute action {action_name}", error=str(e))
            step_elapsed = time.time() - step_start
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
        "active_result_card": active_result_card,
        "queue_replay_hit": False,
        "queue_replay_trace": {},
        "detail_ocr_buffer": detail_ocr_buffer,
        "recorded_steps":    recorded_steps,
        "feedback_episodes": feedback_episodes,
        "pending_human_approval": pending_human_approval,
        "human_approval_request": human_approval_request,
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

    if os.getenv("COMMANDER_GRAPH_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
        from agent.graph.commander_workflow import run_commander_graph

        commander_result = run_commander_graph(query)
        elapsed = time.time() - start_time
        return {
            "last_action_result": commander_result.get("final_answer", ""),
            "is_finished": True,
            "step_durations": [{"node": "qa_commander_graph", "duration": elapsed}],
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
                elif tool_name == "review_recipe_candidates":
                    result_str = review_recipe_candidates.invoke(tool_args)
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

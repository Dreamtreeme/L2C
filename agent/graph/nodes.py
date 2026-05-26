import os
import json
import base64
import time
from typing import Dict, Any, List, Optional

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from agent.graph.state import GraphState
from agent.tools.perception import PerceptionEngine
from agent.tools.actions import ActionTools
from agent.prompts.commander import commander_prompt, COMMANDER_SYSTEM_PROMPT, QA_COMMANDER_SYSTEM_PROMPT
from agent.utils.logger import logger
from agent.tools.sqlite_query import sqlite_query
from agent.tools.realtime_scraping import realtime_scraping

# 싱글톤으로 도구 유지
perception = PerceptionEngine()
action_tools = ActionTools(perception)

# --- LLM 도구 정의용 Pydantic 모델 ---
class click_marker(BaseModel):
    """화면의 특정 ID 마커를 클릭합니다."""
    marker_id: int = Field(..., description="클릭할 마커의 ID")

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

class get_credentials(BaseModel):
    """특정 사이트(예: 'wanted')의 ID/PW를 보안 저장소에서 가져와 반환합니다. 로그인 폼이 보일 때 호출하세요."""
    site: str = Field(..., description="자격 증명을 가져올 사이트 식별자 (예: 'wanted')")

class update_extracted_info(BaseModel):
    """현재 화면에서 식별한 채용 공고 정보를 수집 상태에 누적 업데이트합니다. 스크롤하면서 새로운 정보를 찾을 때마다 이 도구를 호출하여 정보를 보존해 두세요. (예: {'회사명': '로이드케이', '주요업무': ['A', 'B']} 형태의 JSON 문자열)"""
    data_json: str = Field(..., description="업데이트할 정보 키-값 딕셔너리의 JSON 문자열")

class go_back(BaseModel):
    """브라우저의 뒤로가기(이전 페이지 이동) 기능을 실행합니다. Alt + Left Arrow 단축키를 시뮬레이션합니다."""
    pass

class update_plan_progress(BaseModel):
    """현재 실행 중인 계획 단계를 업데이트하거나 필요시 계획을 수정합니다."""
    current_step: int = Field(..., description="수행 중인 계획 단계 인덱스 (0-indexed)")
    plan: Optional[List[str]] = Field(None, description="수정된 계획 단계 목록 (필요한 경우)")

class finish_task(BaseModel):
    """작업을 완료하고 최종 데이터를 반환합니다."""
    result: str = Field(..., description="최종 완료 요약 또는 결과 데이터")


def perception_node(state: GraphState) -> Dict[str, Any]:
    """화면을 캡처하고 마커를 파싱하여 상태를 업데이트합니다."""
    start_time = time.time()
    logger.info("Executing Perception Node")
    
    # 화면 캡처
    image_path = perception.capture_screen()
    
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
    
    # 텍스트가 있는 요소와 없는 요소를 구분하여 프롬프트 토큰 최적화 (지연 시간 절감)
    text_elements = []
    icon_ids = []
    for m in markers:
        text = m['text']
        if text.startswith("상호작용 가능한 요소 (") or text == "상호작용 가능한 요소":
            icon_ids.append(m['id'])
        else:
            text_elements.append(f"[id: {m['id']}] {text}")
            
    ui_context = ""
    if text_elements:
        ui_context += "식별된 텍스트 요소:\n" + "\n".join(text_elements) + "\n"
    if icon_ids:
        ui_context += f"기타 아이콘/버튼 마커 ID 목록: {icon_ids}"
    if not ui_context:
        ui_context = "발견된 UI 마커 없음"
    
    elapsed = time.time() - start_time
    logger.info(f"Perception Node completed in {elapsed:.2f} seconds")
    return {
        "recent_images": [image_path],
        "marked_image": marked_image,
        "current_markers": markers,
        "ui_context": ui_context,
        "step_durations": [{"node": "perception", "duration": elapsed}]
    }


# 글로벌 모델 초기화로 커넥션 풀링 유지 (TCP/SSL 핸드셰이크 레이턴시 절감)
# 브라우저 자동화용 (temperature=0.1, 브라우저 조작 도구 바인딩)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)
llm_with_tools = llm.bind_tools([
    click_marker,
    type_in_marker,
    scroll,
    press_key,
    open_browser,
    get_credentials,
    update_extracted_info,
    go_back,
    update_plan_progress,
    finish_task
])

# QA 지휘자용 (temperature=0.0, DB조회/실시간수집 도구 바인딩)
qa_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
qa_llm_with_tools = qa_llm.bind_tools([sqlite_query, realtime_scraping])


def reasoning_node(state: GraphState) -> Dict[str, Any]:
    """Gemini Flash를 호출하여 다음 행동을 결정합니다."""
    start_time = time.time()
    logger.info("Executing Reasoning Node")
    
    # 루프 감지 로직
    action_history = state.get("action_history", [])
    ui_context = state.get("ui_context", "")
    loop_warning = ""
    error_increment = 0
    
    if len(action_history) >= 3:
        last_3 = action_history[-3:]
        # 각 액션의 대표값 (action 종류 + args의 json 문자열) 추출
        actions_set = set(
            (a.get("action"), json.dumps(a.get("args", {}), sort_keys=True)) 
            for a in last_3 if isinstance(a, dict)
        )
        if len(actions_set) == 1:
            repeated = last_3[-1]
            logger.warning(f"Loop detected! Repeated action: {repeated.get('action')} with args: {repeated.get('args')}")
            loop_warning = f"\n\n[경고: 무한 루프 감지됨] 당신은 직전 3회 동안 동일한 행동({repeated.get('action')}: {repeated.get('args')})을 반복했습니다. 절대 동일한 행동(동일 마커 클릭 등)을 다시 수행하지 마십시오. 새로운 마커를 클릭하거나, 스크롤을 하거나, 다른 방식으로 목표를 해결해야 합니다."
            
            # 4회 이상 반복 시 에러 카운트 증가를 통한 자동 중단 유도
            if len(action_history) >= 4:
                last_4 = action_history[-4:]
                actions_set_4 = set(
                    (a.get("action"), json.dumps(a.get("args", {}), sort_keys=True)) 
                    for a in last_4 if isinstance(a, dict)
                )
                if len(actions_set_4) == 1:
                    logger.error("Persistent loop detected. Increasing error count to terminate.")
                    error_increment = 1
                    
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
    human_prompt_text = (
        f"{plan_context}"
        f"현재까지 누적 수집된 정보:\n{json.dumps(extracted_jd, ensure_ascii=False, indent=2)}\n\n"
        f"현재 화면 상태 (UI 마커):\n{ui_context + loop_warning}\n\n"
        f"이전 행동 내역:\n{json.dumps(action_history[-5:], ensure_ascii=False, indent=2)}\n\n"
        f"다음 행동을 결정하세요. 새로운 정보가 식별되었다면 update_extracted_info를 먼저 부르고, "
        f"계획 단계 전환이 일어났다면 update_plan_progress를 함께 체이닝 호출하여 계획 진행률을 반영하십시오."
    )
    
    # 마킹 이미지 로드 및 Base64 인코딩 (VLM 리사이징 & JPEG 압축 최적화)
    marked_image_path = state.get("marked_image")
    base64_image = ""
    if marked_image_path and os.path.exists(marked_image_path):
        try:
            from PIL import Image
            from io import BytesIO
            with Image.open(marked_image_path) as img:
                # VLM 최적화를 위해 최대 해상도를 1024px로 축소 (Gemini 3.5 Flash 권장 사양)
                width, height = img.size
                max_dim = 1024
                if width > max_dim or height > max_dim:
                    ratio = max_dim / max(width, height)
                    new_w = int(width * ratio)
                    new_h = int(height * ratio)
                    # 고품질 LANCZOS 대신 속도가 빠른 BILINEAR로 고속 리사이징
                    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                
                # PNG 대신 압축률이 높은 JPEG(퀄리티 70)로 변환하여 페이로드 크기 75% 이상 감축
                buffered = BytesIO()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(buffered, format="JPEG", quality=70)
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as img_err:
            logger.warning("Failed to read/resize marked_image for reasoning node", error=str(img_err))
            
    if base64_image:
        logger.info("Invoking reasoning node with multimodal SoM marked image...")
        messages = [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=[
                {"type": "text", "text": human_prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ])
        ]
    else:
        logger.info("Invoking reasoning node with text-only prompts...")
        messages = [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=human_prompt_text)
        ]
    
    # LLM 추론
    response = llm_with_tools.invoke(messages)
    
    elapsed = time.time() - start_time
    logger.info(f"Reasoning Node completed in {elapsed:.2f} seconds")
    
    # 결과를 State에 임시 저장 및 에러 카운트 업데이트
    result = {
        "last_action_result": response,
        "step_durations": [{"node": "reasoning", "duration": elapsed}]
    }
    if error_increment > 0:
        result["error_count"] = state.get("error_count", 0) + error_increment
        
    return result


def action_node(state: GraphState) -> Dict[str, Any]:
    """Reasoning Node가 선택한 도구(들)를 순차적으로 실행(Action Chaining)합니다."""
    start_time = time.time()
    logger.info("Executing Action Node (with potential Action Chaining)")
    
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
    current_jd = dict(state.get("extracted_jd", {}))
    is_finished = state.get("is_finished", False)
    collected_data = list(state.get("collected_data", []))
    error_count = state.get("error_count", 0)
    step_durations = []
    current_plan_step = state.get("current_plan_step", 0)
    current_plan = list(state.get("plan", []))
    
    # 헬퍼 함수: marker_id -> bbox 매핑
    def get_bbox(marker_id: int):
        for m in state.get("current_markers", []):
            if m["id"] == marker_id:
                return m["bbox"]
        raise ValueError(f"Marker ID {marker_id} not found in current screen.")
        
    for idx, tool_call in enumerate(ai_msg.tool_calls):
        action_name = tool_call["name"]
        args = tool_call["args"]
        
        logger.info(f"LLM decided to call (chained {idx+1}/{len(ai_msg.tool_calls)}): {action_name} with args: {args}")
        step_start = time.time()
        
        try:
            if action_name == "click_marker":
                result = action_tools.click_marker(get_bbox(args["marker_id"]))
            elif action_name == "type_in_marker":
                result = action_tools.type_in_marker(get_bbox(args["marker_id"]), args["text"])
            elif action_name == "scroll":
                result = action_tools.scroll(direction=args.get("direction", "down"))
            elif action_name == "press_key":
                result = action_tools.press_key(args["key"])
            elif action_name == "open_browser":
                result = action_tools.open_browser(args["url"])
            elif action_name == "get_credentials":
                result = action_tools.get_credentials(args["site"])
            elif action_name == "go_back":
                result = action_tools.go_back()
            elif action_name == "update_plan_progress":
                current_plan_step = args["current_step"]
                if args.get("plan") is not None:
                    current_plan = args["plan"]
                result = {
                    "action": "update_plan_progress",
                    "status": "success",
                    "result": f"Plan progress updated. Current step index: {current_plan_step}",
                    "args": args
                }
            elif action_name == "update_extracted_info":
                try:
                    new_data = json.loads(args["data_json"])
                    current_jd.update(new_data)
                    result_str = f"Extracted data updated with: {new_data}"
                except Exception as e:
                    result_str = f"Failed to parse data_json: {e}"
                
                result = {
                    "action": "update_extracted_info",
                    "status": "success" if "Failed" not in result_str else "error",
                    "result": result_str,
                    "args": args
                }
            elif action_name == "finish_task":
                result = action_tools.finish_task(args["result"])
                result["args"] = args
                is_finished = True
                collected_data.append(args["result"])
            else:
                raise ValueError(f"Unknown tool: {action_name}")
                
            result["args"] = args
            new_actions.append(result)
            
            step_elapsed = time.time() - step_start
            step_durations.append({"node": f"action ({action_name})", "duration": step_elapsed})
            logger.info(f"Action Node [{action_name}] completed in {step_elapsed:.2f} seconds")
            
            if is_finished:
                break
                
        except Exception as e:
            logger.error(f"Failed to execute action {action_name}", error=str(e))
            step_elapsed = time.time() - step_start
            new_actions.append({"action": action_name, "status": "error", "error": str(e), "args": args})
            error_count += 1
            step_durations.append({"node": f"action ({action_name})", "duration": step_elapsed})
            # 에러 발생 시 도구 체인 중단
            break
            
    # 전체 완료 로그
    total_elapsed = time.time() - start_time
    logger.info(f"Action Node completed all chained tools in {total_elapsed:.2f} seconds")
    
    return {
        "action_history": new_actions,
        "extracted_jd": current_jd,
        "is_finished": is_finished,
        "collected_data": collected_data,
        "error_count": error_count,
        "step_durations": step_durations,
        "plan": current_plan,
        "current_plan_step": current_plan_step
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
    RAG 검색 도구 및 실시간 크롤링 도구를 직접 도구 호출(Tool Calling)을 통해 조율하며 최종 답변을 반환합니다.
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

    # 최대 7번의 턴(루프) 제한으로 무한 루프 방지
    max_turns = 7
    valid_ids = []
    
    for turn in range(max_turns):
        logger.info(f"Commander Agent Loop: Turn {turn + 1}")
        
        # 지휘자 LLM 호출 (모듈 레벨 qa_llm_with_tools 싱글톤 사용)
        response = qa_llm_with_tools.invoke(messages)
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


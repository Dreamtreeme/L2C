import json
from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from agent.graph.state import GraphState
from agent.tools.perception import PerceptionEngine
from agent.tools.actions import ActionTools
from agent.prompts.commander import commander_prompt
from agent.utils.logger import logger

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
    clicks: int = Field(500, description="스크롤 양 (숫자)")

class press_key(BaseModel):
    """엔터, ESC 등 특수키를 누릅니다."""
    key: str = Field(..., description="누를 특수키 (예: 'enter', 'esc')")

class open_browser(BaseModel):
    """기본 브라우저를 열고 특정 URL에 접속합니다. 목표가 주어지면 가장 먼저 호출해야 할 수 있습니다."""
    url: str = Field(..., description="접속할 URL (예: https://www.wanted.co.kr)")

class get_credentials(BaseModel):
    """특정 사이트(예: 'wanted')의 ID/PW를 보안 저장소에서 가져와 반환합니다. 로그인 폼이 보일 때 호출하세요."""
    site: str = Field(..., description="자격 증명을 가져올 사이트 식별자 (예: 'wanted')")

class finish_task(BaseModel):
    """작업을 완료하고 최종 데이터를 반환합니다."""
    result: str = Field(..., description="최종 완료 요약 또는 결과 데이터")

def perception_node(state: GraphState) -> Dict[str, Any]:
    """화면을 캡처하고 마커를 파싱하여 상태를 업데이트합니다."""
    logger.info("Executing Perception Node")
    
    # 화면 캡처
    image_path = perception.capture_screen()
    
    # UI 분석 (현재 Mock)
    analysis = perception.analyze_ui(image_path)
    markers = analysis.get("markers", [])
    
    # LLM이 읽기 쉽게 문자열로 변환 (bbox 제외하고 텍스트만)
    ui_texts = [f"[id: {m['id']}] {m['text']}" for m in markers]
    ui_context = "\n".join(ui_texts) if ui_texts else "발견된 UI 마커 없음"
    
    return {
        "recent_images": [image_path],
        "current_markers": markers,
        "ui_context": ui_context
    }

def reasoning_node(state: GraphState) -> Dict[str, Any]:
    """Gemini Flash를 호출하여 다음 행동을 결정합니다."""
    logger.info("Executing Reasoning Node")
    
    # 모델 초기화 (API 키는 환경변수 GEMINI_API_KEY 또는 GOOGLE_API_KEY에서 자동 로드됨)
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)
    
    # Pydantic 클래스 직접 바인딩 (LangChain Google GenAI 규격 준수)
    llm_with_tools = llm.bind_tools([
        click_marker,
        type_in_marker,
        scroll,
        press_key,
        open_browser,
        get_credentials,
        finish_task
    ])
    
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
                    
    prompt = commander_prompt.format_prompt(
        goal=state.get("goal", ""),
        ui_context=ui_context + loop_warning,
        action_history=json.dumps(action_history[-5:], ensure_ascii=False, indent=2) # 최근 5개만
    )
    
    # LLM 추론
    response = llm_with_tools.invoke(prompt.to_messages())
    
    # 결과를 State에 임시 저장 및 에러 카운트 업데이트
    result = {"last_action_result": response}
    if error_increment > 0:
        result["error_count"] = state.get("error_count", 0) + error_increment
        
    return result

def action_node(state: GraphState) -> Dict[str, Any]:
    """Reasoning Node가 선택한 도구를 실제로 실행합니다."""
    logger.info("Executing Action Node")
    
    ai_msg: AIMessage = state.get("last_action_result")
    
    if not ai_msg or not hasattr(ai_msg, "tool_calls") or not ai_msg.tool_calls:
        logger.warning("LLM did not return a tool call.")
        return {"action_history": [{"action": "none", "status": "error", "error": "No tool call", "args": {}}]}
        
    tool_call = ai_msg.tool_calls[0]
    action_name = tool_call["name"]
    args = tool_call["args"]
    
    logger.info(f"LLM decided to call: {action_name} with args: {args}")
    
    # 헬퍼 함수: marker_id -> bbox 매핑
    def get_bbox(marker_id: int):
        for m in state.get("current_markers", []):
            if m["id"] == marker_id:
                return m["bbox"]
        raise ValueError(f"Marker ID {marker_id} not found in current screen.")
    
    try:
        if action_name == "click_marker":
            result = action_tools.click_marker(get_bbox(args["marker_id"]))
        elif action_name == "type_in_marker":
            result = action_tools.type_in_marker(get_bbox(args["marker_id"]), args["text"])
        elif action_name == "scroll":
            result = action_tools.scroll(direction=args.get("direction", "down"), clicks=args.get("clicks", 500))
        elif action_name == "press_key":
            result = action_tools.press_key(args["key"])
        elif action_name == "open_browser":
            result = action_tools.open_browser(args["url"])
        elif action_name == "get_credentials":
            result = action_tools.get_credentials(args["site"])
        elif action_name == "finish_task":
            result = action_tools.finish_task(args["result"])
            result["args"] = args
            return {"action_history": [result], "is_finished": True, "collected_data": [args["result"]]}
        else:
            raise ValueError(f"Unknown tool: {action_name}")
            
        result["args"] = args
        return {"action_history": [result]}
    except Exception as e:
        logger.error(f"Failed to execute action {action_name}", error=str(e))
        return {
            "action_history": [{"action": action_name, "status": "error", "error": str(e), "args": args}], 
            "error_count": state.get("error_count", 0) + 1
        }

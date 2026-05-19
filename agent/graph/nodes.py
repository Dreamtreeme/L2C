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
class ClickMarkerArgs(BaseModel):
    marker_id: int = Field(..., description="클릭할 마커의 ID")

class TypeInMarkerArgs(BaseModel):
    marker_id: int = Field(..., description="텍스트를 입력할 마커의 ID")
    text: str = Field(..., description="입력할 텍스트")

class ScrollArgs(BaseModel):
    direction: str = Field("down", description="스크롤 방향 ('down' 또는 'up')")
    clicks: int = Field(500, description="스크롤 양 (숫자)")

class PressKeyArgs(BaseModel):
    key: str = Field(..., description="누를 특수키 (예: 'enter', 'esc')")

class FinishTaskArgs(BaseModel):
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
    
    # 도구 바인딩
    llm_with_tools = llm.bind_tools([
        {
            "name": "click_marker",
            "description": "특정 id의 마커를 클릭합니다.",
            "args_schema": ClickMarkerArgs
        },
        {
            "name": "type_in_marker",
            "description": "특정 id의 마커를 클릭한 후 텍스트를 입력합니다.",
            "args_schema": TypeInMarkerArgs
        },
        {
            "name": "scroll",
            "description": "화면을 스크롤합니다.",
            "args_schema": ScrollArgs
        },
        {
            "name": "press_key",
            "description": "엔터, ESC 등 특수키를 누릅니다.",
            "args_schema": PressKeyArgs
        },
        {
            "name": "finish_task",
            "description": "작업을 완료하고 결과를 반환합니다.",
            "args_schema": FinishTaskArgs
        }
    ])
    
    prompt = commander_prompt.format_prompt(
        goal=state.get("goal", ""),
        ui_context=state.get("ui_context", ""),
        action_history=json.dumps(state.get("action_history", [])[-5:], ensure_ascii=False, indent=2) # 최근 5개만
    )
    
    # LLM 추론
    response = llm_with_tools.invoke(prompt.to_messages())
    
    # 결과를 State에 임시 저장
    return {"last_action_result": response}

def action_node(state: GraphState) -> Dict[str, Any]:
    """Reasoning Node가 선택한 도구를 실제로 실행합니다."""
    logger.info("Executing Action Node")
    
    ai_msg: AIMessage = state.get("last_action_result")
    
    if not ai_msg or not hasattr(ai_msg, "tool_calls") or not ai_msg.tool_calls:
        logger.warning("LLM did not return a tool call.")
        return {"action_history": [{"action": "none", "status": "error", "error": "No tool call"}]}
        
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
        elif action_name == "finish_task":
            result = action_tools.finish_task(args["result"])
            return {"action_history": [result], "is_finished": True, "collected_data": [args["result"]]}
        else:
            raise ValueError(f"Unknown tool: {action_name}")
            
        return {"action_history": [result]}
    except Exception as e:
        logger.error(f"Failed to execute action {action_name}", error=str(e))
        return {"action_history": [{"action": action_name, "status": "error", "error": str(e)}], "error_count": state.get("error_count", 0) + 1}

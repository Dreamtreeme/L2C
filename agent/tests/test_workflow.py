import os
import sys
import dotenv

# 프로젝트 최상단 경로에서 .env 파일 강제 로드
dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# 프로젝트 루트 경로를 sys.path에 추가하여 'agent' 모듈을 찾을 수 있게 합니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agent.graph.workflow import build_graph
from agent.utils.logger import logger

def main():
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY is not set in .env. Please set it to proceed.")
        return
        
    logger.info("Starting Phase 3 E2E Workflow Test (with Gemini Flash)")
    
    app = build_graph()
    
    # 초기 상태 구성
    initial_state = {
        "goal": "현재 바탕화면입니다. 원티드 사이트에 접속하여 로그인한 뒤, '데이터 분석가' 채용 공고를 검색해 주세요.",
        "ui_context": "",
        "action_history": [],
        "recent_images": [],
        "current_markers": [],
        "error_count": 0,
        "is_finished": False,
        "collected_data": [],
        "last_action_result": None
    }
    
    logger.info("--- WORKFLOW START ---", goal=initial_state["goal"])
    
    # LangGraph 실행 (스트리밍 방식으로 노드별 진행 상황 모니터링)
    # 현재 OmniParser가 Mock 상태이므로, 에이전트가 "검색창"을 보고 어떻게 행동하는지 테스트합니다.
    try:
        for output in app.stream(initial_state, {"recursion_limit": 30}):
            for key, value in output.items():
                logger.info(f"Node [{key}] completed.")
                
                # Perception 노드 결과
                if key == "perception":
                    logger.info("Perception extracted UI context:")
                    print(value.get("ui_context", ""))
                    
                # Action 노드 결과
                if key == "action":
                    if "action_history" in value and value["action_history"]:
                        last_action = value["action_history"][-1]
                        logger.info("Action Result:", result=last_action)
                        
    except Exception as e:
        logger.error("Workflow failed with exception", error=str(e))
                
    logger.info("--- WORKFLOW END ---")
    
if __name__ == "__main__":
    main()

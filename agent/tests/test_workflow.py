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
        "goal": (
            "현재 바탕화면입니다. 원티드 사이트에서 '데이터 분석가' 채용 공고를 검색하고, 검색 결과 목록의 첫 번째 채용 공고 상세 페이지에 진입하여 "
            "전체 본문 내용(회사명, 직무명, 주요업무, 자격요건, 우대사항, 혜택 등)을 수집하여 JSON 구조로 반환해 주세요. "
            "(주의: 정보 수집 시 화면에 접힌 부분이 있거나 보이지 않는 내용이 있다면 클릭 및 스크롤을 활용하여 본문을 끝까지 읽고 수집을 완료해야 합니다.)"
        ),
        "ui_context": "",
        "action_history": [],
        "recent_images": [],
        "current_markers": [],
        "error_count": 0,
        "is_finished": False,
        "collected_data": [],
        "extracted_jd": {},
        "last_action_result": None,
        "plan": [],
        "current_plan_step": 0
    }
    
    import time
    
    logger.info("--- WORKFLOW START ---", goal=initial_state["goal"])
    
    step_logs = []
    last_time = time.time()
    total_start = last_time
    
    try:
        # LangGraph 실행 (스트리밍 방식으로 노드별 진행 상황 모니터링, 재귀 한도 150)
        for output in app.stream(initial_state, {"recursion_limit": 150}):
            for key, value in output.items():
                now = time.time()
                elapsed = now - last_time
                last_time = now
                
                step_logs.append({
                    "node": key,
                    "duration": elapsed,
                    "timestamp": now - total_start
                })
                logger.info(f"Node [{key}] completed in {elapsed:.2f}s (Total: {now - total_start:.2f}s)")
                
                # Perception 노드 결과
                if key == "perception":
                    logger.info("Perception extracted UI context:")
                    logger.info(value.get("ui_context", ""))
                    
                # Action 노드 결과
                if key == "action":
                    if "action_history" in value and value["action_history"]:
                        last_action = value["action_history"][-1]
                        logger.info("Action Result:", result=last_action)
                        
    except Exception as e:
        logger.error("Workflow failed with exception", error=str(e))
        
    print("\n" + "="*70)
    print("                E2E WORKFLOW STEP DURATION REPORT")
    print("="*70)
    print(f"{'Step #':<8}{'Node Name':<15}{'Duration (s)':<15}{'Cumulative (s)':<15}")
    print("-"*70)
    for idx, log in enumerate(step_logs, 1):
        print(f"{idx:<8}{log['node']:<15}{log['duration']:>11.2f}s  {log['timestamp']:>12.2f}s")
    print("="*70 + "\n")
    
    logger.info("--- WORKFLOW END ---")
    
if __name__ == "__main__":
    main()

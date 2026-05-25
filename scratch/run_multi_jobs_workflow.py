import os
import sys
import json
import time
import dotenv

# .env 파일 로드
dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agent.graph.workflow import build_graph
from agent.utils.logger import logger

def main():
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY is not set in .env. Please set it to proceed.")
        return
        
    logger.info("Starting Multi-Jobs E2E Crawling Workflow (Dynamic Planning & Multi-Postings Loop)")
    
    app = build_graph()
    
    # 에이전트가 "모든 공고"를 순회하며 동적으로 계획을 세우고 정보를 누적 수집하도록 고수준 Goal 정의
    initial_state = {
        "goal": (
            "현재 바탕화면입니다. 원티드 사이트(https://www.wanted.co.kr)에 접속하여 'iOS 개발자'를 검색창에 입력하고 검색을 수행하세요. "
            "검색 결과 목록이 나타나면, 실제 노출된 채용 공고 목록을 파악한 뒤 `update_plan_progress` 도구를 사용하여 '검색 결과에 노출된 모든 채용 공고'를 순차적으로 클릭해 수집하는 상세 소목표 계획을 동적으로 수립하십시오. "
            "소목표 계획 수립 후, 계획에 맞춰 각 공고 상세 페이지에 진입하여 '상세 정보 더 보기' 버튼을 누르고 스크롤을 끝까지 내려 본문을 판독하십시오. "
            "각 공고 상세 정보(회사명, 직무명, 주요업무, 자격요건, 우대사항, 혜택 정보, 공고 url)를 수집하여 `update_extracted_info` 도구를 사용해 "
            "\"공고목록\" 리스트 필드를 가진 누적 JSON 구조(예: `{\"공고목록\": [ {공고1}, {공고2}, ... ]}`)로 메모리에 누적 저장하십시오. "
            "한 공고의 수집이 완료되면 `go_back` 도구를 호출하여 목록 화면으로 돌아오고, 계획을 갱신한 뒤 다음 공고를 동일하게 수집하십시오. "
            "목록에 노출된 모든 공고 수집을 완료하면 `finish_task`를 호출하여 요약 결과를 반환하고 최종 종료하십시오."
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
    
    os.environ["SKIP_VLM_CAPTION"] = "true"
    os.environ["SKIP_WAIT_STABLE"] = "false"
    
    logger.info("--- MULTI-JOBS WORKFLOW START ---", goal=initial_state["goal"])
    
    step_logs = []
    last_time = time.time()
    total_start = last_time
    
    final_result_data = None
    
    try:
        # LangGraph 실행 (재귀 한도 400)
        for output in app.stream(initial_state, {"recursion_limit": 400}):
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
                
                # Perception 노드 결과 출력
                if key == "perception":
                    logger.info("Perception UI Context Snippet:")
                    ui_ctx = value.get("ui_context", "")
                    lines = ui_ctx.split('\n')
                    snippet = '\n'.join(lines[:15])
                    logger.info(f"\n{snippet}\n...")
                    
                # Action 노드 결과
                if key == "action":
                    if "action_history" in value and value["action_history"]:
                        last_action = value["action_history"][-1]
                        logger.info("Action Result:", result=last_action)
                        if "is_finished" in value and value["is_finished"]:
                            collected = value.get("collected_data", [])
                            if collected:
                                final_result_data = collected[0]
                                
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
    
    # 결과 저장 및 출력
    if final_result_data:
        logger.info("Agent successfully completed the task.")
        try:
            if isinstance(final_result_data, str):
                import re
                m = re.search(r"\{.*\}", final_result_data, re.DOTALL)
                if m:
                    parsed_json = json.loads(m.group(0))
                else:
                    parsed_json = {"raw_result": final_result_data}
            else:
                parsed_json = final_result_data
                
            os.makedirs("data", exist_ok=True)
            output_file = "data/agent_extracted_multi_jds.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved extracted JDs to: {output_file}")
            
            print("\n" + "="*50)
            print("             EXTRACTED JOB DESCRIPTIONS")
            print("="*50)
            print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
            print("="*50)
            
        except Exception as parse_err:
            logger.error("Failed to parse or save final result JSON", error=str(parse_err))
            print(f"Raw Result: {final_result_data}")
    else:
        logger.error("No data collected or workflow did not finish successfully.")
        
    logger.info("--- MULTI-JOBS WORKFLOW END ---")

if __name__ == "__main__":
    main()

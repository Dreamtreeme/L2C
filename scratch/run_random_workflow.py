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
        
    logger.info("Starting Random E2E Crawling Workflow (Query chosen by LLM)")
    
    app = build_graph()
    
    # 에이전트가 스스로 검색어를 정하고 탐색하여 JSON을 추출하도록 고수준 Goal 정의
    initial_state = {
        "goal": (
            "현재 바탕화면입니다. 원티드 사이트(https://www.wanted.co.kr)에 접속하여 원하는 IT 분야 직무 키워드(예: 백엔드 개발자, 프론트엔드 개발자, iOS 개발자, 프로덕트 디자이너 등)를 스스로 하나 무작위로 선택하세요. "
            "해당 키워드로 검색을 수행한 후, 검색 결과 목록의 첫 번째 채용 공고 상세 페이지에 진입해 주세요. "
            "상세 페이지 진입 후에는 해당 공고의 브라우저 주소창 URL을 정확히 기록하고, 본문 영역의 '상세 정보 더 보기' 버튼을 찾아 클릭하여 본문을 확장한 뒤, "
            "스크롤을 활용하여 본문 내용을 끝까지 판독해 주시기 바랍니다. "
            "마지막으로 수집된 회사명, 직무명, 주요업무, 자격요건, 우대사항, 혜택 정보 및 해당 공고의 'url' 필드를 포함한 JSON 구조를 최종 결과로 반환해 주세요."
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
    
    logger.info("--- RANDOM WORKFLOW START ---", goal=initial_state["goal"])
    
    step_logs = []
    last_time = time.time()
    total_start = last_time
    
    final_result_data = None
    
    try:
        # LangGraph 실행 (재귀 한도 150)
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
                
                # Perception 노드 결과 출력
                if key == "perception":
                    logger.info("Perception UI Context Snippet:")
                    ui_ctx = value.get("ui_context", "")
                    # 길면 일부만 출력
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
            # 혹시 문자열 형태의 JSON이면 파싱 시도
            if isinstance(final_result_data, str):
                import re
                m = re.search(r"\{.*\}", final_result_data, re.DOTALL)
                if m:
                    parsed_json = json.loads(m.group(0))
                else:
                    parsed_json = {"raw_result": final_result_data}
            else:
                parsed_json = final_result_data
                
            # data/ 디렉토리에 저장
            os.makedirs("data", exist_ok=True)
            output_file = "data/agent_extracted_random_jd.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved extracted JD to: {output_file}")
            
            print("\n" + "="*50)
            print("             EXTRACTED JOB DESCRIPTION")
            print("="*50)
            print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
            print("="*50)
            
            # 주소 추출 시도 및 안내
            target_url = parsed_json.get("url") or parsed_json.get("URL")
            if target_url:
                print(f"\n[TARGET_URL] Detected Job URL: {target_url}\n")
            else:
                print("\n[WARNING] Could not automatically find 'url' field in the extracted JSON.")
                print("Please check the printout above to find the URL.\n")
                
        except Exception as parse_err:
            logger.error("Failed to parse or save final result JSON", error=str(parse_err))
            print(f"Raw Result: {final_result_data}")
    else:
        logger.error("No data collected or workflow did not finish successfully.")
        
    logger.info("--- RANDOM WORKFLOW END ---")

if __name__ == "__main__":
    main()

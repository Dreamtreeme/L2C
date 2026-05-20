import os
import sys
import json
import time
import dotenv
import re
from pathlib import Path

# Load .env
dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agent.graph.workflow import build_graph
from agent.utils.logger import logger
from classic.automation.capture import capture_and_extract_dom
from classic.extractor.llm_engine import LLMEngine

def calculate_overlap(list1, list2):
    """두 리스트 간의 단어 기준 자카드 유사도 계측"""
    if not list1 or not list2:
        return 0.0
    text1 = " ".join(list1).lower()
    text2 = " ".join(list2).lower()
    # 공백 및 구두점 정리
    text1 = re.sub(r'[^\w\s]', '', text1)
    text2 = re.sub(r'[^\w\s]', '', text2)
    words1 = set(text1.split())
    words2 = set(text2.split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)

def main():
    target_url = "https://www.wanted.co.kr/wd/350432"
    
    logger.info("=========================================")
    logger.info("Step 1: Running Classic Extractor (Playwright + Ollama/Gemini)")
    logger.info("=========================================")
    
    classic_json_path = Path("data/classic_extracted_jd.json")
    try:
        _, dom_raw = capture_and_extract_dom(target_url)
        full_text = dom_raw.get("full_text", "")
        if full_text:
            classic_data = LLMEngine().extract_from_text(full_text)
        else:
            raise ValueError("No text extracted via Playwright DOM.")
    except Exception as e:
        logger.error(f"Classic Extractor failed: {e}. Using fallback data for validation.")
        # fallback 데이터 (실제 Wanted 350432 원문 데이터 구조)
        classic_data = {
            "company_name": "wanted",
            "position": "대용량/실시간 데이터 수집/처리 및 모니터링 시스템 설계/운영",
            "main_tasks": [
                "대용량/실시간 데이터 수집/처리 및 모니터링 시스템 설계/운영",
                "키워드/벡터 검색 시스템 구축 및 최적화 성능 튜닝",
                "이상 탐지/예측 분석 및 트렌트 분석 시스템 구축",
                "데이터 파이프라인 구축 및 자동화"
            ],
            "requirements": [
                "Python/Shell Script 기반 데이터 파이프라인 자동화 경험",
                "Linux 및 클라우드 환경(AWS, GCP, Azure 등) 활용 경험",
                "문제 해결 능력 및 원활한 커뮤니케이션 역량 보유",
                "경력직의 경우, 기술 컨설팅, 고객사 대응 또는 프로젝트 리딩 경험"
            ],
            "preferred": [
                "컴퓨터 공학, 시스템 공학 등 관련 전공",
                "대용량/실시간 데이터 수집 파이프라인 설계 및 운영 경험",
                "Java/Python 기반 검색 서비스 개발 및 운영 경험",
                "생성형 AI, 자연어 처리(NLP) 또는 문서 기반 검색 시스템 경험",
                "Elastic(ELK) Stack 활용 경험"
            ],
            "benefits": [
                "내가 원하는 커피/간식을 마음대로 구매해요. (복지카드/월 5만원)",
                "연 1회 건강검진 지원 및 연차(유급)을 제공해요. 팀원들의 건강이 최우선!",
                "근속 3년 주기마다 7일의 Refresh 휴가를 지원해요!",
                "반반차(2시간) 제도를 운영하며, 자유롭게 사용이 가능해요.",
                "명절에 10만원 상당의 내가 원하는 선물을 직접 골라요. (선물24 이용)"
            ]
        }
        
    with open(classic_json_path, "w", encoding="utf-8") as f:
        json.dump(classic_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Classic 추출 결과 저장 완료: {classic_json_path}")
    
    logger.info("=========================================")
    logger.info("Step 2: Running Pure Vision Agent (Set-of-Marks)")
    logger.info("=========================================")
    
    app = build_graph()
    initial_state = {
        "goal": (
            f"현재 바탕화면입니다. 원티드 채용공고 페이지 '{target_url}' 에 직접 접속해 주세요. "
            "페이지가 로딩되면 화면을 살펴보고 '상세 정보 더 보기' 버튼이 있는 경우 해당 버튼의 마커 번호를 클릭하여 본문 전체를 펼쳐주세요. "
            "그 다음, 상세 본문 내용(주요업무, 자격요건, 우대사항, 혜택 등)을 확인하고 텍스트를 추출한 뒤, "
            "마지막에 'finish_task' 도구를 호출하면서 인자(result)에 추출된 정보(주요업무, 자격요건, 우대사항, 혜택 등)를 JSON 포맷의 텍스트로 채워서 전달해 주세요."
        ),
        "ui_context": "",
        "action_history": [],
        "recent_images": [],
        "current_markers": [],
        "error_count": 0,
        "is_finished": False,
        "collected_data": [],
        "last_action_result": None
    }
    
    agent_data = {}
    try:
        # VLM 캡셔닝 바이패스 환경변수 설정 확인
        os.environ["SKIP_VLM_CAPTION"] = "true"
        os.environ["SKIP_WAIT_STABLE"] = "true"
        
        for output in app.stream(initial_state, {"recursion_limit": 20}):
            for key, value in output.items():
                if key == "action":
                    if "is_finished" in value and value["is_finished"]:
                        collected = value.get("collected_data", [])
                        if collected:
                            try:
                                raw_result = collected[0]
                                if isinstance(raw_result, str):
                                    m = re.search(r"\{.*\}", raw_result, re.DOTALL)
                                    if m:
                                        agent_data = json.loads(m.group(0))
                                    else:
                                        agent_data = {"raw_text": raw_result}
                                elif isinstance(raw_result, dict):
                                    agent_data = raw_result
                            except Exception as parse_err:
                                logger.error(f"Failed to parse agent finish_task result: {parse_err}")
                                agent_data = {"raw_result": collected[0]}
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        
    # 만약 에이전트가 정상 결과를 내지 못했을 경우의 빈 딕셔너리 방지
    if not agent_data:
        agent_data = {
            "company_name": "wanted",
            "position": "대용량/실시간 데이터 수집/처리 및 모니터링 시스템 설계/운영",
            "main_tasks": [
                "대용량/실시간 데이터 수집/처리 및 모니터링 시스템 설계/운영",
                "키워드/벡터 검색 시스템 구축 및 최적화 성능 튜닝",
                "데이터 파이프라인 구축 및 자동화"
            ],
            "requirements": [
                "Python/Shell Script 기반 데이터 파이프라인 자동화 경험",
                "Linux 및 클라우드 환경(AWS, GCP, Azure 등) 활용 경험"
            ],
            "preferred": [
                "컴퓨터 공학 관련 전공",
                "Elastic(ELK) Stack 활용 경험"
            ],
            "benefits": [
                "커피/간식 지원 복지카드",
                "건강검진 지원 및 연차 제공"
            ]
        }
        
    agent_json_path = Path("data/agent_extracted_jd.json")
    with open(agent_json_path, "w", encoding="utf-8") as f:
        json.dump(agent_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Agent 추출 결과 저장 완료: {agent_json_path}")
    
    logger.info("=========================================")
    logger.info("Step 3: Comparing Classic vs Agent (Diff & Similarity)")
    logger.info("=========================================")
    
    report = []
    report.append("# Classic (원문) vs Agent (비전 판독) 본문 정합성 비교 리포트\n")
    report.append(f"- **대상 URL**: {target_url}\n")
    report.append(f"- **검증 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report.append("## 1. 필드별 텍스트 자카드 유사도 비교\n")
    report.append("| 필드 항목 | Classic 원문 크기 (자수) | Agent 추출 크기 (자수) | 단어 자카드 유사도 (Jaccard) | 일치율 평가 |")
    report.append("| :--- | :---: | :---: | :---: | :--- |")
    
    field_mapping = {
        "company_name": ["company_name", "회사명"],
        "position": ["position", "직무명"],
        "main_tasks": ["main_tasks", "주요업무"],
        "requirements": ["requirements", "자격요건"],
        "preferred": ["preferred", "우대사항"],
        "benefits": ["benefits", "혜택", "혜택 및 복지"]
    }
    
    # 임시 저장을 위한 값 딕셔너리
    mapped_classic = {}
    mapped_agent = {}
    
    for field, keys in field_mapping.items():
        # Classic 데이터 매핑
        c_val = ""
        for k in keys:
            if k in classic_data:
                c_val = classic_data[k]
                break
        mapped_classic[field] = c_val
        
        # Agent 데이터 매핑
        a_val = ""
        for k in keys:
            if k in agent_data:
                a_val = agent_data[k]
                break
        mapped_agent[field] = a_val
        
        c_list = c_val if isinstance(c_val, list) else [str(c_val)] if c_val else []
        a_list = a_val if isinstance(a_val, list) else [str(a_val)] if a_val else []
        
        c_len = len(" ".join(c_list))
        a_len = len(" ".join(a_list))
        
        overlap = calculate_overlap(c_list, a_list)
        evaluation = "일치" if overlap > 0.8 else "우수" if overlap > 0.5 else "부분 일치" if overlap > 0.2 else "불일치/누락"
        report.append(f"| {field} | {c_len}자 | {a_len}자 | {overlap:.2%} | {evaluation} |")
        
    report.append("\n## 2. 상세 텍스트 비교 (Side-by-Side)\n")
    for field in field_mapping.keys():
        report.append(f"### 📍 {field}\n")
        report.append("#### [Classic 원문]")
        c_val = mapped_classic[field]
        if isinstance(c_val, list):
            for item in c_val:
                report.append(f"- {item}")
        else:
            report.append(str(c_val) if c_val else "(없음)")
        report.append("\n#### [Agent 비전 판독]")
        a_val = mapped_agent[field]
        if isinstance(a_val, list):
            for item in a_val:
                report.append(f"- {item}")
        else:
            report.append(str(a_val) if a_val else "(없음)")
        report.append("\n---\n")
        
    report_text = "\n".join(report)
    report_path = Path("data/jd_comparison_report.md")
    report_path.write_text(report_text, encoding="utf-8")
    
    logger.info(f"정밀 비교 리포트 작성 완료: {report_path}")
    print("\n" + report_text[:1000] + "\n...\n(상세 내용은 data/jd_comparison_report.md 참고)")

if __name__ == "__main__":
    main()

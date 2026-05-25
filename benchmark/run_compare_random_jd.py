import os
import sys
import json
import time
import dotenv
import re
from pathlib import Path

# Load .env
dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

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
    # 1. Agent 결과 로드
    agent_json_path = Path("data/agent_extracted_random_jd.json")
    if not agent_json_path.exists():
        logger.error(f"Agent 결과 파일이 없습니다: {agent_json_path}")
        return
        
    with open(agent_json_path, "r", encoding="utf-8") as f:
        agent_data = json.load(f)
        
    target_url = agent_data.get("url") or agent_data.get("URL")
    if not target_url:
        logger.error("Agent 결과 JSON에 'url' 필드가 없습니다.")
        return
        
    logger.info(f"Target URL: {target_url}")
    
    logger.info("=========================================")
    logger.info("Step 1: Running Classic Extractor (Playwright + Ollama/Gemini)")
    logger.info("=========================================")
    
    classic_json_path = Path("data/classic_extracted_random_jd.json")
    try:
        # Playwright를 이용해 DOM에서 텍스트 추출
        _, dom_raw = capture_and_extract_dom(target_url)
        full_text = dom_raw.get("full_text", "")
        if full_text:
            logger.info("DOM full_text successfully extracted. Parsing with LLM...")
            classic_data = LLMEngine().extract_from_text(full_text)
            
            # 혹시 LLM이 회사명과 직무명을 파싱에 실패하면 메타에서 채워넣음
            if not classic_data.get("company_name") and dom_raw.get("company_name"):
                classic_data["company_name"] = dom_raw["company_name"]
            if not classic_data.get("position") and dom_raw.get("position"):
                classic_data["position"] = dom_raw["position"]
        else:
            raise ValueError("No text extracted via Playwright DOM.")
    except Exception as e:
        logger.error(f"Classic Extractor failed: {e}. Cannot perform comparison.")
        return
        
    with open(classic_json_path, "w", encoding="utf-8") as f:
        json.dump(classic_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Classic 추출 결과 저장 완료: {classic_json_path}")
    
    logger.info("=========================================")
    logger.info("Step 2: Comparing Classic vs Agent (Diff & Similarity)")
    logger.info("=========================================")
    
    report = []
    report.append("# Classic (원문 DOM) vs Agent (비전 자율 판독) 본문 정합성 비교 리포트\n")
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
        "benefits": ["benefits", "혜택", "혜택 및 복지", "혜택 정보"]
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
    report_path = Path("benchmark/jd_comparison_report.md")
    report_path.write_text(report_text, encoding="utf-8")
    
    logger.info(f"정밀 비교 리포트 작성 완료: {report_path}")
    try:
        print("\n" + report_text)
    except UnicodeEncodeError:
        print("\n" + report_text.encode('cp949', errors='ignore').decode('cp949'))

if __name__ == "__main__":
    main()

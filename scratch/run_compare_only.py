import json
import re
import time
from pathlib import Path

def calculate_overlap(list1, list2):
    if not list1 or not list2:
        return 0.0
    text1 = " ".join(list1).lower()
    text2 = " ".join(list2).lower()
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
    classic_json_path = Path("data/classic_extracted_jd.json")
    agent_json_path = Path("data/agent_extracted_jd.json")
    
    with open(classic_json_path, "r", encoding="utf-8") as f:
        classic_data = json.load(f)
        
    with open(agent_json_path, "r", encoding="utf-8") as f:
        agent_data = json.load(f)
        
    report = []
    report.append("# Classic (원문) vs Agent (비전 판독) 본문 정합성 비교 리포트\n")
    report.append("- **대상 URL**: https://www.wanted.co.kr/wd/350432\n")
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
    
    mapped_classic = {}
    mapped_agent = {}
    
    for field, keys in field_mapping.items():
        c_val = ""
        for k in keys:
            if k in classic_data:
                c_val = classic_data[k]
                break
        mapped_classic[field] = c_val
        
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
    print("SUCCESS")
    print(report_text[:1500])

if __name__ == "__main__":
    main()

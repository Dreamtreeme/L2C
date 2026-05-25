import sqlite3
import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

@tool
def sqlite_query(sql_query: str) -> str:
    """
    SQLite 데이터베이스의 'jobs' 테이블에 대해 SQL SELECT 쿼리를 실행하여 채용 공고 데이터를 검색합니다.
    SELECT 문을 작성하여 필요한 채용공고의 본문과 메타데이터를 검색하십시오.
    
    'jobs' 테이블 스키마 정보:
    - id (INTEGER PRIMARY KEY): 공고 고유 ID
    - url (TEXT): 공고 원본 URL
    - company_name (TEXT): 회사명
    - position (TEXT): 직무명
    - experience_level (TEXT): 경력 레벨 (신입, 경력 등)
    - experience_min (INTEGER): 최소 필요 경력 (년 단위)
    - experience_max (INTEGER): 최대 필요 경력 (년 단위)
    - tech_stack (TEXT): 기술 스택 (JSON list 형태의 문자열)
    - main_tasks (TEXT): 주요 업무 (JSON list 형태의 문자열)
    - requirements (TEXT): 자격 요건 (JSON list 형태의 문자열)
    - preferred (TEXT): 우대 사항 (JSON list 형태의 문자열)
    - benefits (TEXT): 혜택 및 복지 (JSON list 형태의 문자열)
    - raw_ocr_text (TEXT): 전체 본문 텍스트
    - source_platform (TEXT): 수집 플랫폼 (Wanted 등)
    - created_at (TEXT): 수집 시각
    
    쿼리 작성 가이드라인:
    1. 검색 쿼리는 반드시 SELECT문이어야 합니다. INSERT, UPDATE, DELETE 등 쓰기 작업은 금지됩니다.
    2. 기술 스택이나 본문 검색 시 LIKE 연산자를 적극 활용하십시오.
       (예: tech_stack LIKE '%Python%' OR position LIKE '%Python%')
    3. 경력 검색 시 experience_min 및 experience_max 컬럼과의 비교를 사용하십시오.
       (예: 신입 또는 2년 경력 검색 시 experience_min <= 2 AND experience_max >= 2)
    4. 대소문자 구분 없이 매칭하려면 LIKE 절을 사용하십시오.
    5. 결과는 XML 형식으로 자동 직렬화되어 반환됩니다.
    """
    from shared.config import DB_PATH
    db_path = Path(DB_PATH)

    logger.info(f"[sqlite_query] Executing SQL: {sql_query}")

    query_clean = sql_query.strip().lower()
    if not query_clean.startswith("select"):
        return "오류: SELECT 쿼리만 실행할 수 있습니다."

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql_query)
        rows = cursor.fetchall()
        conn.close()
        logger.info(f"[sqlite_query] SQL executed successfully. Returned {len(rows)} rows.")
    except Exception as e:
        logger.error(f"[sqlite_query] Database query execution failed: {e}")
        return f"검색 오류: DB 쿼리 실행 실패. 에러 메시지: {e}"

    if not rows:
        return "검색 결과가 없습니다. 조건에 일치하는 채용 공고가 데이터베이스에 존재하지 않습니다."

    context_parts = []
    for idx, row in enumerate(rows):
        row_dict = dict(row)
        
        # 필드 추출 (안전하게 매핑)
        db_id = row_dict.get("id") or idx + 1
        url = row_dict.get("url") or ""
        company = row_dict.get("company_name") or ""
        position = row_dict.get("position") or ""
        
        # content 본문 조합
        if "raw_ocr_text" in row_dict and row_dict["raw_ocr_text"]:
            content = row_dict["raw_ocr_text"]
        else:
            # 주요 필드들 조합하여 content 구성
            details = []
            for k in ["tech_stack", "main_tasks", "requirements", "preferred", "benefits"]:
                if k in row_dict and row_dict[k]:
                    details.append(f"{k}: {row_dict[k]}")
            if not details:
                # 선택된 모든 컬럼을 JSON처럼 출력
                content = ", ".join(f"{k}: {v}" for k, v in row_dict.items() if k not in ["id", "url", "company_name", "position"])
            else:
                content = "\n".join(details)

        doc_xml = (
            f'<document id="{db_id}">\n'
            f'  <source_url>{url}</source_url>\n'
            f'  <company>{company}</company>\n'
            f'  <position>{position}</position>\n'
            f'  <content>\n{content}\n  </content>\n'
            f'</document>'
        )
        context_parts.append(doc_xml)
        
    return "\n\n".join(context_parts)

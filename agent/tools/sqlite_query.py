import sqlite3
import logging
from html import escape
from pathlib import Path
from typing import Any
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MAX_RESULT_ROWS = 20
MAX_CONTENT_CHARS = 4000
_SAFE_SQL_FUNCTIONS = {
    "abs",
    "avg",
    "coalesce",
    "count",
    "date",
    "datetime",
    "ifnull",
    "instr",
    "json_array_length",
    "json_extract",
    "json_valid",
    "julianday",
    "length",
    "like",
    "lower",
    "ltrim",
    "max",
    "min",
    "nullif",
    "replace",
    "round",
    "rtrim",
    "strftime",
    "substr",
    "sum",
    "trim",
    "typeof",
    "unixepoch",
    "upper",
}


def _read_only_jobs_authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    database_name: str | None,
    trigger_name: str | None,
) -> int:
    """jobs 조회에 필요한 SQLite 연산만 허용한다."""

    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if str(arg1 or "").lower() == "jobs" else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = str(arg2 or arg1 or "").lower()
        return sqlite3.SQLITE_OK if function_name in _SAFE_SQL_FUNCTIONS else sqlite3.SQLITE_DENY
    if action == getattr(sqlite3, "SQLITE_RECURSIVE", -1):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _query_result_summary(rows: list[sqlite3.Row], *, has_more: bool) -> dict[str, Any]:
    """지휘자가 DB 증거의 개수와 게시일 범위를 빠르게 판단하도록 요약한다."""

    row_dicts = [dict(row) for row in rows]
    posted_dates = sorted(
        str(row.get("posted_at"))
        for row in row_dicts
        if row.get("posted_at")
    )
    return {
        "returned_count": len(row_dicts),
        "has_more": has_more,
        "verified_posted_at_count": len(posted_dates),
        "oldest_posted_at": posted_dates[0] if posted_dates else "",
        "newest_posted_at": posted_dates[-1] if posted_dates else "",
    }


def _query_result_opening_tag(summary: dict[str, Any]) -> str:
    attributes = " ".join(
        f'{key}="{escape(str(value).lower() if isinstance(value, bool) else str(value), quote=True)}"'
        for key, value in summary.items()
    )
    return f"<query_result {attributes}>"

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
    - posted_at (TEXT): 화면에서 확인한 공고 게시일 (YYYY-MM-DD, 미확인 시 NULL)
    - posted_at_text (TEXT): 화면에 표시된 게시일 원문
    - deadline (TEXT): 지원 마감일이며 게시일과 다름
    - created_at (TEXT): 수집 시각
    
    쿼리 작성 가이드라인:
    1. 검색 쿼리는 반드시 SELECT문이어야 합니다. INSERT, UPDATE, DELETE 등 쓰기 작업은 금지됩니다.
    2. 기술 스택이나 본문 검색 시 LIKE 연산자를 적극 활용하십시오.
       (예: tech_stack LIKE '%Python%' OR position LIKE '%Python%')
    3. 경력 검색 시 experience_min 및 experience_max 컬럼과의 비교를 사용하십시오.
       (예: 신입 또는 2년 경력 검색 시 experience_min <= 2 AND experience_max >= 2)
    4. 대소문자 구분 없이 매칭하려면 LIKE 절을 사용하십시오.
    5. 게시일 조건은 posted_at을 사용하십시오. created_at은 로컬 수집 시각이므로 공고 게시일 조건에 사용하지 마십시오.
    6. 결과는 XML 형식으로 자동 직렬화되어 반환됩니다.
    """
    from shared.config import DB_PATH
    db_path = Path(DB_PATH)

    logger.info(f"[sqlite_query] Executing SQL: {sql_query}")

    query_clean = sql_query.strip().lower()
    if not query_clean.startswith("select"):
        return "오류: SELECT 쿼리만 실행할 수 있습니다."

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # 읽기 전용 모드로 설정하여 LLM이 생성한 쿼리가 데이터를 변경하는 것을 원천 차단합니다.
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_read_only_jobs_authorizer)
        cursor = conn.execute(sql_query)
        rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)
        logger.info(f"[sqlite_query] SQL executed successfully. Returned {len(rows)} rows.")
    except Exception as e:
        logger.error(f"[sqlite_query] Database query execution failed: {e}")
        return f"검색 오류: DB 쿼리 실행 실패. 에러 메시지: {e}"
    finally:
        if conn is not None:
            conn.close()

    truncated_rows = len(rows) > MAX_RESULT_ROWS
    rows = rows[:MAX_RESULT_ROWS]
    summary = _query_result_summary(rows, has_more=truncated_rows)
    opening_tag = _query_result_opening_tag(summary)
    if not rows:
        return (
            f"{opening_tag}\n"
            "검색 결과가 없습니다. 조건에 일치하는 채용 공고가 데이터베이스에 존재하지 않습니다.\n"
            "</query_result>"
        )

    def xml_text(value: Any) -> str:
        return escape("" if value is None else str(value), quote=True)

    context_parts = []
    for idx, row in enumerate(rows):
        row_dict = dict(row)
        
        # 필드 추출 (안전하게 매핑)
        db_id = row_dict.get("id") or idx + 1
        url = row_dict.get("url") or ""
        company = row_dict.get("company_name") or ""
        position = row_dict.get("position") or ""
        metadata_fields = {
            key: row_dict.get(key)
            for key in ("posted_at", "posted_at_text", "deadline", "created_at")
            if key in row_dict and row_dict.get(key) not in (None, "")
        }
        metadata_xml = "\n".join(
            f"    <{key}>{xml_text(value)}</{key}>"
            for key, value in metadata_fields.items()
        )
        
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

        content = str(content)
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS] + "\n[내용 일부 생략]"

        doc_xml = (
            f'<document id="{xml_text(db_id)}">\n'
            f'  <source_url>{xml_text(url)}</source_url>\n'
            f'  <company>{xml_text(company)}</company>\n'
            f'  <position>{xml_text(position)}</position>\n'
            f'  <metadata>\n{metadata_xml}\n  </metadata>\n'
            f'  <content>\n{xml_text(content)}\n  </content>\n'
            f'</document>'
        )
        context_parts.append(doc_xml)

    if truncated_rows:
        context_parts.append(
            f"<notice>결과가 {MAX_RESULT_ROWS}건을 초과하여 상위 {MAX_RESULT_ROWS}건만 반환했습니다.</notice>"
        )

    return f"{opening_tag}\n" + "\n\n".join(context_parts) + "\n</query_result>"

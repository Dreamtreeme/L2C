import re
import sqlite3
import logging
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from agent.utils.query_cache import EmbeddingLRUCache
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
embedding_cache = EmbeddingLRUCache(max_size=128)

class QueryFilters(BaseModel):
    company: Optional[str] = Field(None, description="질문에서 언급된 특정 기업/회사 이름 (예: '토스', '카카오', '로이드케이', '네이버')")
    tech_stack: Optional[str] = Field(None, description="질문에서 요구하는 기술 스택이나 직무 키워드 (예: 'iOS', 'Android', 'AI', 'Python', 'Swift', 'Kotlin')")
    career: Optional[int] = Field(None, description="질문에서 명시한 최소 경력 요구 연차 숫자 (예: '3년 이상' -> 3, '신입' -> 0)")
    days_limit: Optional[int] = Field(None, description="질문에서 필터링을 요구하는 시간적 범위 제약 일수 (예: '3개월 내' -> 90, '최근 한 달' -> 30)")
    semantic_query: str = Field(..., description="필터용 속성들을 제외하고, 순수 의미론적 임베딩 검색에 활용할 정제된 쿼리 스트링")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터 간의 코사인 유사도를 계산합니다."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def parse_filters(query: str) -> Dict[str, Any]:
    """
    자연어 쿼리에서 회사명, 기술 스택, 경력 연차, 시간적 제약 조건을 LLM(Gemini 3.5 Flash)을 사용하여 구조화된 필터로 추출합니다.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
        structured_llm = llm.with_structured_output(QueryFilters)
        
        prompt = (
            f"사용자 질문: {query}\n\n"
            f"위 질문에서 필터링 조건(회사명, 기술 스택, 경력 연차, 시간 범위)을 추출하고, "
            f"임베딩 기반 검색에 사용될 핵심 의미 쿼리(semantic_query)를 필터링 키워드를 제거하여 정제한 상태로 작성해 주세요."
        )
        
        result = structured_llm.invoke(prompt)
        
        filters = {}
        if result:
            if result.company:
                filters["company"] = result.company.strip()
            if result.tech_stack:
                filters["tech_stack"] = result.tech_stack.strip()
            if result.career is not None:
                filters["career"] = result.career
            if result.days_limit is not None:
                filters["days_limit"] = result.days_limit
            if result.semantic_query:
                filters["semantic_query"] = result.semantic_query.strip()
            else:
                filters["semantic_query"] = query
                
        logger.info(f"LLM parsed filters: {filters}")
        return filters
    except Exception as e:
        logger.error(f"Failed to parse filters using LLM: {e}")
        # 오류 발생 시 빈 dict를 반환하여 전체 검색으로 정상 폴백 처리
        return {}

def retrieve(query: str, filters: Dict[str, Any], db_path: str | Path, top_k: int = 5) -> Dict[str, Any]:
    """
    1차 SQL Hard Filter 및 2차 Numpy Cosine Similarity (768차원) 연산을 실행하고,
    Top-1 vs Top-2~5 평균 스코어 갭 기반의 Pre-LLM Check로 검증 결과 데이터를 리턴합니다.
    """
    # 1. 쿼리 임베딩 획득 (LRU 캐시 연동)
    db_path = Path(db_path)
    
    # filters에 semantic_query가 전달되었으면 임베딩 검색 쿼리로 사용, 없으면 query 사용
    embed_query = filters.get("semantic_query", query) or query
    
    cached_vector = embedding_cache.get(embed_query)
    if cached_vector is not None:
        query_vector = np.array(cached_vector, dtype=np.float32)
    else:
        try:
            logger.info("Requesting embedding via text-embedding-004...")
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            raw_vector = embeddings_model.embed_query(embed_query)
            # 768차원 확인
            if len(raw_vector) != 768:
                logger.error(f"Returned embedding dimension is {len(raw_vector)} instead of 768")
                return {"is_empty": True, "results": []}
                
            embedding_cache.set(embed_query, raw_vector)
            query_vector = np.array(raw_vector, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return {"is_empty": True, "results": []}

    # 2. SQLite 1차 Hard Filter 조합 질의
    # collected_at 대신 schema에 있는 created_at 사용
    sql = """
    SELECT id, company_name, position, tech_stack, career_min, career_max, raw_ocr_text, url, created_at, embedding 
    FROM jobs 
    WHERE 1=1
    """
    params = []

    if "company" in filters:
        sql += " AND company_name LIKE ?"
        params.append(f"%{filters['company']}%")
        
    if "tech_stack" in filters:
        tech = filters["tech_stack"]
        sql += " AND (tech_stack LIKE ? OR position LIKE ?)"
        params.append(f"%{tech}%")
        params.append(f"%{tech}%")
        
    if "career" in filters:
        career_val = filters["career"]
        sql += " AND career_min <= ? AND career_max >= ?"
        params.append(career_val)
        params.append(career_val)
        
    if "days_limit" in filters:
        days_val = filters["days_limit"]
        limit_date = (datetime.now() - timedelta(days=days_val)).isoformat(timespec="seconds")
        sql += " AND created_at >= ?"
        params.append(limit_date)

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        logger.debug(f"SQLite filter search returned {len(rows)} candidate rows")
    except Exception as db_err:
        logger.error(f"SQLite query execution error: {db_err}")
        return {"is_empty": True, "results": []}

    if not rows:
        return {"is_empty": True, "results": []}

    # 3. 2차 Numpy Cosine Similarity 계산
    candidates = []
    for row in rows:
        db_id = row["id"]
        embedding_blob = row["embedding"]
        if not embedding_blob:
            logger.warning(f"Skip job id {db_id}: embedding field is empty")
            continue
            
        try:
            doc_vector = np.frombuffer(embedding_blob, dtype=np.float32)
            if doc_vector.shape[0] != 768:
                logger.warning(f"Skip job id {db_id}: dimension mismatch {doc_vector.shape[0]}")
                continue
                
            score = cosine_similarity(query_vector, doc_vector)
            
            # raw_ocr_text가 비어있으면 raw_json의 내용을 사용하거나 position/requirements 등 수동 구성
            raw_text = row["raw_ocr_text"] or f"회사명: {row['company_name']}\n직무: {row['position']}\n기술스택: {row['tech_stack']}"
            
            candidates.append({
                "id": db_id,
                "company_name": row["company_name"],
                "position": row["position"],
                "raw_text": raw_text,
                "url": row["url"],
                "collected_at": row["created_at"],
                "score": score
            })
        except Exception as similarity_err:
            logger.error(f"Failed similarity calculation for job {db_id}: {similarity_err}")
            continue

    if not candidates:
        return {"is_empty": True, "results": []}

    # 스코어 내림차순 정렬 및 Top-K 추출
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_results = candidates[:top_k]

    # 4. Score Gap 기반 Pre-LLM Check
    n = len(top_results)
    top1_score = top_results[0]["score"]

    if n == 1:
        # 단일 문서 매칭 시 절대 기준 0.40 검사
        if top1_score < 0.40:
            logger.info(f"Pre-LLM Check: Rejection. Top-1 score {top1_score:.3f} < 0.40")
            return {"is_empty": True, "results": []}
        gap = top1_score
    else:
        # Top-2~5 평균과의 Gap 계산
        other_scores = [item["score"] for item in top_results[1:]]
        avg_others = sum(other_scores) / len(other_scores)
        gap = top1_score - avg_others
        logger.info(f"Pre-LLM Check - Top1: {top1_score:.3f}, AvgOthers({len(other_scores)}): {avg_others:.3f}, Gap: {gap:.3f}")
        
        if gap < 0.15:
            logger.info(f"Pre-LLM Check: Rejection. Score Gap {gap:.3f} < 0.15")
            return {"is_empty": True, "results": []}

    return {"is_empty": False, "results": top_results}

import sqlite3
import logging
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
from langchain_core.tools import tool

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from agent.utils.query_cache import EmbeddingLRUCache
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
embedding_cache = EmbeddingLRUCache(max_size=128)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터 간의 코사인 유사도를 계산합니다."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

@tool
def rag_search(
    query: str,
    company: str = None,
    tech_stack: str = None,
    career: int = None,
    days_limit: int = None
) -> str:
    """
    데이터베이스 내의 채용 공고를 검색하는 도구입니다.
    사용자의 질문 의미 쿼리(query)와 필터 조건(company, tech_stack, career, days_limit)을 활용하여
    데이터베이스에서 가장 유사하고 관련 있는 채용 공고 본문 텍스트 목록(XML 형식)을 반환합니다.
    """
    from shared.config import DB_PATH
    db_path = Path(DB_PATH)

    logger.info(f"[rag_search] Called with query='{query}', company='{company}', tech_stack='{tech_stack}', career={career}, days_limit={days_limit}")

    # 1. 쿼리 임베딩 획득
    cached_vector = embedding_cache.get(query)
    if cached_vector is not None:
        query_vector = np.array(cached_vector, dtype=np.float32)
    else:
        try:
            logger.info("[rag_search] Requesting embedding via GoogleGenerativeAIEmbeddings (models/text-embedding-004)...")
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            raw_vector = embeddings_model.embed_query(query)
            if len(raw_vector) != 768:
                logger.error(f"[rag_search] Returned embedding dimension is {len(raw_vector)} instead of 768")
                return "검색 오류: 임베딩 차원 오류"
                
            embedding_cache.set(query, raw_vector)
            query_vector = np.array(raw_vector, dtype=np.float32)
        except Exception as e:
            logger.warning(f"[rag_search] Failed to generate embedding: {e}. Fallback to dummy zero vector.")
            query_vector = np.zeros(768, dtype=np.float32)

    # 2. SQLite 1차 Hard Filter 조합 질의
    sql = """
    SELECT id, company_name, position, tech_stack, experience_min, experience_max, raw_ocr_text, url, created_at, embedding 
    FROM jobs 
    WHERE 1=1
    """
    params = []

    if company:
        sql += " AND company_name LIKE ?"
        params.append(f"%{company}%")
        
    if tech_stack:
        sql += " AND (tech_stack LIKE ? OR position LIKE ?)"
        params.append(f"%{tech_stack}%")
        params.append(f"%{tech_stack}%")
        
    if career is not None:
        sql += " AND experience_min <= ? AND experience_max >= ?"
        params.append(career)
        params.append(career)
        
    if days_limit is not None:
        limit_date = (datetime.now() - timedelta(days=days_limit)).isoformat(timespec="seconds")
        sql += " AND created_at >= ?"
        params.append(limit_date)

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        logger.info(f"[rag_search] SQLite filter search returned {len(rows)} candidate rows")
    except Exception as db_err:
        logger.error(f"[rag_search] SQLite query execution error: {db_err}")
        return "검색 오류: DB 쿼리 실행 실패"

    if not rows:
        return "검색 결과가 없습니다. 관련 조건의 채용 공고가 데이터베이스에 존재하지 않습니다."

    # 3. 2차 Numpy Cosine Similarity 계산
    candidates = []
    for row in rows:
        db_id = row["id"]
        embedding_blob = row["embedding"]
        if not embedding_blob:
            continue
            
        try:
            doc_vector = np.frombuffer(embedding_blob, dtype=np.float32)
            if doc_vector.shape[0] != 768:
                continue
                
            score = cosine_similarity(query_vector, doc_vector)
            raw_text = row["raw_ocr_text"] or f"회사명: {row['company_name']}\n직무: {row['position']}\n기술스택: {row['tech_stack']}"
            
            candidates.append({
                "id": db_id,
                "company_name": row["company_name"],
                "position": row["position"],
                "raw_text": raw_text,
                "url": row["url"],
                "score": score
            })
        except Exception as similarity_err:
            logger.error(f"[rag_search] Failed similarity calculation: {similarity_err}")
            continue

    if not candidates:
        return "검색 결과가 없습니다. 매칭 가능한 유효한 데이터가 없습니다."

    # 스코어 내림차순 정렬 및 Top-K 추출
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_results = candidates[:5]

    # Pre-LLM Check Score Gap을 판단하여 변별력 낮음에 대한 경고 메타를 제공하되, 검색 결과를 원천 차단하지 않음
    n = len(top_results)
    top1_score = top_results[0]["score"]
    is_weak = False
    
    if n == 1:
        if top1_score < 0.40:
            is_weak = True
    else:
        other_scores = [item["score"] for item in top_results[1:]]
        avg_others = sum(other_scores) / len(other_scores)
        gap = top1_score - avg_others
        if gap < 0.15:
            is_weak = True

    # 4. XML 포맷 텍스트 컨텍스트로 직렬화하여 반환
    context_parts = []
    if is_weak:
        context_parts.append("<!-- 주의: 검색된 아래 공고들의 유사도 변별력이 다소 낮습니다. 질문과 정확히 부합하는지 비판적으로 분석하십시오. -->")
        
    for item in top_results:
        doc_xml = (
            f'<document id="{item["id"]}">\n'
            f'  <source_url>{item["url"]}</source_url>\n'
            f'  <company>{item["company_name"]}</company>\n'
            f'  <position>{item["position"]}</position>\n'
            f'  <content>\n{item["raw_text"]}\n  </content>\n'
            f'</document>'
        )
        context_parts.append(doc_xml)
    
    return "\n\n".join(context_parts)

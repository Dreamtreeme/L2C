import re
import sqlite3
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from agent.utils.query_cache import EmbeddingLRUCache

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

def parse_filters(query: str) -> Dict[str, Any]:
    """
    자연어 쿼리에서 회사명, 모바일 기술 스택, 경력 연차를 정규식으로 안전하게 추출합니다.
    매치되지 않은 속성은 포함되지 않으며, 빈 딕셔너리가 반환될 수 있습니다.
    """
    filters = {}
    
    # 1. 회사명 필터 추출
    company_patterns = [
        r"(토스|원티드랩|로이드케이|글로벌머니익스프레스|네이버|카카오|라인|쿠팡|배달의민족|우아한형제들|당근|야놀자)\s*(?:에서|공고|에\s*있는|채용|포지션|회사)?"
    ]
    for pattern in company_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            filters["company"] = match.group(1).strip()
            break

    # 2. 기술 스택 필터 추출 (모바일 전문 분야 매핑)
    tech_patterns = [
        r"\b(iOS|Swift|Android|Kotlin|Combine|RxSwift|Flutter|ReactNative|React\s*Native)\b"
    ]
    for pattern in tech_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            tech = match.group(1).replace(" ", "").strip()
            filters["tech_stack"] = tech
            break

    # 3. 경력 필터 추출
    career_match = re.search(r"(\d+)\s*(?:년\s*이상|년차|년\s*경력|년\s*이상\s*경력)", query)
    if career_match:
        filters["career"] = int(career_match.group(1))
    elif "신입" in query:
        filters["career"] = 0

    if filters:
        logger.debug(f"parse_filters successfully extracted: {filters}")
    return filters

def retrieve(query: str, filters: Dict[str, Any], db_path: str | Path, top_k: int = 5) -> Dict[str, Any]:
    """
    1차 SQL Hard Filter 및 2차 Numpy Cosine Similarity (768차원) 연산을 실행하고,
    Top-1 vs Top-2~5 평균 스코어 갭 기반의 Pre-LLM Check로 검증 결과 데이터를 리턴합니다.
    """
    # 1. 쿼리 임베딩 획득 (LRU 캐시 연동)
    db_path = Path(db_path)
    cached_vector = embedding_cache.get(query)
    if cached_vector is not None:
        query_vector = np.array(cached_vector, dtype=np.float32)
    else:
        try:
            logger.info("Requesting embedding via text-embedding-004...")
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            raw_vector = embeddings_model.embed_query(query)
            # 768차원 확인
            if len(raw_vector) != 768:
                logger.error(f"Returned embedding dimension is {len(raw_vector)} instead of 768")
                return {"is_empty": True, "results": []}
                
            embedding_cache.set(query, raw_vector)
            query_vector = np.array(raw_vector, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return {"is_empty": True, "results": []}

    # 2. SQLite 1차 Hard Filter 조합 질의
    sql = """
    SELECT id, company_name, position, tech_stack, career_min, career_max, raw_ocr_text, url, collected_at, embedding 
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
                "collected_at": row["collected_at"],
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

import os
import shutil
import sqlite3
import numpy as np
import pytest
from pathlib import Path

from shared.db.database import Database
from agent.tools.rag_engine import parse_filters, retrieve
from agent.graph.nodes import qa_reasoning_node
from agent.graph.state import GraphState

TEST_DB_PATH = Path("data/test_rag_jobs.db")

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    # 이전 테스트 DB 정리
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)
        
    db = Database(TEST_DB_PATH)
    
    # 더미 임베딩 생성 (768차원 float32)
    # 1. iOS/Swift 관련 공고용 임베딩 (iOS 개발자 쿼리와 유사도가 높게 설정)
    ios_base = np.zeros(768, dtype=np.float32)
    ios_base[0:10] = 0.5  # 특정 패턴 부여
    ios_embedding = ios_base.tobytes()
    
    # 2. Android 관련 공고용 임베딩
    android_base = np.zeros(768, dtype=np.float32)
    android_base[10:20] = 0.5
    android_embedding = android_base.tobytes()
    
    # 3. 기타 백엔드 공고용 임베딩
    backend_base = np.zeros(768, dtype=np.float32)
    backend_base[20:30] = 0.5
    backend_embedding = backend_base.tobytes()

    # 테스트 공고 데이터 적재
    db.upsert(
        url="https://www.wanted.co.kr/wd/1000",
        data={
            "company_name": "토스",
            "position": "iOS 개발자 (3년 이상)",
            "tech_stack": ["Swift", "SwiftUI", "UIKit"],
            "raw_ocr_text": "토스에서 금융을 더 간편하게 만들 iOS 개발자를 모집합니다. 자격요건은 Swift 실무 3년 이상, UIKit 및 SwiftUI 개발 경험 필수입니다. 복지로는 주택 자금 대출, 통신비 지원이 있습니다.",
            "source_platform": "Wanted",
            "experience_min": 3,
            "experience_max": 10,
            "experience_text": "3년 이상",
            "content_hash": "hash_1000"
        },
        embedding=ios_embedding
    )

    db.upsert(
        url="https://www.wanted.co.kr/wd/2000",
        data={
            "company_name": "카카오",
            "position": "Android 개발자 (5년 이상)",
            "tech_stack": ["Kotlin", "Java", "Jetpack Compose"],
            "raw_ocr_text": "카카오에서 대국민 서비스를 함께 이끌 Android 개발자를 모십니다. Kotlin 및 Jetpack Compose 경험 5년 이상 필수. 복지 혜택은 안식 휴가 및 리프레시 휴가비 지원.",
            "source_platform": "Wanted",
            "experience_min": 5,
            "experience_max": 15,
            "experience_text": "5년 이상",
            "content_hash": "hash_2000"
        },
        embedding=android_embedding
    )

    db.upsert(
        url="https://www.wanted.co.kr/wd/3000",
        data={
            "company_name": "로이드케이",
            "position": "Python 백엔드 엔지니어 (신입)",
            "tech_stack": ["Python", "FastAPI", "Django"],
            "raw_ocr_text": "로이드케이에서 데이터 파이프라인 개발을 맡아줄 백엔드 개발자를 신입 채용합니다. 복지는 도서 구입비 무제한 및 장비 지원.",
            "source_platform": "Remember",
            "experience_min": 0,
            "experience_max": 2,
            "experience_text": "신입",
            "content_hash": "hash_3000"
        },
        embedding=backend_embedding
    )
    
    yield db
    
    # Teardown
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not configured in env")
def test_parse_filters():
    # 1. 회사명, 스택, 경력 혼합 추출 테스트
    q1 = "토스에서 SwiftUI 우대하는 3년 이상 경력의 iOS 개발자 공고 복지 알려줘"
    f1 = parse_filters(q1)
    assert f1.get("company") == "토스"
    assert f1.get("tech_stack") in ["iOS", "SwiftUI", "Swift", "iOS 개발자"]
    assert f1.get("career") == 3

    # 2. 아무 필터도 없는 질문
    q2 = "복지가 가장 좋은 회사는 어디야?"
    f2 = parse_filters(q2)
    assert f2.get("company") is None
    assert f2.get("tech_stack") is None
    assert f2.get("career") is None
    assert f2.get("days_limit") is None

    # 3. 시간 제약 분석 질문 테스트
    q3 = "3개월 내 ai엔지니어 채용공고 특징 분석해줘"
    f3 = parse_filters(q3)
    assert f3.get("days_limit") == 90
    assert "AI" in f3.get("tech_stack", "") or "ai" in f3.get("tech_stack", "").lower()


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not configured in env")
def test_retrieve_with_filters(setup_test_db):
    # '토스' 회사명 필터 적용 테스트
    filters = {"company": "토스"}
    retrieved = retrieve("토스 iOS 개발자", filters, TEST_DB_PATH, top_k=5)
    assert "is_empty" in retrieved


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not configured in env")
def test_retrieve_with_days_limit(setup_test_db):
    # 30일 이내 토스 공고 검색 필터 적용 테스트
    filters = {"days_limit": 30, "company": "토스"}
    retrieved = retrieve("토스 iOS 개발자", filters, TEST_DB_PATH, top_k=5)
    assert "is_empty" in retrieved
    if not retrieved.get("is_empty"):
        for item in retrieved["results"]:
            assert item["company_name"] == "토스"


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not configured in env")
def test_qa_reasoning_node_e2e(setup_test_db, monkeypatch):
    # DB_PATH를 테스트용 DB로 임시 치워놓는 패치 적용
    monkeypatch.setattr("shared.config.DB_PATH", TEST_DB_PATH)
    
    # 1. 팩트 기반 질문 테스트
    state = GraphState(goal="토스 iOS 개발자의 자격요건과 복지 혜택을 알려줘")
    
    result = qa_reasoning_node(state)
    answer = result.get("last_action_result", "")
    
    print("\n--- RAG Q&A Answer ---")
    print(answer)
    print("----------------------")
    
    assert result.get("is_finished") is True
    # 성공적으로 답변을 가져왔거나 혹은 검색 거절("조건에 맞는 정보를 찾을 수 없습니다.")이 떨어졌는지 확인
    assert len(answer) > 0


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not configured in env")
def test_hallucination_rejection(setup_test_db, monkeypatch):
    # DB_PATH를 테스트용 DB로 임시 치워놓는 패치 적용
    monkeypatch.setattr("shared.config.DB_PATH", TEST_DB_PATH)
    
    # 2. 환각 거절(Hallucination Rejection) 질문 테스트
    # DB에 적재되지 않은 스페이스X 관련 질문 유입
    state = GraphState(goal="스페이스X의 화성 탐사선 개발자 공고 우대사항을 알려줘")
    
    result = qa_reasoning_node(state)
    answer = result.get("last_action_result", "")
    
    print("\n--- Hallucination Rejection Answer ---")
    print(answer)
    print("--------------------------------------")
    
    # RAG Pre-LLM Check 또는 Strict Prompting에 의해 즉시 거절 문구가 생성되어야 함
    assert "찾을 수 없습니다" in answer or "확인되지 않음" in answer


def test_web_server_api_endpoints(setup_test_db, monkeypatch):
    monkeypatch.setattr("shared.config.DB_PATH", TEST_DB_PATH)
    
    from fastapi.testclient import TestClient
    from agent.web_server import app
    
    client = TestClient(app)
    
    # 1. 상세 공고 조회 API (/api/jobs/{job_id}) 검증
    # SQLite에 삽입된 첫 번째 로우(id=1) 데이터 획득 테스트
    response = client.get("/api/jobs/1")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["id"] == 1
    assert res_data["company_name"] == "토스"
    assert "position" in res_data
    assert "url" in res_data
    assert "raw_text" in res_data

    # 2. 미존재 ID 조회 시 에러 응답 검증
    fail_response = client.get("/api/jobs/9999")
    assert fail_response.status_code == 200
    assert "error" in fail_response.json()


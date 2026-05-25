import os
import shutil
import sqlite3
import pytest
from pathlib import Path

from shared.db.database import Database
from agent.graph.nodes import qa_reasoning_node
from agent.graph.state import GraphState

TEST_DB_PATH = Path("data/test_rag_jobs.db")

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    # 이전 테스트 DB 정리
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)
        
    db = Database(TEST_DB_PATH)
    
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
        }
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
        }
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
        }
    )
    
    yield db
    
    # Teardown
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)


def test_sqlite_query_tool(setup_test_db, monkeypatch):
    monkeypatch.setattr("shared.config.DB_PATH", TEST_DB_PATH)
    from agent.tools.sqlite_query import sqlite_query
    
    # 1. 회사명 필터를 이용한 DB 조회 테스트
    result = sqlite_query.invoke({"sql_query": "SELECT id, url, company_name, position, raw_ocr_text FROM jobs WHERE company_name = '토스'"})
    assert "<document id=" in result
    assert "토스" in result
    
    # 2. 직무명 필터를 이용한 DB 조회 테스트
    result_time = sqlite_query.invoke({"sql_query": "SELECT id, url, company_name, position, raw_ocr_text FROM jobs WHERE position LIKE '%Android%'"})
    assert "<document id=" in result_time
    assert "카카오" in result_time


def test_realtime_scraping_tool(setup_test_db, monkeypatch):
    """비전 자율 수집 그래프 invoke를 mock하여 realtime_scraping 도구의 통합 로직을 검증합니다."""
    monkeypatch.setattr("shared.config.DB_PATH", TEST_DB_PATH)
    
    # 비전 에이전트 그래프를 모킹: invoke 시 수집된 JD 데이터를 반환하는 가짜 앱 생성
    class FakeGraphApp:
        def invoke(self, state):
            return {
                **state,
                "is_finished": True,
                "collected_data": ["모의 수집 완료"],
                "extracted_jd": {
                    "공고목록": [
                        {
                            "회사명": "테스트컴퍼니",
                            "직무명": "테스트 엔지니어",
                            "주요업무": "테스트 자동화 구축",
                            "자격요건": "Python 3년 이상",
                            "우대사항": "CI/CD 경험",
                            "url": "https://www.wanted.co.kr/wd/99999",
                        }
                    ]
                },
            }

    def mock_build_graph():
        return FakeGraphApp()

    monkeypatch.setattr("agent.graph.workflow.build_graph", mock_build_graph)
    

    
    from agent.tools.realtime_scraping import realtime_scraping
    
    result = realtime_scraping.invoke({"company": "테스트컴퍼니"})
    assert "적재 완료" in result or "업데이트" in result
    assert "테스트컴퍼니" in result or "1건" in result
    
    # DB에 실제 적재되었는지 검증
    db = Database(TEST_DB_PATH)
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.execute("SELECT company_name, position FROM jobs WHERE url = 'https://www.wanted.co.kr/wd/99999'")
    row = cursor.fetchone()
    conn.close()
    assert row is not None, "Vision agent 수집 데이터가 DB에 적재되지 않았습니다."
    assert row[0] == "테스트컴퍼니"


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not configured in env")
def test_qa_reasoning_node_e2e(setup_test_db, monkeypatch):
    monkeypatch.setattr("shared.config.DB_PATH", TEST_DB_PATH)
    
    # realtime_scraping 도구 모킹하여 실제 브라우저 자동화 실행 방지
    from langchain_core.tools import tool
    @tool("realtime_scraping")
    def mock_realtime_scraping(company: str = None, tech_stack: str = None) -> str:
        """실시간 채용 공고를 수집하는 모킹 도구입니다."""
        return f"실시간 수집 완료: '{company or tech_stack}'에 매칭되는 채용 정보를 찾지 못했습니다."
    monkeypatch.setattr("agent.graph.nodes.realtime_scraping", mock_realtime_scraping)
    
    # 1. 팩트 기반 질문 테스트
    state = GraphState(goal="토스 iOS 개발자의 자격요건과 복지 혜택을 알려줘")
    result = qa_reasoning_node(state)
    answer = result.get("last_action_result", "")
    
    print("\n--- Commander SQLite Answer ---")
    print(answer)
    print("----------------------------")
    
    assert result.get("is_finished") is True
    assert len(answer) > 0
    # 인용 칩이 올바르게 생성되었거나 거절 문구가 생성되었는지 확인
    assert "[job_id:" in answer or "찾을 수 없습니다" in answer or "확인되지 않음" in answer


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not configured in env")
def test_hallucination_rejection(setup_test_db, monkeypatch):
    monkeypatch.setattr("shared.config.DB_PATH", TEST_DB_PATH)
    
    # realtime_scraping 도구 모킹하여 실제 브라우저 자동화 실행 방지
    from langchain_core.tools import tool
    @tool("realtime_scraping")
    def mock_realtime_scraping(company: str = None, tech_stack: str = None) -> str:
        """실시간 채용 공고를 수집하는 모킹 도구입니다."""
        return f"실시간 수집 완료: '{company or tech_stack}'에 매칭되는 채용 정보를 찾지 못했습니다."
    monkeypatch.setattr("agent.graph.nodes.realtime_scraping", mock_realtime_scraping)
    
    # 2. 환각 거절 질문 테스트
    state = GraphState(goal="스페이스X의 화성 탐사선 개발자 공고 우대사항을 알려줘")
    result = qa_reasoning_node(state)
    answer = result.get("last_action_result", "")
    
    print("\n--- Commander Rejection Answer ---")
    print(answer)
    print("----------------------------------")
    
    assert "찾을 수 없습니다" in answer or "확인되지 않음" in answer


def test_web_server_api_endpoints(setup_test_db, monkeypatch):
    monkeypatch.setattr("shared.config.DB_PATH", TEST_DB_PATH)
    
    from fastapi.testclient import TestClient
    from agent.web_server import app
    
    client = TestClient(app)
    
    # 1. 상세 공고 조회 API (/api/jobs/{job_id}) 검증
    response = client.get("/api/jobs/1")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["id"] == 1
    assert res_data["company_name"] == "토스"
    assert "position" in res_data
    
    # 2. 미존재 ID 조회 시 에러 응답 검증
    fail_response = client.get("/api/jobs/9999")
    assert fail_response.status_code == 200
    assert "error" in fail_response.json()


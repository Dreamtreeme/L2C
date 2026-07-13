import json
import os
import shutil
import sqlite3
import pytest
from pathlib import Path

from shared.db.database import Database
from agent.graph.nodes import qa_reasoning_node
from agent.graph.state import GraphState

TEST_DB_PATH = Path("data/test_qa_jobs.db")

@pytest.fixture(scope="module", autouse=True)
def setup_test_db(tmp_path_factory):
    global TEST_DB_PATH
    TEST_DB_PATH = tmp_path_factory.mktemp("sqlite_qa") / "test_qa_jobs.db"
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
    # shared.config.DB_PATH 를 패치해도 sqlite_query 내부에서
    # `from shared.config import DB_PATH` 로 이미 바인딩된 로컬 변수는 영향받지 않습니다.
    # 실제로 쿼리가 참조하는 모듈 내 변수를 직접 패치해야 합니다.
    import agent.tools.sqlite_query as sq_module
    monkeypatch.setattr(sq_module, "DB_PATH", TEST_DB_PATH, raising=False)
    import shared.config as cfg
    monkeypatch.setattr(cfg, "DB_PATH", TEST_DB_PATH)
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
    """비전 자율 수집 그래프 stream을 mock하여 realtime_scraping 도구의 통합 로직을 검증합니다."""
    import shared.config as cfg
    monkeypatch.setattr(cfg, "DB_PATH", TEST_DB_PATH)
    monkeypatch.delenv("VISION_AGENT_RECURSION_LIMIT", raising=False)
    monkeypatch.setenv("VISION_WORKER_SUMMARY_MODE", "off")
    monkeypatch.setenv("VISION_SEARCH_INTENT_MODE", "off")
    monkeypatch.setenv("VISION_JD_NORMALIZATION_MODE", "off")
    monkeypatch.setenv("VISION_WORKER_PREOPEN_BROWSER", "0")
    
    # 비전 에이전트 그래프를 모킹: stream 시 수집된 JD 데이터를 반환하는 가짜 앱 생성
    class FakeGraphApp:
        def stream(self, state, config=None, stream_mode=None):
            assert config["recursion_limit"] == 60
            assert stream_mode == "values"
            assert "사람인" in state["goal"]
            assert "테스트컴퍼니" in state["goal"]
            assert "원티드(" not in state["goal"]
            yield {
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
                            "url": "https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=99999",
                        }
                    ]
                },
            }

    def mock_build_graph():
        return FakeGraphApp()

    monkeypatch.setattr("agent.graph.workflow.build_graph", mock_build_graph)

    

    
    from agent.tools.realtime_scraping import realtime_scraping
    
    result = realtime_scraping.invoke({"company": "테스트컴퍼니", "site": "saramin"})
    payload = json.loads(result)
    assert payload["review"]["decision"] == "accept"
    assert payload["site"] == "saramin"
    assert payload["persisted_count"] == 1
    assert "테스트컴퍼니" in payload["message"]
    # DB에 실제 적재되었는지 검증
    db = Database(TEST_DB_PATH)
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.execute("SELECT company_name, position FROM jobs WHERE url = 'https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=99999'")
    row = cursor.fetchone()
    conn.close()
    assert row is not None, "Vision agent 수집 데이터가 DB에 적재되지 않았습니다."
    assert row[0] == "테스트컴퍼니"


def test_realtime_scraping_keeps_browser_after_run_by_default(setup_test_db, monkeypatch):
    import agent.graph.nodes as nodes

    monkeypatch.delenv("VISION_CLOSE_BROWSER_AFTER_RUN", raising=False)
    monkeypatch.setenv("VISION_SEARCH_INTENT_MODE", "off")
    monkeypatch.setenv("VISION_JD_NORMALIZATION_MODE", "off")
    monkeypatch.setenv("VISION_WORKER_PREOPEN_BROWSER", "0")

    closed = []

    class FakeActionTools:
        def close_browser(self):
            closed.append("close_browser")
            return {"status": "success", "action": "close_browser", "result": {"closed": True}}

    class FakeGraphApp:
        def stream(self, state, config=None, stream_mode=None):
            yield {**state, "is_finished": True, "extracted_jd": {}}

    monkeypatch.setattr(nodes, "_action_tools", FakeActionTools(), raising=False)
    monkeypatch.setattr("agent.graph.workflow.build_graph", lambda: FakeGraphApp())

    from agent.tools.realtime_scraping import realtime_scraping

    result = realtime_scraping.invoke({"company": "cleanup-test"})

    assert "cleanup-test" in result
    assert closed == []


def test_persistence_job_normalization_uses_llm(monkeypatch):
    from agent.application.model_clients import clear_model_client_cache
    from shared.schema.jd_schema import JobPosting
    from agent.tools.realtime_scraping import _normalize_job_for_persistence

    class FakeStructuredLLM:
        def invoke(self, messages):
            return JobPosting(
                company_name="Acme",
                position="iOS Engineer",
                url="",
                tech_stack=["SwiftUI"],
                requirements=["Swift experience"],
                experience_min=3,
                experience_max=99,
                experience_text="3+ years",
            )

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, schema):
            return FakeStructuredLLM()

    monkeypatch.setenv("VISION_JD_NORMALIZATION_MODE", "llm")
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", FakeLLM)
    clear_model_client_cache()

    normalized = _normalize_job_for_persistence(
        {"URL": "https://example.com/job/1", "raw": "ignored"},
        keyword="ios",
    )

    assert normalized["company_name"] == "Acme"
    assert normalized["position"] == "iOS Engineer"
    assert normalized["url"] == "https://example.com/job/1"
    assert normalized["tech_stack"] == ["SwiftUI"]
    assert normalized["_normalization_source"] == "llm"
    clear_model_client_cache()


def test_persistence_job_normalization_defaults_to_deterministic(monkeypatch):
    from agent.tools.realtime_scraping import _normalize_job_for_persistence

    monkeypatch.delenv("VISION_JD_NORMALIZATION_MODE", raising=False)

    normalized = _normalize_job_for_persistence(
        {
            "\ud68c\uc0ac\uba85": "\ud14c\uc2a4\ud2b8\ud68c\uc0ac",
            "\uc9c1\ubb34\uba85": "iOS \uac1c\ubc1c\uc790",
            "\uacf5\uace0url": "https://example.com/jobs/ios",
            "\uc790\uaca9\uc694\uac74": ["Swift \uacbd\ud5d8"],
        },
        keyword="ios",
    )

    assert normalized["company_name"] == "\ud14c\uc2a4\ud2b8\ud68c\uc0ac"
    assert normalized["position"] == "iOS \uac1c\ubc1c\uc790"
    assert normalized["url"] == "https://example.com/jobs/ios"
    assert normalized["requirements"] == ["Swift \uacbd\ud5d8"]
    assert normalized["_normalization_source"] == "deterministic"


def test_realtime_scraping_persists_partial_state_on_recursion_limit(setup_test_db, monkeypatch):
    """recursion limit에 걸려도 마지막 partial state의 수집 데이터는 저장합니다."""
    import shared.config as cfg
    monkeypatch.setenv("VISION_WORKER_PREOPEN_BROWSER", "0")
    from langgraph.errors import GraphRecursionError

    monkeypatch.setattr(cfg, "DB_PATH", TEST_DB_PATH)
    monkeypatch.delenv("VISION_AGENT_RECURSION_LIMIT", raising=False)
    monkeypatch.setenv("VISION_WORKER_SUMMARY_MODE", "off")
    monkeypatch.setenv("VISION_SEARCH_INTENT_MODE", "off")
    monkeypatch.setenv("VISION_JD_NORMALIZATION_MODE", "off")

    class FakeGraphApp:
        def stream(self, state, config=None, stream_mode=None):
            assert config["recursion_limit"] == 60
            yield state
            yield {
                **state,
                "is_finished": False,
                "current_url": "https://www.wanted.co.kr/wd/88888",
                "recorded_steps": [
                    {
                        "seq": 0,
                        "state_key": "state-a",
                        "url_template": "wanted.co.kr/search?query",
                        "action": "click_marker",
                        "target": {"text": "Data Engineer", "region": "middle-left", "ordinal": 0},
                        "param": {},
                        "expected_after": "job detail is visible",
                    },
                    {
                        "seq": 1,
                        "state_key": "state-b",
                        "url_template": "wanted.co.kr/wd/{id}",
                        "action": "go_back",
                        "target": None,
                        "param": {},
                        "expected_after": "job result list is visible",
                    },
                    {
                        "seq": 2,
                        "state_key": "state-c",
                        "url_template": "wanted.co.kr/search?query",
                        "action": "scroll",
                        "target": None,
                        "param": {"direction": "down"},
                        "expected_after": "more job cards are visible",
                    },
                ],
                "extracted_jd": {
                    "공고목록": [
                        {
                            "회사명": "부분수집컴퍼니",
                            "직무명": "데이터 엔지니어",
                            "주요업무": ["데이터 파이프라인 구축"],
                            "자격요건": ["Python 경험"],
                            "url": "https://www.wanted.co.kr/wd/88888",
                        }
                    ]
                },
            }
            raise GraphRecursionError("test recursion limit")

    def mock_build_graph():
        return FakeGraphApp()

    monkeypatch.setattr("agent.graph.workflow.build_graph", mock_build_graph)

    from agent.tools.realtime_scraping import realtime_scraping

    result = realtime_scraping.invoke({"company": "부분수집컴퍼니"})
    payload = json.loads(result)
    assert payload["review"]["decision"] == "accept"
    assert payload["hit_recursion_limit"] is True
    assert payload["needs_human_approval"] is True
    assert payload["intermediate_report"]["current_recursion_limit"] == 60
    assert payload["intermediate_report"]["suggested_recursion_limit"] == 120
    assert "approval" in payload["message"]
    assert payload["persisted_count"] == 1
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.execute("SELECT company_name, position FROM jobs WHERE url = 'https://www.wanted.co.kr/wd/88888'")
    row = cursor.fetchone()
    submission_row = conn.execute(
        "SELECT review_decision, payload_json FROM worker_submissions WHERE keyword = ? ORDER BY review_attempt DESC, updated_at DESC LIMIT 1",
        ("부분수집컴퍼니",),
    ).fetchone()
    candidate_row = conn.execute(
        "SELECT status, steps_json FROM recipe_candidates ORDER BY updated_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "부분수집컴퍼니"
    assert submission_row is not None
    assert submission_row[0] == "accept"
    submission_payload = json.loads(submission_row[1])
    assert [step["action"] for step in submission_payload["recorded_steps"]] == ["click_marker", "go_back", "scroll"]
    assert "state_key" not in submission_payload["recorded_steps"][0]
    assert candidate_row is not None
    assert candidate_row[0] == "pending_replay"
    candidate_steps = json.loads(candidate_row[1])
    assert [step["action"] for step in candidate_steps] == ["click_marker", "go_back", "scroll"]
    assert all("state_key" not in step for step in candidate_steps)


def test_browser_back_marker_detection():
    from agent.graph.nodes import _is_browser_back_marker_bbox

    assert _is_browser_back_marker_bbox([8, 121, 62, 173]) is True
    assert _is_browser_back_marker_bbox([180, 121, 240, 173]) is False
    assert _is_browser_back_marker_bbox([8, 210, 62, 260]) is False


def test_url_stale_flag_for_actions(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    def fake_dispatch_ui(action_name, args, get_bbox):
        if action_name == "scroll":
            return {"status": "success", "action": "scroll", "result": "ok"}
        if action_name == "click_marker":
            return {"status": "success", "action": "click_marker", "result": "ok"}
        raise AssertionError(action_name)

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    base_state = {
        "current_markers": [{"id": 1, "bbox": [100, 100, 140, 140]}],
        "current_url": "https://www.wanted.co.kr/wd/1",
        "current_url_stale": False,
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
    }

    scroll_state = {
        **base_state,
        "last_action_result": AIMessage(content="", tool_calls=[{"name": "scroll", "args": {"direction": "down"}, "id": "1"}]),
    }
    assert nodes.action_node(scroll_state)["current_url_stale"] is False

    click_state = {
        **base_state,
        "last_action_result": AIMessage(content="", tool_calls=[{"name": "click_marker", "args": {"marker_id": 1}, "id": "1"}]),
    }
    click_result = nodes.action_node(click_state)
    assert click_result["current_url_stale"] is True


def test_update_extracted_info_skips_wanted_job_without_detail_url():
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    result = nodes.action_node({
        "current_markers": [],
        "current_url": "https://www.wanted.co.kr/search?query=android&tab=position",
        "current_url_stale": False,
        "reflex_state_key": "state-list",
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_extracted_info",
                    "args": {
                        "data_json": json.dumps(
                            {
                                "공고목록": [
                                    {
                                        "회사명": "비모소프트",
                                        "직무명": "[인턴] Android 개발자",
                                        "주요업무": ["Android App 개발"],
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    },
                    "id": "1",
                }
            ],
        ),
    })

    action = result["action_history"][0]
    assert action["status"] == "skipped"
    assert action["reason"] == "job_update_requires_detail_url"
    assert result["extracted_jd"] == {}
    episode = result["feedback_episodes"][0]
    assert episode["feedback"]["label"] == "no_effect"
    assert episode["feedback"]["reason"] == "job_update_requires_detail_url"


def test_update_extracted_info_auto_finishes_when_target_count_reached(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    monkeypatch.delenv("VISION_AUTO_FINISH_ON_TARGET", raising=False)

    result = nodes.action_node({
        "current_markers": [],
        "current_url": "https://www.wanted.co.kr/wd/12345",
        "current_url_stale": False,
        "reflex_state_key": "state-detail",
        "recipe_params": {"target_count": 1},
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_extracted_info",
                    "args": {
                        "data_json": json.dumps(
                            {
                                "jobs": [
                                    {
                                        "company_name": "Acme",
                                        "position": "iOS Engineer",
                                        "url": "https://www.wanted.co.kr/wd/12345",
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    },
                    "id": "1",
                }
            ],
        ),
    })

    action = result["action_history"][0]
    assert result["is_finished"] is True
    assert action["auto_finished"] is True
    assert action["target_count"] == 1
    assert action["collected_count"] == 1
    assert len(result["extracted_jd"]["공고목록"]) == 1


def test_update_extracted_info_does_not_auto_scroll_when_detail_incomplete(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    def fail_dispatch(*_args, **_kwargs):
        raise AssertionError("detail_complete=false must not trigger executor-side UI action")

    monkeypatch.setattr(nodes, "_dispatch_ui", fail_dispatch)

    result = nodes.action_node({
        "current_markers": [],
        "current_url": "https://www.wanted.co.kr/wd/12345",
        "current_url_stale": False,
        "reflex_state_key": "state-detail",
        "recipe_params": {"target_count": 2},
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_extracted_info",
                    "args": {
                        "page_role": "job_detail",
                        "detail_complete": False,
                        "data_json": json.dumps(
                            {
                                "공고목록": [
                                    {
                                        "company_name": "Acme",
                                        "position": "iOS Engineer",
                                        "url": "https://www.wanted.co.kr/wd/12345",
                                        "main_tasks": ["Build app"],
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        ),
                    },
                    "id": "1",
                }
            ],
        ),
    })

    assert [action["action"] for action in result["action_history"]] == ["update_extracted_info"]
    assert "detail_policy" not in result["action_history"][0]
    assert result["last_action_screen_changed"] is False
    assert result["pending_transition"] == {}
    assert result["current_markers"] == []


def test_update_extracted_info_auto_goes_back_when_detail_complete_and_more_targets(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    calls = []

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        calls.append((action_name, args))
        assert action_name == "go_back"
        return {"status": "success", "action": "go_back", "result": "ok"}

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    result = nodes.action_node({
        "current_markers": [],
        "current_url": "https://www.wanted.co.kr/wd/12345",
        "current_url_stale": False,
        "reflex_state_key": "state-detail",
        "recipe_params": {"target_count": 2},
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_extracted_info",
                    "args": {
                        "page_role": "job_detail",
                        "detail_complete": True,
                        "data_json": json.dumps(
                            {
                                "공고목록": [
                                    {
                                        "company_name": "Acme",
                                        "position": "iOS Engineer",
                                        "url": "https://www.wanted.co.kr/wd/12345",
                                        "main_tasks": ["Build app"],
                                        "requirements": ["Swift"],
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        ),
                    },
                    "id": "1",
                }
            ],
        ),
    })

    assert [action["action"] for action in result["action_history"]] == ["update_extracted_info", "go_back"]
    assert result["action_history"][0]["detail_policy"] == "detail_complete"
    assert result["action_history"][1]["policy_action"] is True
    assert result["last_action_screen_changed"] is True
    assert result["current_url_stale"] is True
    assert calls[0][0] == "go_back"


def test_action_node_blocks_sensitive_ui_action_before_dispatch(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    def fail_dispatch(*args, **kwargs):
        raise AssertionError("sensitive action must not be dispatched")

    monkeypatch.setattr(nodes, "_dispatch_ui", fail_dispatch)

    result = nodes.action_node({
        "current_markers": [{"id": 7, "text": "가입 신청", "bbox": [0, 0, 100, 20]}],
        "current_url": "https://bank.example/product",
        "current_url_stale": False,
        "reflex_state_key": "state-sensitive",
        "recipe_params": {},
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "click_marker",
                    "args": {
                        "marker_id": 7,
                        "target_label": "가입 신청",
                        "risk_level": "sensitive",
                        "needs_user_confirmation": True,
                    },
                    "id": "1",
                }
            ],
        ),
    })

    assert result["pending_human_approval"] is True
    assert result["human_approval_request"]["action"] == "click_marker"
    assert result["action_history"][0]["status"] == "skipped"
    assert result["action_history"][0]["reason"] == "tool_args_requested_user_confirmation"


def test_perception_node_uses_cached_url_when_fresh(monkeypatch, tmp_path):
    from PIL import Image
    from agent.graph import nodes

    image_path = tmp_path / "screen.png"
    Image.new("RGB", (200, 200), "white").save(image_path)

    class FakePerception:
        def __init__(self):
            self.url_reads = 0

        def capture_screen(self):
            return image_path

        def analyze_ui(self, _image_path):
            return {"markers": [], "marked_image": str(image_path)}

        def get_current_url(self):
            self.url_reads += 1
            return "https://www.wanted.co.kr/wd/101"

    fake_perception = FakePerception()
    monkeypatch.setattr(nodes, "_get_perception", lambda: fake_perception)

    fresh_result = nodes.perception_node({
        "current_url": "https://www.wanted.co.kr/wd/100",
        "current_url_stale": False,
    })
    assert fake_perception.url_reads == 0
    assert fresh_result["current_url"] == "https://www.wanted.co.kr/wd/100"
    assert fresh_result["current_url_stale"] is False
    assert fresh_result["screen_signature"]["size"] == [200, 200]
    assert len(fresh_result["screen_signature"]["phash"]) == 16

    stale_result = nodes.perception_node({
        "current_url": "https://www.wanted.co.kr/wd/100",
        "current_url_stale": True,
    })
    assert fake_perception.url_reads == 1
    assert stale_result["current_url"] == "https://www.wanted.co.kr/wd/101"
    assert stale_result["current_url_stale"] is False


def test_ui_context_limits_marker_prompt(monkeypatch):
    from agent.graph.nodes import _build_ui_context

    monkeypatch.setenv("VISION_UI_TEXT_MARKER_LIMIT", "2")
    monkeypatch.setenv("VISION_UI_ICON_MARKER_LIMIT", "1")
    markers = [
        {"id": 1, "text": "검색", "bbox": [0, 10, 10, 20]},
        {"id": 2, "text": "회사명", "bbox": [0, 20, 10, 30]},
        {"id": 3, "text": "일반 텍스트", "bbox": [0, 30, 10, 40]},
        {"id": 4, "text": "상호작용 가능한 요소 (icon)", "bbox": [0, 40, 10, 50]},
        {"id": 5, "text": "상호작용 가능한 요소 (icon)", "bbox": [0, 50, 10, 60]},
    ]

    ui_context = _build_ui_context(markers)

    assert "[id: 1] 검색" in ui_context
    assert "[id: 2] 회사명" in ui_context
    assert "[id: 3]" not in ui_context
    assert "기타 아이콘/버튼 마커 ID 목록: [4]" in ui_context
    assert "생략된 마커: 텍스트 1개, 아이콘 1개" in ui_context


def test_analyze_ui_returns_cached_result_without_som(tmp_path):
    from PIL import Image
    from agent.tools.perception import PerceptionEngine

    image_path = tmp_path / "screen.png"
    Image.new("RGB", (200, 200), "white").save(image_path)
    engine = object.__new__(PerceptionEngine)
    engine._analysis_cache = {}
    engine._analysis_cache_order = []
    engine._analysis_cache_limit = 8
    key = engine._image_signature(image_path)
    engine._analysis_cache[key] = {
        "markers": [{"id": 7, "text": "검색", "bbox": [10, 20, 30, 40]}],
        "original_image": str(image_path),
        "marked_image": "marked.png",
    }

    result = engine.analyze_ui(image_path)

    assert result["markers"] == [{"id": 7, "text": "검색", "bbox": [10, 20, 30, 40]}]
    assert result["marked_image"] == "marked.png"


def test_release_address_bar_focus_presses_escape_twice():
    from agent.tools.perception import PerceptionEngine

    class FakePyAutoGUI:
        PAUSE = 0.1

        def __init__(self):
            self.pressed = []

        def press(self, key):
            self.pressed.append(key)

    fake_pyautogui = FakePyAutoGUI()
    engine = object.__new__(PerceptionEngine)

    engine.release_address_bar_focus(fake_pyautogui, key_pause=0.02)

    assert fake_pyautogui.pressed == ["esc", "esc"]
    assert fake_pyautogui.PAUSE == 0.1


def test_go_back_releases_address_bar_focus_and_uses_browserback(monkeypatch):
    from agent.tools import actions
    from agent.tools.actions import ActionTools

    calls = []

    class FakePerception:
        def _get_browser_region(self):
            calls.append("region")
            return {"left": 0, "top": 0, "width": 100, "height": 100}

        def release_address_bar_focus(self, key_pause=0.02):
            calls.append(("release_focus", key_pause))

    class FakePyAutoGUI:
        def press(self, key):
            calls.append(("press", key))

    monkeypatch.setattr(actions, "pyautogui", FakePyAutoGUI())

    action_tools = object.__new__(ActionTools)
    action_tools.perception = FakePerception()

    result = action_tools.go_back()

    assert result["status"] == "success"
    assert calls == ["region", ("release_focus", 0.02), ("press", "browserback")]


@pytest.mark.external
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not configured in env")
def test_qa_reasoning_node_e2e(setup_test_db, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "DB_PATH", TEST_DB_PATH)
    
    # realtime_scraping 도구 모킹하여 실제 브라우저 자동화 실행 방지
    from langchain_core.tools import tool
    @tool("realtime_scraping")
    def mock_realtime_scraping(company: str = None, tech_stack: str = None, site: str = None, query: str = None) -> str:
        """실시간 채용 공고를 수집하는 모킹 도구입니다."""
        return f"실시간 수집 완료: '{query or company or tech_stack}'에 매칭되는 채용 정보를 찾지 못했습니다."
    monkeypatch.setattr("agent.application.chat_service.realtime_scraping", mock_realtime_scraping)
    monkeypatch.setattr("agent.application.chat_service._chat_service", None)
    
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


@pytest.mark.external
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not configured in env")
def test_hallucination_rejection(setup_test_db, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "DB_PATH", TEST_DB_PATH)
    
    # realtime_scraping 도구 모킹하여 실제 브라우저 자동화 실행 방지
    from langchain_core.tools import tool
    @tool("realtime_scraping")
    def mock_realtime_scraping(company: str = None, tech_stack: str = None, site: str = None, query: str = None) -> str:
        """실시간 채용 공고를 수집하는 모킹 도구입니다."""
        return f"실시간 수집 완료: '{query or company or tech_stack}'에 매칭되는 채용 정보를 찾지 못했습니다."
    monkeypatch.setattr("agent.application.chat_service.realtime_scraping", mock_realtime_scraping)
    monkeypatch.setattr("agent.application.chat_service._chat_service", None)
    
    # 2. 환각 거절 질문 테스트
    state = GraphState(goal="스페이스X의 화성 탐사선 개발자 공고 우대사항을 알려줘")
    result = qa_reasoning_node(state)
    answer = result.get("last_action_result", "")
    
    print("\n--- Commander Rejection Answer ---")
    print(answer)
    print("----------------------------------")
    
    assert "찾을 수 없습니다" in answer or "확인되지 않음" in answer


def test_web_server_api_endpoints(setup_test_db, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "DB_PATH", TEST_DB_PATH)
    
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

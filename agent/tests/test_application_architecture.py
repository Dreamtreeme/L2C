import json
import threading
from datetime import datetime, timedelta, timezone

import pytest


def test_commander_prompt_includes_runtime_date_and_sufficiency_rules():
    from agent.prompts.commander import build_qa_commander_system_prompt

    now = datetime(
        2026,
        7,
        13,
        9,
        30,
        tzinfo=timezone(timedelta(hours=9), name="KST"),
    )
    prompt = build_qa_commander_system_prompt(now)

    assert "date=2026-07-13" in prompt
    assert "datetime=2026-07-13T09:30:00+09:00" in prompt
    assert "timezone=KST" in prompt
    assert "verified_posted_at_count" in prompt
    assert "created_at이 최근이어도 공고 게시일이 확인된 것은 아닙니다" in prompt
    assert "검색어를 넓히거나 바꾸어 realtime_scraping을 다시 호출하지 마십시오" in prompt
    assert "실시간 수집을 한 번 수행하십시오" in prompt


def test_run_context_collects_usage_steps_and_events():
    from agent.application.run_context import (
        emit_run_event,
        measure_step,
        record_external_llm_usage,
        run_context,
    )
    from agent.application.run_contracts import RunPhase

    events = []
    with run_context(
        run_id="chat-run-1",
        query="iOS 개발자 공고",
        event_sink=events.append,
    ) as (context, created):
        assert created is True
        with measure_step("test_step"):
            pass
        record_external_llm_usage(
            component="detail_extraction",
            provider="openai",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 5},
            duration_sec=0.25,
        )
        emit_run_event("test_progress", RunPhase.COLLECTION, "수집 중")
        snapshot = context.snapshot()

    assert snapshot["run_id"] == "chat-run-1"
    assert snapshot["llm"]["totals"]["input_tokens"] == 10
    assert snapshot["llm"]["totals"]["output_tokens"] == 5
    assert snapshot["llm"]["cost"]["estimated_total"] is None
    assert snapshot["steps"][0]["component"] == "test_step"
    assert [event.event for event in events] == ["run_started", "test_progress"]


def test_llm_cost_uses_external_exact_model_pricing(tmp_path):
    from agent.application.llm_cost import estimate_llm_cost

    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps(
            {
                "models": {
                    "test-model": {
                        "input_usd_per_million": 1.0,
                        "cached_input_usd_per_million": 0.2,
                        "output_usd_per_million": 4.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = estimate_llm_cost(
        {
            "test-model": {
                "input_tokens": 1_000_000,
                "output_tokens": 500_000,
                "input_token_details": {"cache_read": 250_000},
            }
        },
        pricing_path=pricing_path,
    )

    assert result["estimated_total"] == 2.8
    assert result["unpriced_models"] == []


def test_chat_service_returns_run_contract_and_progress_events():
    from langchain_core.messages import AIMessage

    from agent.application.chat_service import ChatService

    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(content="DB 근거 답변", tool_calls=[])

    events = []
    result = ChatService(llm_with_tools=FakeLLM()).run(
        "질문",
        run_id="chat-contract-1",
        event_sink=events.append,
    )

    assert result["run_id"] == "chat-contract-1"
    assert result["run_status"] == "completed"
    assert result["last_action_result"] == "DB 근거 답변"
    assert result["metrics"]["run_id"] == "chat-contract-1"
    assert result["duration_sec"] >= 0
    assert events[-1].event == "run_completed"


def test_chat_service_returns_structured_clarification_without_collection():
    from langchain_core.messages import AIMessage

    from agent.application.chat_service import ChatService

    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "request_clarification",
                        "args": {
                            "question": "AI 직무 중 개발과 기획 중 어느 쪽을 찾을까요?",
                            "missing_fields": ["직무 범위"],
                            "reason": "검색 결과가 크게 달라짐",
                        },
                        "id": "clarification-1",
                        "type": "tool_call",
                    }
                ],
            )

    events = []
    result = ChatService(llm_with_tools=FakeLLM()).run(
        "AI 쪽 채용공고 찾아줘",
        run_id="chat-clarification-1",
        event_sink=events.append,
    )

    assert result["run_status"] == "waiting_input"
    assert result["is_finished"] is False
    assert result["clarification"]["missing_fields"] == ["직무 범위"]
    assert result["last_action_result"] == "AI 직무 중 개발과 기획 중 어느 쪽을 찾을까요?"
    assert events[-1].event == "clarification_required"
    assert events[-1].status.value == "waiting_input"


def test_chat_service_prioritizes_clarification_over_accidental_collection_call():
    from langchain_core.messages import AIMessage

    from agent.application.chat_service import ChatService

    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "realtime_scraping",
                        "args": {"query": "AI"},
                        "id": "collection-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "request_clarification",
                        "args": {"question": "어느 AI 직무를 찾을까요?"},
                        "id": "clarification-1",
                        "type": "tool_call",
                    },
                ],
            )

    class FailingCollectionTool:
        def invoke(self, args):
            raise AssertionError("확인 질문 전에 수집 도구가 실행됨")

    service = ChatService(llm_with_tools=FakeLLM())
    service._tools["realtime_scraping"] = FailingCollectionTool()

    result = service.run("AI 공고", run_id="clarification-priority")

    assert result["run_status"] == "waiting_input"
    assert result["last_action_result"] == "어느 AI 직무를 찾을까요?"


def test_run_registry_tracks_progress_and_completion():
    from agent.application.run_contracts import RunEvent, RunPhase, RunStatus
    from agent.application.run_registry import RunRegistry

    registry = RunRegistry(limit=10)
    registry.start("run-1", "질문")
    registry.apply_event(
        RunEvent(
            run_id="run-1",
            event="collection_started",
            phase=RunPhase.COLLECTION,
            status=RunStatus.RUNNING,
            message="수집 중",
        )
    )
    registry.complete("run-1", {"last_action_result": "완료"})

    item = registry.get("run-1")
    assert item is not None
    assert item["status"] == "completed"
    assert item["result"]["last_action_result"] == "완료"


def test_run_registry_preserves_waiting_input_status():
    from agent.application.run_contracts import RunEvent, RunPhase, RunStatus
    from agent.application.run_registry import RunRegistry

    registry = RunRegistry(limit=10)
    registry.start("run-waiting", "모호한 질문")
    registry.apply_event(
        RunEvent(
            run_id="run-waiting",
            event="clarification_required",
            phase=RunPhase.CLARIFICATION,
            status=RunStatus.WAITING_INPUT,
            message="어느 직무인가요?",
        )
    )
    registry.complete(
        "run-waiting",
        {
            "run_status": "waiting_input",
            "clarification": {"question": "어느 직무인가요?"},
        },
    )

    item = registry.get("run-waiting")
    assert item["status"] == "waiting_input"
    assert item["phase"] == "clarification"


def test_run_registry_tracks_cancellation_and_conversation_history():
    from agent.application.run_registry import RunRegistry

    registry = RunRegistry(limit=10)
    registry.start(
        "conversation-run-1",
        "첫 질문",
        conversation_id="conversation-1",
        user_query="첫 질문",
    )
    registry.complete(
        "conversation-run-1",
        {"run_status": "completed", "last_action_result": "첫 답변"},
    )
    registry.start("conversation-run-2", "두 번째 질문", conversation_id="conversation-1")

    cancelled = registry.request_cancel("conversation-run-2")
    history = registry.conversation_history("conversation-1")

    assert cancelled["cancel_requested"] is True
    assert registry.is_cancel_requested("conversation-run-2") is True
    assert [item["run_id"] for item in history] == ["conversation-run-1"]


def test_chat_service_stops_before_llm_when_cancel_is_requested():
    from agent.application.chat_service import ChatService
    from agent.application.run_registry import get_run_registry

    class FailingLLM:
        def invoke(self, messages):
            raise AssertionError("취소된 실행이 LLM을 호출함")

    registry = get_run_registry()
    registry.start("cancel-before-llm", "취소할 질문")
    registry.request_cancel("cancel-before-llm")

    result = ChatService(llm_with_tools=FailingLLM()).run(
        "취소할 질문",
        run_id="cancel-before-llm",
    )

    assert result["run_status"] == "cancelled"
    assert result["is_finished"] is False
    assert result["last_action_result"] == "실행을 취소했습니다."


def test_chat_api_streams_structured_progress_without_character_delay(monkeypatch):
    from fastapi.testclient import TestClient

    from agent.application.run_contracts import RunEvent, RunPhase
    from agent.web_server import app

    class FakeChatService:
        def run(self, query, *, run_id=None, event_sink=None, **kwargs):
            assert query == "테스트 질문"
            assert run_id
            event_sink(
                RunEvent(
                    run_id=run_id,
                    event="collection_started",
                    phase=RunPhase.COLLECTION,
                    message="수집 중",
                )
            )
            return {
                "run_id": run_id,
                "run_status": "completed",
                "last_action_result": "최종 답변",
                "metrics": {"duration_sec": 0.01},
            }

    monkeypatch.setattr(
        "agent.web_server.get_chat_service",
        lambda: FakeChatService(),
    )
    response = TestClient(app).post("/api/chat", json={"query": "테스트 질문"})

    assert response.status_code == 200
    assert "[PROCESSING]" in response.text
    assert "[EVENT]" in response.text
    assert '"text": "최종 답변"' in response.text
    assert "data: 최" not in response.text


def test_chat_api_resumes_from_structured_clarification(monkeypatch):
    from fastapi.testclient import TestClient

    from agent.application.run_contracts import RunEvent, RunPhase, RunStatus
    from agent.application.run_registry import get_run_registry
    from agent.web_server import app

    registry = get_run_registry()
    registry.start("previous-clarification", "AI 쪽 채용공고 찾아줘")
    registry.apply_event(
        RunEvent(
            run_id="previous-clarification",
            event="clarification_required",
            phase=RunPhase.CLARIFICATION,
            status=RunStatus.WAITING_INPUT,
            message="개발과 기획 중 어느 쪽인가요?",
        )
    )
    registry.complete(
        "previous-clarification",
        {
            "run_status": "waiting_input",
            "investigation_id": "investigation-clarification",
            "clarification": {
                "question_id": "job_scope",
                "question": "개발과 기획 중 어느 쪽인가요?",
            },
        },
    )

    class FakeChatService:
        def run(self, query, *, run_id=None, event_sink=None, **kwargs):
            assert query == "개발"
            assert kwargs["investigation_id"] == "investigation-clarification"
            assert kwargs["clarification_answer"] == {
                "question_id": "job_scope",
                "selected_option_id": "",
                "value": "",
                "custom_value": "개발",
            }
            return {
                "run_id": run_id,
                "run_status": "completed",
                "last_action_result": "개발 공고를 찾았습니다.",
                "investigation_id": "investigation-clarification",
                "metrics": {},
            }

    monkeypatch.setattr("agent.web_server.get_chat_service", lambda: FakeChatService())
    response = TestClient(app).post(
        "/api/chat",
        json={"query": "개발", "resume_run_id": "previous-clarification"},
    )

    assert response.status_code == 200
    assert "개발 공고를 찾았습니다." in response.text
    assert '"resumed_from_run_id": "previous-clarification"' in response.text


def test_chat_api_uses_recent_conversation_context(monkeypatch):
    from fastapi.testclient import TestClient

    from agent.application.run_registry import get_run_registry
    from agent.web_server import app

    registry = get_run_registry()
    registry.start(
        "conversation-context-1",
        "iOS 공고 두 개 찾아줘",
        conversation_id="conversation-context",
        user_query="iOS 공고 두 개 찾아줘",
    )
    registry.complete(
        "conversation-context-1",
        {"run_status": "completed", "last_action_result": "두 건을 찾았습니다."},
    )

    class FakeChatService:
        def run(self, query, *, run_id=None, event_sink=None, **kwargs):
            assert "[최근 대화 문맥]" in query
            assert "사용자: iOS 공고 두 개 찾아줘" in query
            assert "도우미: 두 건을 찾았습니다." in query
            assert "[현재 사용자 요청]\n그중 경력 조건만 비교해줘" in query
            return {
                "run_id": run_id,
                "run_status": "completed",
                "last_action_result": "비교했습니다.",
                "metrics": {},
            }

    monkeypatch.setattr("agent.web_server.get_chat_service", lambda: FakeChatService())
    response = TestClient(app).post(
        "/api/chat",
        json={
            "query": "그중 경력 조건만 비교해줘",
            "conversation_id": "conversation-context",
        },
    )

    assert response.status_code == 200
    assert "비교했습니다." in response.text


def test_cancel_run_api_marks_active_run():
    from fastapi.testclient import TestClient

    from agent.application.run_registry import get_run_registry
    from agent.web_server import app

    registry = get_run_registry()
    registry.start("cancel-api-run", "중단할 요청")

    response = TestClient(app).post("/api/runs/cancel-api-run/cancel")

    assert response.status_code == 200
    assert response.json()["cancel_requested"] is True
    assert registry.is_cancel_requested("cancel-api-run") is True


def test_cancelled_run_resume_restarts_from_original_request():
    from agent.application.run_registry import get_run_registry
    from agent.web_server import _effective_chat_query

    registry = get_run_registry()
    registry.start(
        "cancelled-resume-run",
        "원티드 iOS 공고 두 개",
        user_query="원티드 iOS 공고 두 개",
    )
    registry.complete(
        "cancelled-resume-run",
        {"run_status": "cancelled", "last_action_result": "실행을 취소했습니다."},
    )

    query = _effective_chat_query(
        "다시 계속해줘",
        resume_run_id="cancelled-resume-run",
        conversation_id="",
    )

    assert "[취소된 사용자 요청]\n원티드 iOS 공고 두 개" in query
    assert "오래된 화면 좌표는 재사용하지 말고" in query
    assert "[사용자의 재개 지시]\n다시 계속해줘" in query


def test_worker_state_factory_returns_independent_mutable_values():
    from agent.graph.state_factory import create_worker_state

    first = create_worker_state("first", recipe_params={"site": "wanted"})
    second = create_worker_state("second")

    first["action_history"].append({"action": "click_marker"})
    first["recipe_params"]["query"] = "ios"

    assert first["goal"] == "first"
    assert second["goal"] == "second"
    assert second["action_history"] == []
    assert second["recipe_params"] == {}


def test_action_request_preserves_executor_tool_call_contract():
    from agent.graph.action_request import ActionRequest, ToolCallRequest

    request = ActionRequest(
        source="card_queue",
        summary="next card",
        tool_calls=[
            ToolCallRequest(
                name="click_marker",
                args={"marker_id": 7, "target_label": "iOS 개발자"},
                id="queue-1",
            )
        ],
    )

    message = request.to_ai_message()

    assert message.content == "[card_queue] next card"
    assert message.tool_calls == [
        {
            "name": "click_marker",
            "args": {"marker_id": 7, "target_label": "iOS 개발자"},
            "id": "queue-1",
            "type": "tool_call",
        }
    ]


def test_worker_execution_service_delegates_prepare_and_stream(monkeypatch):
    from agent.application.worker_execution_service import execute_worker_graph

    calls = []
    fake_app = object()

    monkeypatch.setattr("agent.graph.workflow.build_graph", lambda: fake_app)

    def prepare(initial_state, site_profile):
        calls.append(("prepare", initial_state["goal"], site_profile["entry"]["slug"]))
        return {**initial_state, "prepared": True}

    def run(app, prepared_state, recursion_limit):
        calls.append(("run", app is fake_app, prepared_state["prepared"], recursion_limit))
        return {**prepared_state, "is_finished": True}, False

    final_state, hit_limit = execute_worker_graph(
        {"goal": "collect"},
        {"entry": {"slug": "wanted"}},
        60,
        prepare_screen=prepare,
        run_graph=run,
    )

    assert calls == [
        ("prepare", "collect", "wanted"),
        ("run", True, True, 60),
    ]
    assert final_state["is_finished"] is True
    assert hit_limit is False


def test_collection_service_forces_partial_status_when_explicit_target_is_unmet():
    from agent.application.collection_service import (
        CollectionOperations,
        CollectionRequest,
        CollectionService,
    )

    def run_worker(*args, **kwargs):
        return {
            "submission": {
                "run_id": "worker-partial",
                "collected_count": 1,
                "target_count": 2,
                "task_category": "검색",
            },
            "site_name": "Wanted",
            "site_slug": "wanted",
            "keyword": "iOS 개발자",
            "target_count": 2,
            "is_finished": True,
            "hit_recursion_limit": False,
        }

    def persist(worker_result, review):
        worker_result["persistence_validation"] = {
            "submitted_count": 1,
            "persisted_count": 1,
            "rejected_count": 0,
            "rejected_items": [],
        }
        return 1, worker_result["submission"], review, "submission-partial"

    operations = CollectionOperations(
        normalize_target_count=lambda value: int(value or 0),
        normalize_task_category=lambda value: value or "검색",
        review_retries=lambda: 0,
        run_worker=run_worker,
        review_worker=lambda submission: ({"decision": "accept"}, "submission-partial"),
        persist_result=persist,
        render_review_feedback=lambda review: "",
        needs_approval=lambda **kwargs: False,
        build_intermediate_report=lambda *args, **kwargs: {},
        report_requires_more_collection=lambda report: False,
        close_browser=lambda: None,
    )

    result = CollectionService(operations).collect(
        CollectionRequest(
            search_keyword="iOS 개발자",
            site="wanted",
            target_count=2,
            search_intent_resolved=True,
            collection_intent={
                "search_keyword": "iOS 개발자",
                "count_mode": "explicit",
                "target_count": 2,
            },
        )
    )

    assert result["completion_status"] == "partial"
    assert result["missing_count"] == 1
    assert result["persisted_count"] == 1
    assert "partial collection persisted" in result["message"]


def test_worker_execution_session_serializes_concurrent_requests():
    from agent.application.worker_execution_service import worker_execution_session

    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first_worker():
        with worker_execution_session():
            first_entered.set()
            release_first.wait(timeout=2)

    def second_worker():
        with worker_execution_session():
            second_entered.set()

    first_thread = threading.Thread(target=first_worker)
    second_thread = threading.Thread(target=second_worker)
    first_thread.start()
    assert first_entered.wait(timeout=2)
    second_thread.start()

    assert second_entered.wait(timeout=0.1) is False
    release_first.set()
    assert second_entered.wait(timeout=2)

    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert first_thread.is_alive() is False
    assert second_thread.is_alive() is False


def test_worker_startup_overlaps_ocr_and_browser_before_perception(monkeypatch):
    from agent.application.worker_execution_service import prepare_worker_start_screen
    from agent.graph import nodes

    ocr_started = threading.Event()
    browser_opened = threading.Event()
    order = []

    class FakeSomEngine:
        def ensure_ocr_worker_ready(self):
            order.append("ocr_started")
            ocr_started.set()
            assert browser_opened.wait(timeout=2)
            order.append("ocr_ready")

    class FakePerception:
        som_engine = FakeSomEngine()

    class FakeActionTools:
        perception = FakePerception()

        def open_browser(self, url="", current_url="", site=""):
            assert ocr_started.wait(timeout=2)
            order.append("browser_opened")
            browser_opened.set()
            return {"status": "success", "result": {"url": "https://www.wanted.co.kr"}}

    def fake_perception_node(_state, *, max_capture_attempts=None):
        assert order[-1] == "ocr_ready"
        assert max_capture_attempts == 1
        order.append("perception")
        return {
            "current_url": "https://www.wanted.co.kr",
            "current_url_stale": False,
            "current_markers": [{"id": 1}],
            "recent_images": ["screen.png"],
            "low_information_screen": False,
        }

    monkeypatch.setenv("VISION_WORKER_PREOPEN_BROWSER", "1")
    monkeypatch.setenv("VISION_WORKER_START_OPEN_ATTEMPTS", "1")
    monkeypatch.setattr(nodes, "_get_action_tools", lambda: FakeActionTools())
    monkeypatch.setattr(nodes, "perception_node", fake_perception_node)

    result = prepare_worker_start_screen(
        {"current_url": "", "action_history": []},
        {"entry": {"slug": "wanted"}},
    )

    assert order == ["ocr_started", "browser_opened", "ocr_ready", "perception"]
    assert result["current_url"] == "https://www.wanted.co.kr"


def test_worker_graph_does_not_start_before_failed_ocr_readiness(monkeypatch):
    from agent.application.worker_execution_service import (
        OcrWorkerReadinessError,
        prepare_worker_start_screen,
    )
    from agent.graph import nodes

    perception_called = []

    class FakeSomEngine:
        def ensure_ocr_worker_ready(self):
            raise RuntimeError("startup failed")

    class FakePerception:
        som_engine = FakeSomEngine()

    class FakeActionTools:
        perception = FakePerception()

        def open_browser(self, url="", current_url="", site=""):
            return {"status": "success", "result": {"url": "https://www.wanted.co.kr"}}

    monkeypatch.setenv("VISION_WORKER_PREOPEN_BROWSER", "1")
    monkeypatch.setenv("VISION_WORKER_START_OPEN_ATTEMPTS", "1")
    monkeypatch.setattr(nodes, "_get_action_tools", lambda: FakeActionTools())
    monkeypatch.setattr(
        nodes,
        "perception_node",
        lambda _state, **_kwargs: perception_called.append(True) or {},
    )

    with pytest.raises(OcrWorkerReadinessError):
        prepare_worker_start_screen(
            {"current_url": "", "action_history": []},
            {"entry": {"slug": "wanted"}},
        )

    assert perception_called == []


def test_worker_startup_reopens_after_one_blank_observation(monkeypatch):
    from agent.application.worker_execution_service import prepare_worker_start_screen
    from agent.graph import nodes

    browser_calls = []
    capture_limits = []

    class FakeSomEngine:
        def ensure_ocr_worker_ready(self):
            return None

    class FakePerception:
        som_engine = FakeSomEngine()

    class FakeActionTools:
        perception = FakePerception()

        def open_browser(self, url="", current_url="", site=""):
            browser_calls.append({"current_url": current_url, "site": site})
            return {"status": "success", "result": {"url": "https://www.wanted.co.kr"}}

    def fake_perception_node(_state, *, max_capture_attempts=None):
        capture_limits.append(max_capture_attempts)
        if len(capture_limits) == 1:
            return {
                "current_url": "https://www.wanted.co.kr",
                "current_url_stale": True,
                "low_information_screen": True,
                "low_information_retry_count": 1,
            }
        return {
            "current_url": "https://www.wanted.co.kr",
            "current_url_stale": False,
            "current_markers": [{"id": 1}],
            "recent_images": ["screen.png"],
            "low_information_screen": False,
        }

    monkeypatch.setenv("VISION_WORKER_PREOPEN_BROWSER", "1")
    monkeypatch.setenv("VISION_WORKER_START_OPEN_ATTEMPTS", "2")
    monkeypatch.setattr(nodes, "_get_action_tools", lambda: FakeActionTools())
    monkeypatch.setattr(nodes, "perception_node", fake_perception_node)

    result = prepare_worker_start_screen(
        {"current_url": "", "action_history": []},
        {"entry": {"slug": "wanted"}},
    )

    assert len(browser_calls) == 2
    assert capture_limits == [1, None]
    assert result["low_information_screen"] is False


def test_google_model_clients_are_reused_by_configuration(monkeypatch):
    from agent.application import model_clients
    import langchain_google_genai

    created = []

    class FakeClient:
        def __init__(self, model, temperature):
            self.model = model
            self.temperature = temperature
            created.append((model, temperature))

        def with_structured_output(self, schema):
            return (self, schema)

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeClient)
    model_clients.clear_model_client_cache()
    first = model_clients.get_google_chat_model("model-a", temperature=0.1)
    second = model_clients.get_google_chat_model("model-a", temperature=0.1)
    structured_first = model_clients.get_structured_google_model("model-a", dict, temperature=0.1)
    structured_second = model_clients.get_structured_google_model("model-a", dict, temperature=0.1)

    assert first is second
    assert structured_first is structured_second
    assert structured_first[0] is first
    assert created == [("model-a", 0.1)]
    model_clients.clear_model_client_cache()


def test_browser_is_kept_alive_by_default(monkeypatch):
    from agent.application import worker_execution_service
    from agent.graph import nodes

    closed = []

    class FakeActionTools:
        def close_browser(self):
            closed.append(True)
            return {"status": "success"}

    monkeypatch.delenv("VISION_CLOSE_BROWSER_AFTER_RUN", raising=False)
    monkeypatch.setattr(nodes, "_action_tools", FakeActionTools())

    worker_execution_service.close_browser_after_run()

    assert closed == []


def test_browser_closes_when_explicitly_enabled(monkeypatch):
    from agent.application import worker_execution_service
    from agent.graph import nodes

    closed = []

    class FakeActionTools:
        def close_browser(self):
            closed.append(True)
            return {"status": "success"}

    monkeypatch.setenv("VISION_CLOSE_BROWSER_AFTER_RUN", "1")
    monkeypatch.setattr(nodes, "_action_tools", FakeActionTools())

    worker_execution_service.close_browser_after_run()

    assert closed == [True]


def test_detail_extraction_prompt_prioritizes_page_text_over_ocr_hints():
    from agent.prompts.detail_extraction import build_detail_extraction_system_prompt

    prompt = build_detail_extraction_system_prompt("채용공고를 정리하십시오.")

    assert "서로 인접한 페이지 텍스트를 가장 우선" in prompt
    assert "이미지 내부 로고 OCR" in prompt
    assert "보조 근거로만 사용" in prompt
    assert "active_result_card" not in prompt
    assert "회사명은 로고" not in prompt
    assert "evidence_hash는 출력하지 마십시오" in prompt


def test_detail_extraction_request_keeps_card_metadata_as_fallback_only(monkeypatch):
    from agent.application import detail_extraction_service

    captured = {}

    class FakeExtractionLLM:
        def invoke(self, messages):
            captured["messages"] = messages
            return {
                "company_name": "상세 페이지 회사",
                "position": "상세 페이지 직무",
                "url": "https://example.com/jobs/1",
            }

    monkeypatch.setattr(
        detail_extraction_service,
        "get_detail_extraction_llm",
        lambda: FakeExtractionLLM(),
    )

    result = detail_extraction_service.extract_job_from_detail_ocr_buffer(
        {
            "detail_ocr_buffer": {
                "url": "https://example.com/jobs/1",
                "lines": [{"text": "상세 페이지 회사 상세 페이지 직무"}],
            },
            "active_result_card": {
                "company": "잘못 인식한 카드 회사",
                "title": "잘못 인식한 카드 직무",
            },
        },
        "https://example.com/jobs/1",
    )

    request_payload = json.loads(captured["messages"][1].content)
    assert "active_result_card" not in request_payload
    assert result["company_name"] == "상세 페이지 회사"
    assert result["position"] == "상세 페이지 직무"

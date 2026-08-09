"""FastAPI 전송 경계와 장기 실행 자원의 수명주기를 검증한다."""

import threading
from contextlib import contextmanager

import pytest


def test_citation_validation_normalizes_grouped_ids_before_validation():
    from agent.graph.investigation_answer_nodes import validate_citations

    answer = "공통 기술입니다 [job_id:64, 85, 999]."

    assert validate_citations(answer, [64, 85]) == (
        "공통 기술입니다 [job_id:64] [job_id:85] [출처 확인 불가]."
    )


def test_chat_service_returns_run_contract_and_progress_events():
    from agent.application.chat_service import ChatService
    from agent.observability.run_context import emit_run_event
    from agent.observability.run_contracts import RunPhase, RunStatus

    class FakeWorkflow:
        def run(self, query, **kwargs):
            emit_run_event(
                "run_completed",
                RunPhase.COMPLETED,
                "답변을 완료했습니다.",
                status=RunStatus.COMPLETED,
            )
            return {
                "investigation": {"investigation_id": "investigation-contract"},
                "run_status": "completed",
                "final_answer": "DB 근거 답변",
                "valid_ids": [],
            }

    events = []
    result = ChatService(investigation_workflow=FakeWorkflow()).run(
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


def test_run_registry_tracks_cancellation_and_conversation_history():
    from agent.observability.run_registry import RunRegistry

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
    registry.start(
        "conversation-run-2", "두 번째 질문", conversation_id="conversation-1"
    )

    cancelled = registry.request_cancel("conversation-run-2")
    history = registry.conversation_history("conversation-1")

    assert cancelled["cancel_requested"] is True
    assert registry.is_cancel_requested("conversation-run-2") is True
    assert [item["run_id"] for item in history] == ["conversation-run-1"]


def test_chat_service_stops_before_llm_when_cancel_is_requested():
    from agent.application.chat_service import ChatService
    from agent.observability.run_registry import get_run_registry

    class FailingWorkflow:
        def run(self, query, **kwargs):
            raise AssertionError("취소된 실행이 조사 그래프를 호출함")

    registry = get_run_registry()
    registry.start("cancel-before-llm", "취소할 질문")
    registry.request_cancel("cancel-before-llm")

    result = ChatService(investigation_workflow=FailingWorkflow()).run(
        "취소할 질문",
        run_id="cancel-before-llm",
    )

    assert result["run_status"] == "cancelled"
    assert result["is_finished"] is False
    assert result["last_action_result"] == "실행을 취소했습니다."


def test_chat_api_streams_structured_progress_without_character_delay(monkeypatch):
    from fastapi.testclient import TestClient

    from agent.observability.run_contracts import RunEvent, RunPhase
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
        "agent.web_server._chat_service_for_app",
        lambda _application: FakeChatService(),
    )
    response = TestClient(app).post("/api/chat", json={"query": "테스트 질문"})

    assert response.status_code == 200
    assert "[PROCESSING]" in response.text
    assert "[EVENT]" in response.text
    assert '"text": "최종 답변"' in response.text
    assert "data: 최" not in response.text


def test_chat_api_resumes_from_structured_clarification(monkeypatch):
    from fastapi.testclient import TestClient

    from agent.observability.run_contracts import RunEvent, RunPhase, RunStatus
    from agent.observability.run_registry import get_run_registry
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
            assert kwargs["resume_run_id"] == "previous-clarification"
            assert kwargs["investigation_id"] == ""
            assert kwargs["clarification_answer"] is None
            return {
                "run_id": run_id,
                "run_status": "completed",
                "last_action_result": "개발 공고를 찾았습니다.",
                "investigation_id": "investigation-clarification",
                "resume_mode": "checkpoint_resume",
                "metrics": {},
            }

    monkeypatch.setattr(
        "agent.web_server._chat_service_for_app",
        lambda _application: FakeChatService(),
    )
    response = TestClient(app).post(
        "/api/chat",
        json={"query": "개발", "resume_run_id": "previous-clarification"},
    )

    assert response.status_code == 200
    assert "개발 공고를 찾았습니다." in response.text
    assert '"resumed_from_run_id": "previous-clarification"' in response.text
    assert '"resume_mode": "checkpoint_resume"' in response.text


def test_chat_api_passes_conversation_id_without_rewriting_query(monkeypatch):
    from fastapi.testclient import TestClient

    from agent.observability.run_registry import get_run_registry
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
            assert query == "그중 경력 조건만 비교해줘"
            assert kwargs["conversation_id"] == "conversation-context"
            return {
                "run_id": run_id,
                "run_status": "completed",
                "last_action_result": "비교했습니다.",
                "metrics": {},
            }

    monkeypatch.setattr(
        "agent.web_server._chat_service_for_app",
        lambda _application: FakeChatService(),
    )
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

    from agent.observability.run_registry import get_run_registry
    from agent.web_server import app

    registry = get_run_registry()
    registry.start("cancel-api-run", "중단할 요청")

    response = TestClient(app).post("/api/runs/cancel-api-run/cancel")

    assert response.status_code == 200
    assert response.json()["cancel_requested"] is True
    assert registry.is_cancel_requested("cancel-api-run") is True


def test_cancelled_run_is_loaded_as_structured_graph_context():
    from agent.application.conversation_context_service import (
        load_conversation_context,
    )
    from agent.observability.run_registry import get_run_registry

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

    context = load_conversation_context(
        "",
        "cancelled-resume-run",
        registry=registry,
    )

    assert len(context) == 1
    assert context[0].user_query == "원티드 iOS 공고 두 개"
    assert context[0].run_status == "cancelled"


# 브라우저·OCR·후처리 작업자의 프로세스 수명주기
def test_worker_execution_session_serializes_concurrent_requests():
    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime

    runtime = VisionWorkerRuntime()

    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first_worker():
        with runtime.execution_session():
            first_entered.set()
            release_first.wait(timeout=2)

    def second_worker():
        with runtime.execution_session():
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


def test_worker_graph_does_not_start_before_failed_ocr_readiness():
    from agent.application.worker_execution_service import (
        OcrWorkerReadinessError,
        prepare_worker_start_screen,
    )
    from agent.sites import load_site_profile
    from agent.runtime.worker_contracts import create_worker_state

    class FakeActionTools:
        def open_browser(self, url="", current_url="", site=""):
            return {"status": "success", "result": {"url": "https://www.wanted.co.kr"}}

    class FakeRuntime:
        def get_action_tools(self):
            return FakeActionTools()

        def ensure_ocr_worker_ready(self):
            raise RuntimeError("startup failed")

        def prepare_reasoning_models(self, _tool_schemas):
            return None

    with pytest.raises(OcrWorkerReadinessError):
        prepare_worker_start_screen(
            create_worker_state(),
            load_site_profile("wanted"),
            worker_runtime=FakeRuntime(),
        )


def test_web_lifespan_manages_recipe_promotion_worker(monkeypatch):
    from fastapi.testclient import TestClient

    from agent.web_server import app

    calls = []

    class FakeRuntime:
        def __init__(self, db_path):
            calls.append(("init", db_path))
            self.chat_service = object()

        def start(self):
            calls.append("start")

        def close(self, *, promotion_timeout_sec=1.0):
            calls.append(("close", promotion_timeout_sec))

    monkeypatch.setattr("agent.web_server.ApplicationRuntime", FakeRuntime)

    with TestClient(app):
        pass

    assert calls[0][0] == "init"
    assert calls[1:] == ["start", ("close", 0.5)]


def test_browser_closes_by_default(monkeypatch):
    from agent.application.worker_execution_service import (
        WorkerExecutionService,
    )
    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime

    closed = []

    class FakeActionTools:
        def __init__(self, _perception):
            pass

        def close_browser(self):
            closed.append(True)
            return {"status": "success"}

    runtime = VisionWorkerRuntime(
        perception_factory=object,
        action_tools_factory=FakeActionTools,
    )
    runtime.get_action_tools()

    service = WorkerExecutionService(
        runtime,
        lambda *, worker_runtime: {"runtime": worker_runtime},
    )
    result = service.run()

    assert closed == [True]
    assert result["runtime"] is runtime


def test_browser_cleanup_reports_returned_failure():
    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime

    class FakeActionTools:
        def __init__(self, _perception):
            pass

        def close_browser(self):
            return {
                "status": "success",
                "result": {
                    "closed": False,
                    "reason": "browser_not_found",
                },
            }

    runtime = VisionWorkerRuntime(
        perception_factory=object,
        action_tools_factory=FakeActionTools,
    )
    runtime.get_action_tools()

    assert runtime.close_browser_after_run() is False


def test_worker_execution_service_closes_browser_after_worker_failure():
    from agent.application.worker_execution_service import (
        WorkerExecutionService,
    )

    events = []

    class FakeRuntime:
        @contextmanager
        def execution_session(self):
            events.append("lock_entered")
            try:
                yield
            finally:
                events.append("lock_released")

        def close_browser_after_run(self):
            events.append("browser_closed")

    def fail_worker(*args, **kwargs):
        events.append("worker_started")
        raise RuntimeError("worker failed")

    service = WorkerExecutionService(FakeRuntime(), fail_worker)

    with pytest.raises(RuntimeError, match="worker failed"):
        service.run()

    assert events == [
        "lock_entered",
        "worker_started",
        "browser_closed",
        "lock_released",
    ]


def test_vision_runtime_reuses_ocr_worker_until_application_shutdown(monkeypatch):
    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime

    events = []

    class FakePerception:
        browser_window_id = None
        ocr_worker_pid = 7007

        def close(self):
            events.append("perception_closed")
            events.append("ocr_closed")

    class FakeActionTools:
        def __init__(self, perception):
            self.perception = perception

        def close_browser(self):
            events.append("browser_closed")
            return {"status": "success"}

    runtime = VisionWorkerRuntime(
        perception_factory=FakePerception,
        action_tools_factory=FakeActionTools,
    )

    with runtime.execution_session():
        first_perception = runtime.get_perception()
        first_actions = runtime.get_action_tools()
        assert runtime.ocr_worker_pid == 7007
        assert runtime.resource_snapshot() == {
            "closed": False,
            "initialized": True,
            "ocr_worker_pid": 7007,
            "browser_window_bound": False,
            "ui_model_variant_count": 0,
            "graph_initialized": False,
        }

    runtime.close_browser_after_run()

    with runtime.execution_session():
        assert runtime.get_perception() is first_perception
        assert runtime.get_action_tools() is first_actions
        assert runtime.ocr_worker_pid == 7007

    assert events == ["browser_closed"]

    runtime.close()

    assert events == ["browser_closed", "perception_closed", "ocr_closed"]


def test_application_runtime_keeps_vision_lazy_until_collection(monkeypatch, tmp_path):
    from agent.bootstrap import ApplicationRuntime

    monkeypatch.setenv("VISION_RECIPE_AUTO_PROMOTE", "0")
    runtime = ApplicationRuntime(tmp_path / "runtime.db")

    assert runtime.vision_runtime.is_initialized is False

    runtime.start()

    assert runtime.vision_runtime.is_initialized is False

    runtime.close()

"""FastAPI 전송 경계와 장기 실행 자원의 수명주기를 검증한다."""

import asyncio
import threading

import pytest


def _execution_result(run_id, *, status="completed", answer=""):
    from agent.observability.run_contracts import ChatResult
    from shared.schema.run_schema import RunStatus

    return ChatResult(
        run_id=run_id,
        status=RunStatus(status),
        text=answer,
    )


def test_answer_keeps_valid_field_evidence_and_multiple_documents():
    from agent.graph.investigation_answer_nodes import (
        render_grounded_answer,
        validate_grounded_answer,
    )
    from shared.schema.jd_schema import JobField
    from shared.schema.investigation_schema import (
        AnswerEvidencePointer,
        GroundedAnswerDraft,
        GroundedAnswerDraftLine,
    )

    answer = GroundedAnswerDraft(
        lines=[
            GroundedAnswerDraftLine(
                kind="overview",
                text="LangGraph 워크플로를 개발합니다.",
                evidence=[
                    AnswerEvidencePointer(
                        document_id=64,
                        field=JobField.MAIN_TASKS,
                        item_index=0,
                    )
                ],
            ),
            GroundedAnswerDraftLine(
                kind="overview",
                text="존재하지 않는 항목입니다.",
                evidence=[
                    AnswerEvidencePointer(
                        document_id=64,
                        field=JobField.MAIN_TASKS,
                        item_index=9,
                    )
                ],
            ),
            GroundedAnswerDraftLine(
                kind="overview",
                text="두 공고 모두 에이전트 워크플로 경험을 요구합니다.",
                evidence=[
                    AnswerEvidencePointer(
                        document_id=64,
                        field=JobField.MAIN_TASKS,
                        item_index=0,
                    ),
                    AnswerEvidencePointer(
                        document_id=65,
                        field=JobField.TECH_STACK,
                        item_index=0,
                    ),
                ],
            ),
        ]
    )
    documents = [
        {
            "id": 64,
            "company_name": "예시회사",
            "position": "AI 에이전트 개발자",
            "main_tasks": ["LangGraph 워크플로 개발"],
        },
        {
            "id": 65,
            "company_name": "둘째회사",
            "position": "LLM 엔지니어",
            "tech_stack": ["LangGraph"],
        },
    ]

    validated = validate_grounded_answer(answer, documents)

    assert len(validated.lines) == 2
    assert validated.lines[1].citation_ids == [1, 2]
    assert [reference.evidence_text for reference in validated.citations] == [
        "LangGraph 워크플로 개발",
        "LangGraph",
    ]
    assert render_grounded_answer(validated, documents) == (
        "LangGraph 워크플로를 개발합니다. [job_id:64]\n\n"
        "두 공고 모두 에이전트 워크플로 경험을 요구합니다. "
        "[job_id:64] [job_id:65]"
    )


def test_chat_service_preserves_typed_graph_outcome_and_graph_events():
    from agent.application.chat_service import ChatService
    from agent.observability.run_context import emit_run_event
    from agent.observability.run_contracts import (
        ChatRequest,
        RunPhase,
    )
    from agent.observability.run_registry import RunRegistry
    from shared.schema.investigation_schema import (
        InvestigationOutcome,
        InvestigationRequest,
    )
    from shared.schema.run_schema import RunStatus

    class FakeWorkflow:
        def run(self, query, **kwargs):
            emit_run_event(
                "run_completed",
                RunPhase.COMPLETED,
                "답변을 완료했습니다.",
                status=RunStatus.COMPLETED,
            )
            return InvestigationOutcome(
                investigation=InvestigationRequest(
                    investigation_id="investigation-contract",
                    original_query=query,
                ),
                run_status=RunStatus.COMPLETED,
                final_answer="DB 근거 답변",
            )

    events = []
    result = ChatService(
        investigation_workflow=FakeWorkflow(),
        run_registry=RunRegistry(),
    ).execute(
        ChatRequest(query="질문"),
        run_id="chat-contract-1",
        event_sink=events.append,
    )

    assert result.run_id == "chat-contract-1"
    assert result.status == RunStatus.COMPLETED
    assert result.text == "DB 근거 답변"
    assert result.metrics["run_id"] == "chat-contract-1"
    assert result.metrics["duration_sec"] >= 0
    assert [event.event for event in events] == ["run_started", "run_completed"]


def test_run_registry_tracks_cancellation_and_conversation_history():
    from agent.application.conversation_context_service import (
        load_conversation_context,
    )
    from agent.observability.run_registry import RunRegistry

    registry = RunRegistry(limit=10)
    registry.start(
        "conversation-run-1",
        "첫 질문",
        conversation_id="conversation-1",
    )
    registry.complete(
        "conversation-run-1",
        _execution_result(
            "conversation-run-1",
            answer="첫 답변",
        ),
    )
    registry.start(
        "conversation-run-2", "두 번째 질문", conversation_id="conversation-1"
    )

    cancelled = registry.request_cancel("conversation-run-2")
    registry.complete(
        "conversation-run-2",
        _execution_result(
            "conversation-run-2",
            status="cancelled",
            answer="실행을 취소했습니다.",
        ),
    )
    history = registry.conversation_history("conversation-1")
    context = load_conversation_context("conversation-1", registry=registry)

    assert cancelled["cancel_requested"] is True
    assert registry.is_cancel_requested("conversation-run-2") is True
    assert [item["run_id"] for item in history] == ["conversation-run-1"]
    assert [turn.run_id for turn in context] == ["conversation-run-1"]


def test_chat_service_stops_before_llm_when_cancel_is_requested():
    from agent.application.chat_service import ChatService
    from agent.observability.run_contracts import ChatRequest
    from agent.observability.run_registry import RunRegistry
    from shared.schema.run_schema import RunStatus

    class FailingWorkflow:
        def run(self, query, **kwargs):
            raise AssertionError("취소된 실행이 조사 그래프를 호출함")

    registry = RunRegistry()
    registry.start("cancel-before-llm", "취소할 질문")
    registry.request_cancel("cancel-before-llm")

    result = ChatService(
        investigation_workflow=FailingWorkflow(),
        run_registry=registry,
    ).execute(
        ChatRequest(query="취소할 질문"),
        run_id="cancel-before-llm",
    )

    assert result.status == RunStatus.CANCELLED
    assert result.text == "실행을 취소했습니다."


def test_chat_service_stream_owns_run_lifecycle_and_request_forwarding():
    from agent.application.chat_service import ChatService
    from agent.observability.run_context import emit_run_event
    from agent.observability.run_contracts import (
        ChatRequest,
        RunPhase,
    )
    from agent.observability.run_registry import RunRegistry
    from shared.schema.investigation_schema import (
        ClarificationAnswer,
        InvestigationOutcome,
        InvestigationRequest,
    )
    from shared.schema.run_schema import RunStatus

    calls = []

    class FakeWorkflow:
        def run(self, query, **kwargs):
            calls.append((query, kwargs))
            emit_run_event(
                "collection_started",
                RunPhase.COLLECTION,
                "수집 중",
            )
            return InvestigationOutcome(
                investigation=InvestigationRequest(
                    investigation_id="investigation-clarification",
                    original_query=query,
                ),
                run_status=RunStatus.COMPLETED,
                final_answer="개발 공고를 찾았습니다.",
                resume_mode="checkpoint_resume",
            )

    registry = RunRegistry()
    service = ChatService(FakeWorkflow(), run_registry=registry)
    request = ChatRequest(
        query="개발",
        conversation_id="conversation-context",
        investigation_id="investigation-clarification",
        clarification_answer=ClarificationAnswer(
            question_id="occupation-scope",
            custom_value="개발",
        ),
    )

    async def collect_frames():
        return [frame async for frame in service.stream(request)]

    frames = asyncio.run(collect_frames())
    run_id = frames[0].payload.run_id
    final = next(frame.payload for frame in frames if frame.kind == "final")
    registered = registry.get(run_id)

    assert frames[0].kind == "processing"
    assert frames[-1].kind == "done"
    assert any(frame.kind == "event" for frame in frames)
    assert calls[0][0] == "개발"
    assert "resume_run_id" not in calls[0][1]
    assert calls[0][1]["investigation_id"] == "investigation-clarification"
    assert calls[0][1]["clarification_answer"].question_id == "occupation-scope"
    assert calls[0][1]["clarification_answer"].custom_value == "개발"
    assert calls[0][1]["conversation_id"] == "conversation-context"
    assert final.text == "개발 공고를 찾았습니다."
    assert final.resume_mode == "checkpoint_resume"
    assert registered["status"] == "completed"
    assert registered["result"]["text"] == "개발 공고를 찾았습니다."


def test_chat_service_forwards_follow_up_as_new_investigation():
    from agent.application.chat_service import ChatService
    from agent.observability.run_contracts import ChatRequest
    from agent.observability.run_registry import RunRegistry
    from shared.schema.investigation_schema import (
        InvestigationOutcome,
        InvestigationRequest,
    )
    from shared.schema.run_schema import RunStatus

    calls = []

    class FakeWorkflow:
        def run(self, query, **kwargs):
            calls.append((query, kwargs))
            return InvestigationOutcome(
                investigation=InvestigationRequest(
                    investigation_id="investigation-follow-up",
                    original_query=query,
                ),
                run_status=RunStatus.COMPLETED,
                final_answer="이전 결과를 기준으로 비교했습니다.",
            )

    registry = RunRegistry()
    result = ChatService(FakeWorkflow(), run_registry=registry).execute(
        ChatRequest(
            query="그중 경력 조건을 비교해줘",
            conversation_id="conversation-follow-up",
        ),
        run_id="follow-up-run",
    )

    assert calls[0][1]["investigation_id"] == ""
    assert calls[0][1]["conversation_id"] == "conversation-follow-up"
    assert result.resume_mode == ""


def test_chat_api_only_serializes_service_stream(monkeypatch):
    from fastapi.testclient import TestClient

    from agent.observability.run_contracts import (
        ChatResult,
        ChatStartedPayload,
        ChatStreamFrame,
        RunEvent,
        RunPhase,
    )
    from agent.web_server import app

    requests = []

    class FakeChatService:
        async def stream(self, request):
            requests.append(request)
            yield ChatStreamFrame(
                "processing",
                ChatStartedPayload(run_id="chat-transport"),
            )
            yield ChatStreamFrame(
                "event",
                RunEvent(
                    run_id="chat-transport",
                    event="collection_started",
                    phase=RunPhase.COLLECTION,
                    message="수집 중",
                ),
            )
            yield ChatStreamFrame(
                "final",
                ChatResult(
                    run_id="chat-transport",
                    status="completed",
                    text="최종 답변",
                    conversation_id=request.conversation_id,
                ),
            )
            yield ChatStreamFrame("done")

    monkeypatch.setattr(
        "agent.web_server._chat_service_for_app",
        lambda _application: FakeChatService(),
    )
    response = TestClient(app).post(
        "/api/chat",
        json={
            "query": "테스트 질문",
            "conversation_id": "conversation-transport",
        },
    )

    assert response.status_code == 200
    assert requests[0].query == "테스트 질문"
    assert requests[0].conversation_id == "conversation-transport"
    assert "[PROCESSING]" in response.text
    assert "[EVENT]" in response.text
    assert '"text": "최종 답변"' in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


def test_cancel_run_api_marks_active_run(monkeypatch):
    from fastapi.testclient import TestClient

    from agent.web_server import app

    calls = []

    class FakeChatService:
        def cancel_run(self, run_id):
            calls.append(run_id)
            return {
                "run_id": run_id,
                "cancel_requested": True,
                "status": "running",
            }

    monkeypatch.setattr(
        "agent.web_server._chat_service_for_app",
        lambda _application: FakeChatService(),
    )
    response = TestClient(app).post("/api/runs/cancel-api-run/cancel")

    assert response.status_code == 200
    assert response.json()["cancel_requested"] is True
    assert calls == ["cancel-api-run"]


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


def test_worker_execution_session_closes_browser_after_failure_without_touching_unbound_window(
    monkeypatch,
):
    from types import SimpleNamespace

    from agent.tools.actions import ActionTools
    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime

    events = []

    class FakeActionTools:
        def __init__(self, _perception):
            pass

        def close_browser(self):
            events.append("browser_closed")
            return {"status": "success"}

    runtime = VisionWorkerRuntime(
        perception_factory=object,
        action_tools_factory=FakeActionTools,
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        with runtime.execution_session():
            runtime.get_action_tools()
            events.append("worker_started")
            raise RuntimeError("worker failed")

    assert events == [
        "worker_started",
        "browser_closed",
    ]

    user_window_closed = []
    user_window = SimpleNamespace(
        isMinimized=False,
        close=lambda: user_window_closed.append(True),
    )
    monkeypatch.setattr(
        "agent.tools.actions.gw.getActiveWindow",
        lambda: user_window,
    )
    monkeypatch.setattr(
        "agent.tools.actions.gw.getAllWindows",
        lambda: [user_window],
    )
    tools = ActionTools(SimpleNamespace(browser_window_id=None))

    cleanup = tools.close_browser()

    assert cleanup["result"] == {
        "closed": False,
        "reason": "browser_not_found",
    }
    assert user_window_closed == []


def test_browser_launch_keeps_occluded_window_rendering():
    from types import SimpleNamespace

    from agent.tools.actions import ActionTools

    tools = ActionTools(SimpleNamespace(browser_window_id=None))

    assert "--disable-backgrounding-occluded-windows" in (
        tools._browser_window_cli_args()
    )


def test_browser_window_is_fitted_to_monitor_work_area(monkeypatch):
    from types import SimpleNamespace

    from agent.tools.actions import ActionTools

    moves = []
    resizes = []
    activations = []
    work_area = {"left": 0, "top": 0, "width": 1366, "height": 728}
    perception = SimpleNamespace(
        browser_window_id=None,
        _monitor_work_area=lambda _window: work_area,
    )
    window = SimpleNamespace(
        isMaximized=False,
        width=1366,
        height=728,
        moveTo=lambda left, top: moves.append((left, top)),
        resizeTo=lambda width, height: resizes.append((width, height)),
        activate=lambda: activations.append(True),
    )
    tools = ActionTools(perception)
    monkeypatch.setattr(tools, "_browser_window_dimensions", lambda: (1920, 1080))
    monkeypatch.setattr(tools, "_sleep", lambda _seconds: None)

    assert tools._normalize_browser_window(window) is True
    assert moves == [(0, 0)]
    assert resizes == [(1366, 728)]
    assert activations == [True]


def test_vision_runtime_reuses_ocr_worker_until_application_shutdown():
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

    with runtime.execution_session():
        assert runtime.get_perception() is first_perception
        assert runtime.get_action_tools() is first_actions
        assert runtime.ocr_worker_pid == 7007

    assert events == ["browser_closed", "browser_closed"]

    runtime.close()
    assert events == [
        "browser_closed",
        "browser_closed",
        "perception_closed",
        "ocr_closed",
    ]


def test_application_runtime_keeps_vision_lazy_until_collection(monkeypatch, tmp_path):
    from agent.bootstrap import ApplicationRuntime

    monkeypatch.setenv("VISION_RECIPE_AUTO_PROMOTE", "0")
    runtime = ApplicationRuntime(tmp_path / "runtime.db")

    assert runtime.vision_runtime.is_initialized is False

    runtime.start()

    assert runtime.vision_runtime.is_initialized is False

    runtime.close()

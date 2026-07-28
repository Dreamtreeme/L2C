import json
import threading
from datetime import datetime, timedelta, timezone

import pytest


def _patch_start_observation(monkeypatch, capture):
    """분할된 시작 관찰 노드를 외부 화면 없이 실행하도록 대체한다."""

    monkeypatch.setattr("agent.graph.worker_observation.capture_screen_node", capture)
    monkeypatch.setattr("agent.graph.worker_observation.analyze_screen_node", lambda state: {})
    monkeypatch.setattr("agent.graph.worker_transition.evaluate_transition_node", lambda state: {})
    monkeypatch.setattr("agent.graph.worker_collection.apply_observation_node", lambda state: {})


def test_investigation_prompts_include_runtime_date_and_evidence_rules():
    from agent.prompts.investigation import (
        answer_prompt,
        evidence_plan_prompt,
        request_analysis_prompt,
    )

    now = datetime(
        2026,
        7,
        13,
        9,
        30,
        tzinfo=timezone(timedelta(hours=9), name="KST"),
    )
    analysis_prompt = request_analysis_prompt(now)
    evidence_prompt = evidence_plan_prompt(now)
    final_prompt = answer_prompt()

    assert "2026-07-13" in analysis_prompt
    assert "상대 기간만을 이유로 사용자에게 질문하지 마십시오" in analysis_prompt
    assert "구체적인 날짜 범위를 constraints에 설정" in analysis_prompt
    assert "assumptions에 근거를 남기십시오" in analysis_prompt
    assert "목표 자체를 정할 수 없거나" in analysis_prompt
    assert "evidence_policy=model_knowledge" in analysis_prompt
    assert "evidence_policy=web_required" in analysis_prompt
    assert "도구도 호출하지 마십시오" in analysis_prompt
    assert "posted_at" in evidence_prompt
    assert "job_body 필드가 아니라" in evidence_prompt
    assert "raw_ocr_text" in final_prompt
    assert "job_body 같은 제공되지 않은 필드" in final_prompt
    assert "created_at은 로컬 수집 시각" in evidence_prompt
    assert "트렌드는 현재 기간과 동일 길이의 이전 비교 기간" in evidence_prompt
    assert "사용자가 요청하지 않은 배경, 심화 항목" in final_prompt
    assert "별도의 조사 범위, 가정, 한계 문단을 만들지 마십시오" in final_prompt
    assert "각 ID를 독립된 토큰" in final_prompt


def test_citation_validation_normalizes_grouped_ids_before_validation():
    from agent.application.chat_service import validate_citations

    answer = "공통 기술입니다 [job_id:64, 85, 999]."

    assert validate_citations(answer, [64, 85]) == (
        "공통 기술입니다 [job_id:64] [job_id:85] [출처 확인 불가]."
    )


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


def test_default_pricing_covers_runtime_gemini_models():
    from pathlib import Path

    from agent.application.llm_cost import load_model_pricing

    prices, source = load_model_pricing()

    assert Path(source).parts[-2:] == ("config", "model_pricing.json")
    for model in ("gemini-3.6-flash", "gemini-3.5-flash-lite"):
        assert prices[model]["input_usd_per_million"] > 0
        assert prices[model]["output_usd_per_million"] > 0


def test_chat_service_returns_run_contract_and_progress_events():
    from agent.application.chat_service import ChatService
    from agent.application.run_context import emit_run_event
    from agent.application.run_contracts import RunPhase, RunStatus

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


def test_chat_service_returns_structured_clarification_without_collection():
    from agent.application.chat_service import ChatService
    from agent.application.run_context import emit_run_event
    from agent.application.run_contracts import RunPhase, RunStatus

    class FakeWorkflow:
        def run(self, query, **kwargs):
            clarification = {
                "question_id": "job_scope",
                "field": "job_scope",
                "question": "AI 직무 중 개발과 기획 중 어느 쪽을 찾을까요?",
                "missing_fields": ["직무 범위"],
                "options": [],
            }
            emit_run_event(
                "clarification_required",
                RunPhase.CLARIFICATION,
                clarification["question"],
                status=RunStatus.WAITING_INPUT,
                data=clarification,
            )
            return {
                "investigation": {"investigation_id": "investigation-clarification"},
                "run_status": "waiting_input",
                "final_answer": clarification["question"],
                "valid_ids": [],
                "clarification": clarification,
            }

    events = []
    result = ChatService(investigation_workflow=FakeWorkflow()).run(
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
        "agent.web_server._chat_service_for_app",
        lambda _application: FakeChatService(),
    )
    response = TestClient(app).post("/api/chat", json={"query": "테스트 질문"})

    assert response.status_code == 200
    assert "[PROCESSING]" in response.text
    assert "[EVENT]" in response.text
    assert '"text": "최종 답변"' in response.text
    assert "data: 최" not in response.text


def test_backend_contract_endpoint_exposes_generated_json_schemas():
    from fastapi.testclient import TestClient

    from agent.web_server import app

    response = TestClient(app).get("/api/contracts")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == 3
    assert payload["transport"]["media_type"] == "text/event-stream"
    assert "chat_request" in payload["schemas"]
    assert "taxonomy_resolution" in payload["schemas"]
    properties = payload["schemas"]["collection_tool_arguments"]["properties"]
    assert "query" in properties
    assert "site" in properties


def test_taxonomy_stats_exposes_occupation_domains(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import shared.config as config
    from agent.web_server import app

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")
    response = TestClient(app).get("/api/taxonomy/stats")

    assert response.status_code == 200
    payload = response.json()
    assert "occupation_cardinality" not in payload
    assert payload["occupation_domains"]["field"] == "occupation_domain_concept_keys"
    assert len(payload["occupation_domains"]["options"]) == 6


def test_chat_openapi_declares_event_stream_response():
    from agent.web_server import app

    response_content = app.openapi()["paths"]["/api/chat"]["post"]["responses"]["200"]["content"]

    assert "text/event-stream" in response_content


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


def test_action_request_validates_executor_tool_call_contract():
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

    assert request.source == "card_queue"
    assert request.summary == "next card"
    assert request.tool_calls[0].name == "click_marker"
    assert request.tool_calls[0].args == {
        "marker_id": 7,
        "target_label": "iOS 개발자",
    }
    assert request.tool_calls[0].id == "queue-1"


def test_action_request_rejects_unknown_tool_and_invalid_arguments():
    from pydantic import ValidationError

    from agent.graph.action_request import build_action_request

    with pytest.raises(ValidationError, match="허용되지 않은 작업자 도구"):
        build_action_request(
            "llm",
            "invalid tool",
            [{"name": "delete_account", "args": {}, "id": "bad-tool"}],
        )

    with pytest.raises(ValidationError):
        build_action_request(
            "llm",
            "missing marker",
            [{"name": "click_marker", "args": {}, "id": "bad-args"}],
        )

    with pytest.raises(ValidationError, match="at most 1 item"):
        build_action_request(
            "llm",
            "multiple actions from one capture",
            [
                {
                    "name": "type_in_marker",
                    "args": {"marker_id": 1, "text": "iOS 개발자"},
                    "id": "type",
                },
                {
                    "name": "press_key",
                    "args": {"key": "enter"},
                    "id": "enter",
                },
            ],
        )


def test_model_response_is_converted_once_to_allowed_action_request():
    from agent.graph.action_request import action_request_from_model_response

    class FakeModelResponse:
        content = "검색 결과를 아래로 이동합니다."
        tool_calls = [
            {
                "name": "scroll",
                "args": {"direction": "down", "amount": "small"},
            }
        ]

    request = action_request_from_model_response(
        FakeModelResponse(),
        allowed_tool_names=("scroll",),
    )

    assert request.source == "llm"
    assert request.summary == "검색 결과를 아래로 이동합니다."
    assert request.tool_calls[0].name == "scroll"
    assert request.tool_calls[0].args == {
        "direction": "down",
        "amount": "small",
    }
    assert request.tool_calls[0].id == "llm_0"

    with pytest.raises(ValueError, match="허용되지 않은 도구"):
        action_request_from_model_response(
            FakeModelResponse(),
            allowed_tool_names=("click_marker",),
        )


def test_worker_execution_service_delegates_prepare_and_stream(monkeypatch):
    from agent.application.worker_execution_service import execute_worker_graph
    from agent.sites import load_site_profile

    calls = []
    fake_app = object()

    monkeypatch.setattr("agent.graph.workflow.build_graph", lambda: fake_app)

    def prepare(initial_state, site_profile):
        calls.append(("prepare", initial_state["goal"], site_profile.slug))
        return {**initial_state, "prepared": True}

    def run(app, prepared_state, recursion_limit):
        calls.append(("run", app is fake_app, prepared_state["prepared"], recursion_limit))
        return {**prepared_state, "is_finished": True}, False

    final_state, hit_limit = execute_worker_graph(
        {"goal": "collect"},
        load_site_profile("wanted"),
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


def test_collection_service_completes_with_existing_database_jobs_only():
    from agent.application.collection_service import (
        CollectionOperations,
        CollectionRequest,
        CollectionService,
    )

    def run_worker(*args, **kwargs):
        return {
            "submission": {
                "run_id": "worker-existing",
                "collected_count": 0,
                "observed_job_ids": [7, 8],
                "target_count": 2,
                "task_category": "검색",
            },
            "site_name": "Wanted",
            "site_slug": "wanted",
            "keyword": "iOS 개발자",
            "target_count": 2,
            "is_finished": True,
            "hit_recursion_limit": False,
            "observed_job_ids": [7, 8],
        }

    def persist(worker_result, review):
        worker_result["persistence_validation"] = {
            "submitted_count": 0,
            "persisted_count": 0,
            "persisted_items": [],
            "rejected_count": 0,
            "rejected_items": [],
        }
        return 0, worker_result["submission"], review, "submission-existing"

    operations = CollectionOperations(
        normalize_target_count=lambda value: int(value or 0),
        normalize_task_category=lambda value: value or "검색",
        review_retries=lambda: 0,
        run_worker=run_worker,
        review_worker=lambda submission: ({"decision": "accept"}, "submission-existing"),
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

    assert result["completion_status"] == "complete"
    assert result["missing_count"] == 0
    assert result["persisted_count"] == 0
    assert result["observed_job_ids"] == [7, 8]
    assert "existing database jobs confirmed" in result["message"]


def test_collection_service_completes_when_visible_result_scope_is_exhausted():
    from agent.application.collection_service import (
        CollectionOperations,
        CollectionRequest,
        CollectionService,
    )

    def run_worker(*args, **kwargs):
        return {
            "submission": {
                "run_id": "worker-exhausted",
                "collected_count": 1,
                "target_count": 10,
                "task_category": "검색",
                "extracted_summary": {
                    "result_availability": {
                        "available_result_count": 1,
                        "count_evidence": "포지션 1",
                        "count_confidence": 0.97,
                    }
                },
            },
            "site_name": "Wanted",
            "site_slug": "wanted",
            "keyword": "QA 자동화 엔지니어",
            "target_count": 10,
            "is_finished": False,
            "hit_recursion_limit": True,
        }

    def persist(worker_result, review):
        validation = {
            "submitted_count": 1,
            "persisted_count": 1,
            "persisted_items": [{"job_id": 7, "url": "https://example.com/7"}],
            "rejected_count": 0,
            "rejected_items": [],
        }
        worker_result["persistence_validation"] = validation
        return 1, worker_result["submission"], review, "submission-exhausted"

    operations = CollectionOperations(
        normalize_target_count=lambda value: int(value or 0),
        normalize_task_category=lambda value: value or "검색",
        review_retries=lambda: 1,
        run_worker=run_worker,
        review_worker=lambda submission: (
            {
                "decision": "revise",
                "accept_collected_data": True,
                "continue_collection": True,
            },
            "submission-exhausted",
        ),
        persist_result=persist,
        render_review_feedback=lambda review: "계속 수집",
        needs_approval=lambda **kwargs: True,
        build_intermediate_report=lambda *args, **kwargs: {"target_count": 10, "persisted_count": 1},
        report_requires_more_collection=lambda report: True,
        close_browser=lambda: None,
    )

    result = CollectionService(operations).collect(
        CollectionRequest(
            search_keyword="QA 자동화 엔지니어",
            site="wanted",
            target_count=10,
            collection_intent={
                "search_keyword": "QA 자동화 엔지니어",
                "count_mode": "explicit",
                "target_count": 10,
            },
        )
    )

    assert result["completion_status"] == "complete"
    assert result["search_scope_exhausted"] is True
    assert result["missing_count"] == 9
    assert result["needs_human_approval"] is False
    assert result["persisted_count"] == 1


def test_collection_service_persists_valid_partial_data_before_retry():
    from agent.application.collection_service import (
        CollectionOperations,
        CollectionRequest,
        CollectionService,
    )

    requested_counts = []
    reviews = iter(
        [
            {
                "decision": "revise",
                "accept_collected_data": True,
                "continue_collection": True,
            },
            {
                "decision": "accept",
                "accept_collected_data": True,
                "continue_collection": False,
            },
        ]
    )

    def run_worker(*args, **kwargs):
        requested_counts.append(kwargs["target_count"])
        attempt = len(requested_counts)
        return {
            "submission": {
                "run_id": "worker-retry",
                "review_attempt": attempt - 1,
                "collected_count": 1,
                "target_count": kwargs["target_count"],
            },
            "site_name": "Wanted",
            "site_slug": "wanted",
            "keyword": "백엔드",
            "target_count": kwargs["target_count"],
            "is_finished": attempt == 2,
            "hit_recursion_limit": attempt == 1,
            "attempt": attempt,
        }

    def persist(worker_result, review):
        job_id = int(worker_result["attempt"])
        validation = {
            "submitted_count": 1,
            "persisted_count": 1,
            "persisted_items": [{"job_id": job_id, "url": f"https://example.com/{job_id}"}],
            "rejected_count": 0,
            "rejected_items": [],
        }
        worker_result["persistence_validation"] = validation
        return 1, worker_result["submission"], review, f"submission-{job_id}"

    operations = CollectionOperations(
        normalize_target_count=lambda value: int(value or 0),
        normalize_task_category=lambda value: value or "검색",
        review_retries=lambda: 1,
        run_worker=run_worker,
        review_worker=lambda submission: (next(reviews), "submission"),
        persist_result=persist,
        render_review_feedback=lambda review: "남은 공고 수집",
        needs_approval=lambda **kwargs: False,
        build_intermediate_report=lambda *args, **kwargs: {},
        report_requires_more_collection=lambda report: False,
        close_browser=lambda: None,
    )

    result = CollectionService(operations).collect(
        CollectionRequest(
            search_keyword="백엔드",
            target_count=3,
            collection_intent={
                "search_keyword": "백엔드",
                "count_mode": "explicit",
                "target_count": 3,
            },
        )
    )

    assert requested_counts == [3, 2]
    assert result["persisted_count"] == 2
    assert result["missing_count"] == 1
    assert len(result["persistence_validation"]["persisted_items"]) == 2


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


def test_worker_startup_overlaps_ocr_and_browser_before_perception(monkeypatch):
    from agent.application.worker_execution_service import prepare_worker_start_screen
    from agent.graph import worker_resources
    from agent.sites import load_site_profile

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

    def fake_perception_node(_state):
        assert order[-1] == "ocr_ready"
        order.append("perception")
        return {
            "current_url": "https://www.wanted.co.kr",
            "current_url_stale": False,
            "current_markers": [{"id": 1}],
            "recent_images": ["screen.png"],
            "low_information_screen": False,
        }

    monkeypatch.setenv("VISION_WORKER_PREOPEN_BROWSER", "1")
    monkeypatch.setattr(worker_resources, "get_action_tools", lambda: FakeActionTools())
    _patch_start_observation(monkeypatch, fake_perception_node)

    result = prepare_worker_start_screen(
        {"current_url": "", "action_history": []},
        load_site_profile("wanted"),
    )

    assert order == ["ocr_started", "browser_opened", "ocr_ready", "perception"]
    assert result["current_url"] == "https://www.wanted.co.kr"


def test_worker_graph_does_not_start_before_failed_ocr_readiness(monkeypatch):
    from agent.application.worker_execution_service import (
        OcrWorkerReadinessError,
        prepare_worker_start_screen,
    )
    from agent.graph import worker_resources
    from agent.sites import load_site_profile

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
    monkeypatch.setattr(worker_resources, "get_action_tools", lambda: FakeActionTools())
    _patch_start_observation(
        monkeypatch,
        lambda _state: perception_called.append(True) or {},
    )

    with pytest.raises(OcrWorkerReadinessError):
        prepare_worker_start_screen(
            {"current_url": "", "action_history": []},
            load_site_profile("wanted"),
        )

    assert perception_called == []


def test_worker_startup_does_not_run_graph_with_invalid_initial_screen(monkeypatch):
    from agent.application.worker_execution_service import (
        WorkerStartScreenError,
        prepare_worker_start_screen,
    )
    from agent.graph import worker_resources
    from agent.sites import load_site_profile

    class FakeSomEngine:
        def ensure_ocr_worker_ready(self):
            return None

    class FakePerception:
        som_engine = FakeSomEngine()

    class FakeActionTools:
        perception = FakePerception()

        def open_browser(self, url="", current_url="", site=""):
            return {"status": "success", "result": {"url": "https://www.wanted.co.kr"}}

    def fail_start_observation(_state):
        raise ValueError("invalid capture")

    monkeypatch.setenv("VISION_WORKER_PREOPEN_BROWSER", "1")
    monkeypatch.setattr(worker_resources, "get_action_tools", lambda: FakeActionTools())
    _patch_start_observation(monkeypatch, fail_start_observation)

    with pytest.raises(WorkerStartScreenError, match="invalid capture"):
        prepare_worker_start_screen(
            {"current_url": "", "action_history": []},
            load_site_profile("wanted"),
        )


def test_worker_startup_opens_once_and_delegates_blank_wait_to_perception(monkeypatch):
    from agent.application.worker_execution_service import prepare_worker_start_screen
    from agent.graph import worker_resources
    from agent.sites import load_site_profile

    browser_calls = []
    perception_calls = []

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

    def fake_perception_node(_state):
        perception_calls.append(True)
        return {
            "current_url": "https://www.wanted.co.kr",
            "current_url_stale": True,
            "low_information_screen": True,
        }

    monkeypatch.setenv("VISION_WORKER_PREOPEN_BROWSER", "1")
    monkeypatch.setattr(worker_resources, "get_action_tools", lambda: FakeActionTools())
    _patch_start_observation(monkeypatch, fake_perception_node)

    result = prepare_worker_start_screen(
        {"current_url": "", "action_history": []},
        load_site_profile("wanted"),
    )

    assert len(browser_calls) == 1
    assert perception_calls == [True]
    assert result["low_information_screen"] is True


def test_google_model_clients_are_reused_by_configuration(monkeypatch):
    from agent.application import model_clients
    from agent.config import clear_settings_cache
    import langchain_google_genai

    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.temperature = kwargs.get("temperature")
            created.append(kwargs)

        def with_structured_output(self, schema):
            return (self, schema)

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    clear_settings_cache()
    model_clients.clear_model_client_cache()
    first = model_clients.get_google_chat_model("model-a", temperature=0.1)
    second = model_clients.get_google_chat_model("model-a", temperature=0.1)
    structured_first = model_clients.get_structured_google_model("model-a", dict, temperature=0.1)
    structured_second = model_clients.get_structured_google_model("model-a", dict, temperature=0.1)

    assert first is second
    assert structured_first is structured_second
    assert structured_first[0] is first
    assert created == [
        {"model": "model-a", "api_key": "test-gemini-key", "temperature": 0.1}
    ]
    model_clients.clear_model_client_cache()


def test_new_gemini_models_omit_deprecated_sampling_parameters(monkeypatch):
    from agent.application import model_clients
    from agent.config import clear_settings_cache
    import langchain_google_genai

    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("VISION_LIGHTWEIGHT_MAX_OUTPUT_TOKENS", raising=False)
    clear_settings_cache()
    model_clients.clear_model_client_cache()

    commander = model_clients.get_google_chat_model("gemini-3.6-flash", temperature=0.1)
    assert commander is model_clients.get_google_chat_model(
        "gemini-3.6-flash", temperature=0.0
    )
    model_clients.get_google_chat_model("gemini-3.5-flash-lite", temperature=0.0)

    assert created == [
        {
            "model": "gemini-3.6-flash",
            "api_key": "test-gemini-key",
            "thinking_level": "medium",
        },
        {
            "model": "gemini-3.5-flash-lite",
            "api_key": "test-gemini-key",
            "thinking_level": "minimal",
            "max_tokens": 1536,
        },
    ]
    model_clients.clear_model_client_cache()


def test_new_gemini_models_strip_adapter_generated_sampling_parameters(monkeypatch):
    from agent.application import model_clients
    from agent.config import clear_settings_cache
    import langchain_google_genai

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.max_tokens = kwargs.get("max_tokens")

        def _build_base_generation_config(self, _stop, **_kwargs):
            config = {
                "candidate_count": 1,
                "temperature": 1.0,
                "top_k": 40,
                "top_p": 0.95,
                "thinking_config": {"thinking_level": "minimal"},
            }
            if self.max_tokens is not None:
                config["max_output_tokens"] = self.max_tokens
            return config

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("VISION_LIGHTWEIGHT_MAX_OUTPUT_TOKENS", raising=False)
    clear_settings_cache()
    model_clients.clear_model_client_cache()

    client = model_clients.get_google_chat_model("gemini-3.5-flash-lite")

    assert client._build_base_generation_config(None) == {
        "max_output_tokens": 1536,
        "thinking_config": {"thinking_level": "minimal"}
    }
    model_clients.clear_model_client_cache()


def test_lightweight_output_limit_supports_environment_and_explicit_override(monkeypatch):
    from agent.application import model_clients
    from agent.config import clear_settings_cache
    import langchain_google_genai

    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("VISION_LIGHTWEIGHT_MAX_OUTPUT_TOKENS", "1024")
    clear_settings_cache()
    model_clients.clear_model_client_cache()

    model_clients.get_google_chat_model("gemini-3.5-flash-lite")
    model_clients.get_google_chat_model(
        "gemini-3.5-flash-lite",
        max_output_tokens=512,
    )

    assert [item["max_tokens"] for item in created] == [1024, 512]
    model_clients.clear_model_client_cache()


def test_google_model_clients_cache_thinking_levels_separately(monkeypatch):
    from agent.application import model_clients
    from agent.config import clear_settings_cache
    import langchain_google_genai

    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    clear_settings_cache()
    model_clients.clear_model_client_cache()

    medium = model_clients.get_google_chat_model("gemini-3.6-flash")
    low = model_clients.get_google_chat_model(
        "gemini-3.6-flash",
        thinking_level="low",
    )

    assert medium is not low
    assert [item["thinking_level"] for item in created] == ["medium", "low"]
    model_clients.clear_model_client_cache()


def test_invoke_with_metrics_stream_returns_last_structured_value():
    from agent.application.run_context import invoke_with_metrics

    class FakeStructuredRunnable:
        def invoke(self, _inputs, config=None):
            raise AssertionError("스트리밍 경로에서는 invoke를 호출하면 안 됩니다.")

        def stream(self, _inputs, config=None):
            assert config and config.get("callbacks")
            yield {"value": "partial"}
            yield {"value": "complete"}

    result = invoke_with_metrics(
        FakeStructuredRunnable(),
        "input",
        "stream_test",
        stream=True,
    )

    assert result == {"value": "complete"}


def test_model_policy_uses_role_defaults_and_explicit_overrides(monkeypatch):
    from agent.config import clear_settings_cache
    from agent.application.model_policy import (
        commander_model_name,
        lightweight_model_name,
        worker_reasoning_model_name,
        worker_reasoning_thinking_level,
    )

    for env_name in (
        "COMMANDER_MODEL",
        "VISION_LIGHTWEIGHT_MODEL",
        "VISION_WORKER_REASONING_MODEL",
        "VISION_WORKER_REASONING_THINKING_LEVEL",
        "VISION_DETAIL_FINAL_EXTRACTION_MODEL",
    ):
        monkeypatch.delenv(env_name, raising=False)

    assert commander_model_name() == "gemini-3.6-flash"
    assert worker_reasoning_model_name() == "gemini-3.6-flash"
    assert worker_reasoning_thinking_level() == "low"
    assert lightweight_model_name() == "gemini-3.5-flash-lite"

    monkeypatch.setenv("COMMANDER_MODEL", "commander-override")
    monkeypatch.setenv("VISION_LIGHTWEIGHT_MODEL", "lightweight-override")
    monkeypatch.setenv("VISION_DETAIL_FINAL_EXTRACTION_MODEL", "detail-override")
    monkeypatch.setenv("VISION_WORKER_REASONING_THINKING_LEVEL", "medium")
    clear_settings_cache()

    assert worker_reasoning_model_name() == "commander-override"
    assert worker_reasoning_thinking_level() == "medium"
    assert lightweight_model_name() == "lightweight-override"
    assert lightweight_model_name("VISION_DETAIL_FINAL_EXTRACTION_MODEL") == "detail-override"


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

    with TestClient(app) as client:
        assert client.get("/api/contracts").status_code == 200

    assert calls[0][0] == "init"
    assert calls[1:] == ["start", ("close", 0.5)]


def test_browser_closes_by_default(monkeypatch):
    from agent.application import worker_execution_service
    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime

    closed = []

    class FakeActionTools:
        def __init__(self, _perception):
            pass

        def close_browser(self):
            closed.append(True)
            return {"status": "success"}

    monkeypatch.delenv("VISION_CLOSE_BROWSER_AFTER_RUN", raising=False)
    runtime = VisionWorkerRuntime(
        perception_factory=object,
        action_tools_factory=FakeActionTools,
    )
    runtime.get_action_tools()

    worker_execution_service.close_browser_after_run(worker_runtime=runtime)

    assert closed == [True]


def test_browser_stays_open_when_explicitly_disabled(monkeypatch):
    from agent.application import worker_execution_service
    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime

    closed = []

    class FakeActionTools:
        def __init__(self, _perception):
            pass

        def close_browser(self):
            closed.append(True)
            return {"status": "success"}

    monkeypatch.setenv("VISION_CLOSE_BROWSER_AFTER_RUN", "0")
    runtime = VisionWorkerRuntime(
        perception_factory=object,
        action_tools_factory=FakeActionTools,
    )
    runtime.get_action_tools()

    worker_execution_service.close_browser_after_run(worker_runtime=runtime)

    assert closed == []


def test_vision_runtime_reuses_ocr_worker_until_application_shutdown(monkeypatch):
    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime

    events = []

    class FakeWorker:
        pid = 7007

        @staticmethod
        def poll():
            return None

    class FakeSomEngine:
        def __init__(self):
            self._ocr_worker = FakeWorker()

        def close(self):
            events.append("ocr_closed")
            self._ocr_worker = None

    class FakePerception:
        def __init__(self):
            self.som_engine = FakeSomEngine()
            self._browser_window_id = None

        def close(self):
            events.append("perception_closed")
            self.som_engine.close()

    class FakeActionTools:
        def __init__(self, perception):
            self.perception = perception

        def close_browser(self):
            events.append("browser_closed")
            return {"status": "success"}

    monkeypatch.delenv("VISION_CLOSE_BROWSER_AFTER_RUN", raising=False)
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


def test_vision_runtime_reuses_bound_ui_model(monkeypatch):
    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime

    calls = []
    bound_model = object()

    class FakeModel:
        def bind_tools(self, schemas):
            calls.append(list(schemas))
            return bound_model

    monkeypatch.setattr(
        "agent.application.model_clients.get_google_chat_model",
        lambda *args, **kwargs: FakeModel(),
    )
    runtime = VisionWorkerRuntime()
    schemas = {"click_marker": object()}

    first_model = runtime.get_ui_model_with_tools(("click_marker",), schemas)
    second_model = runtime.get_ui_model_with_tools(("click_marker",), schemas)

    assert first_model is bound_model
    assert second_model is bound_model
    assert calls == [[schemas["click_marker"]]]
    assert runtime.resource_snapshot()["ui_model_variant_count"] == 1

    runtime.close()


def test_application_runtime_keeps_vision_lazy_until_collection(monkeypatch, tmp_path):
    from agent.runtime.application_runtime import ApplicationRuntime

    monkeypatch.setenv("VISION_RECIPE_AUTO_PROMOTE", "0")
    runtime = ApplicationRuntime(tmp_path / "runtime.db")

    assert runtime.vision_runtime.is_initialized is False

    runtime.start()

    assert runtime.is_started is True
    assert runtime.vision_runtime.is_initialized is False

    runtime.close()

    assert runtime.is_started is False


def test_detail_extraction_prompt_prioritizes_page_text_over_ocr_hints():
    from agent.prompts.detail_extraction import build_detail_extraction_system_prompt

    prompt = build_detail_extraction_system_prompt("채용공고를 정리하십시오.")

    assert "서로 인접한 페이지 텍스트를 가장 우선" in prompt
    assert "이미지 내부 로고 OCR" in prompt
    assert "보조 근거로만 사용" in prompt
    assert "active_result_card" not in prompt
    assert "회사명은 로고" not in prompt
    assert "직무명 괄호 안의 세부 분야" in prompt
    assert "evidence_hash는 출력하지 마십시오" in prompt


def test_openai_detail_extraction_schema_keeps_numeric_experience_fields():
    from agent.application.detail_extraction_service import OpenAIDetailExtractionLLM

    properties = OpenAIDetailExtractionLLM._job_posting_schema()["properties"]

    assert "experience_min" in properties
    assert "experience_max" in properties


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
                "screens": ["C:/tmp/loading.png", "C:/tmp/detail-screen.png"],
                "screen_evidence": [
                    {"path": "C:/tmp/loading.png", "added_lines": 0},
                    {"path": "C:/tmp/detail-screen.png", "added_lines": 1},
                ],
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
    assert result["raw_ocr_text"] == "1. 상세 페이지 회사 상세 페이지 직무"
    assert result["_evidence_screenshot_path"] == "C:/tmp/detail-screen.png"

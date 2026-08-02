import time

import pytest

from agent.graph import (
    worker_action_guard,
    worker_execution,
    worker_execution_dispatch,
    worker_observation,
    worker_recording,
)
from agent.graph.action_request import build_action_request
from agent.graph.workflow import (
    route_after_start,
    route_after_recording,
    route_after_reasoning,
    route_after_reflex,
    route_after_selection,
    route_after_transition,
)
from agent.graph.worker_state_contract import current_observation_matches_capture
from agent.graph.worker_state_action_result import (
    StateActionOutcome,
    StateActionUpdate,
)


def _request(source: str, tool_calls: list[dict]):
    return build_action_request(source, "test", tool_calls)


def _execution_state(request, **overrides):
    state = {
        "goal": "테스트",
        "pending_action": request,
        "current_markers": [],
        "current_url": "https://example.com/jobs",
        "current_url_stale": False,
        "current_capture_id": "worker-test:capture:0003",
        "screen_signature": {},
        "recent_images": [],
        "action_history": [],
        "extracted_jd": {},
        "collected_data": [],
        "error_count": 0,
        "is_finished": False,
    }
    state.update(overrides)
    return state


def test_action_request_allows_only_supported_reflex_action_group():
    calls = [
        {
            "name": "type_in_marker",
            "args": {"marker_id": 1, "text": "AI 엔지니어"},
            "id": "input",
        },
        {
            "name": "click_marker",
            "args": {"marker_id": 2},
            "id": "submit",
        },
    ]

    with pytest.raises(ValueError):
        build_action_request("llm", "잘못된 복수 행동", calls)

    with pytest.raises(ValueError):
        build_action_request(
            "reflex",
            "두 타깃 행동은 중간 화면 관찰이 필요함",
            calls,
            metadata={"execution_unit": "recipe_transition"},
        )

    grouped = build_action_request(
        "reflex",
        "검색어 입력 후 제출",
        [
            calls[0],
            {
                "name": "press_key",
                "args": {"key": "enter"},
                "id": "submit",
            },
        ],
        metadata={"execution_unit": "recipe_transition"},
    )
    assert [call.name for call in grouped.tool_calls] == [
        "type_in_marker",
        "press_key",
    ]


def test_selection_routes_by_action_source_without_hit_flags(monkeypatch):
    monkeypatch.setenv("REFLEX_ENABLED", "1")
    observed = {"ocr_complete": True}
    assert route_after_selection(observed) == "reflex"

    queue_request = _request(
        "job_card_queue",
        [{"name": "click_marker", "args": {"marker_id": 1}, "id": "queue"}],
    )
    assert (
        route_after_selection(
            {**observed, "pending_action": queue_request}
        )
        == "execution"
    )

    reflex_request = _request(
        "reflex",
        [{"name": "press_key", "args": {"key": "enter"}, "id": "reflex"}],
    )
    assert (
        route_after_reflex({"pending_action": reflex_request})
        == "execution"
    )
    assert route_after_reflex({}) == "reasoning"


def test_active_reflex_recipe_keeps_selection_for_reflex(monkeypatch):
    from agent.graph import worker_selection

    monkeypatch.setenv("REFLEX_ENABLED", "1")

    result = worker_selection.selection_node(
        {
            "ocr_complete": True,
            "transition_result": {
                "status": "ready",
                "action": "type_in_marker",
            },
            "active_reflex_recipe": {
                "recipe_key": "recipe-search-set",
                "current_transition_index": 1,
                "transition_count": 2,
            },
        }
    )

    assert (
        route_after_selection(
            {
                **result,
                "ocr_complete": True,
                "transition_result": {"status": "ready"},
                "active_reflex_recipe": {
                    "recipe_key": "recipe-search-set",
                    "current_transition_index": 1,
                    "transition_count": 2,
                },
            }
        )
        == "reflex"
    )


def test_transition_requires_collection_only_after_ocr():
    assert route_after_transition({"ocr_complete": False}) == "selection"
    assert route_after_transition({"ocr_complete": True}) == "collection"
    assert (
        route_after_transition(
            {
                "ocr_complete": True,
                "current_capture_id": "capture:2",
                "ocr_capture_id": "capture:1",
            }
        )
        == "selection"
    )


def test_worker_observation_contract_rejects_mixed_capture_state():
    valid = {
        "ocr_complete": True,
        "current_capture_id": "capture:2",
        "ocr_capture_id": "capture:2",
        "current_screenshot": "screen.png",
    }
    mixed = {
        **valid,
        "ocr_capture_id": "capture:1",
    }

    assert current_observation_matches_capture(valid) is True
    assert current_observation_matches_capture(mixed) is False


def test_worker_start_does_not_reuse_ocr_from_another_capture():
    stale_observation = {
        "ocr_complete": True,
        "current_capture_id": "capture:2",
        "ocr_capture_id": "capture:1",
        "current_markers": [
            {"id": 1, "bbox": [0, 0, 10, 10], "text": "검색"},
        ],
        "current_page_role": "search",
        "recent_images": ["screen.png"],
    }

    assert route_after_start(stale_observation) == "capture"
    assert (
        route_after_start(
            {
                **stale_observation,
                "ocr_capture_id": "capture:2",
            }
        )
        == "selection"
    )


def test_unchanged_transition_probe_clears_ocr_capture_owner(monkeypatch):
    class ProbeOnlyPerception:
        def wait_for_transition_phash_match(self, *args, **kwargs):
            return False

    monkeypatch.setattr(
        worker_observation,
        "_perception_engine",
        lambda: ProbeOnlyPerception(),
    )

    result = worker_observation.capture_node(
        {
            "current_capture_id": "capture:0001",
            "ocr_capture_id": "capture:0001",
            "ocr_complete": True,
            "transition_request": {
                "pending_target_phash": "abcd",
                "started_at": time.time(),
            },
        }
    )

    assert result["transition_probe_unchanged"] is True
    assert result["ocr_complete"] is False
    assert result["ocr_capture_id"] == ""


def test_screen_changing_action_always_routes_to_capture():
    assert (
        route_after_recording(
            {"transition_request": {"action": "click_marker"}}
        )
        == "capture"
    )
    assert route_after_recording({}) == "reasoning"


def test_loading_card_screen_routes_from_reasoning_to_recapture():
    state = {
        "pending_action": None,
        "job_card_selection_trace": {"reason": "screen_loading"},
    }

    assert route_after_reasoning(state) == "capture"
    assert (
        route_after_reasoning({"pending_action": object()})
        == "execution"
    )


def test_capture_screen_assigns_run_scoped_incrementing_id(monkeypatch):
    captures = iter(["screen-1.png", "screen-2.png", "screen-retry-1.png"])

    class FakePerception:
        last_capture_quality = {}

        def capture_usable_screen(self):
            return next(captures)

    monkeypatch.setattr(
        worker_observation,
        "_perception_engine",
        lambda: FakePerception(),
    )
    state = {
        "worker_run_id": "worker-test",
        "capture_sequence": 0,
        "current_url": "https://example.com",
        "current_url_stale": False,
    }

    first = worker_observation.capture_node(state)
    second = worker_observation.capture_node({**state, **first})
    next_run_first = worker_observation.capture_node(
        {
            **state,
            "worker_run_id": "worker-test-retry",
        }
    )

    assert first["current_capture_id"] == "worker-test:capture:0001"
    assert first["capture_sequence"] == 1
    assert second["current_capture_id"] == "worker-test:capture:0002"
    assert second["capture_sequence"] == 2
    assert (
        next_run_first["current_capture_id"]
        == "worker-test-retry:capture:0001"
    )


def test_worker_state_factory_matches_the_declared_state_contract():
    from agent.graph.state import GraphState
    from agent.graph.state_factory import create_worker_state

    state = create_worker_state("테스트 목표")

    assert set(state).issubset(GraphState.__annotations__)
    assert "worker_attempt_index" not in state
    assert state["goal"] == "테스트 목표"


def test_atomic_execution_and_recording_are_separate(monkeypatch):
    calls: list[str] = []

    def fake_dispatch(action_name, args, get_bbox, current_url=""):
        calls.append(action_name)
        get_bbox(args["marker_id"])
        return {"action": action_name, "status": "success", "result": "clicked"}

    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_ui_action",
        fake_dispatch,
    )
    monkeypatch.setattr(
        worker_action_guard,
        "check_current_reasoning_screen",
        lambda *_args, **_kwargs: {
            "checked": True,
            "stale": False,
            "must_refresh": False,
            "reason": "screen_unchanged",
        },
    )
    request = _request(
        "llm",
        [
            {"name": "click_marker", "args": {"marker_id": 1}, "id": "click"},
        ],
    )
    result = worker_execution.execution_node(
        {
            "goal": "테스트",
            "pending_action": request,
            "current_markers": [{"id": 1, "bbox": [0, 0, 10, 10], "text": "공고"}],
            "current_url": "https://example.com/jobs",
            "current_url_stale": False,
            "current_capture_id": "worker-test:capture:0003",
            "screen_signature": {},
            "recent_images": [],
            "action_history": [],
            "extracted_jd": {},
            "collected_data": [],
            "error_count": 0,
            "is_finished": False,
        }
    )

    assert calls == ["click_marker"]
    assert [item["status"] for item in result["action_history"]] == ["success"]
    assert result["transition_request"]["action"] == "click_marker"
    assert (
        result["transition_request"]["from_capture_id"]
        == "worker-test:capture:0003"
    )
    assert (
        result["action_history"][0]["decision_capture_id"]
        == "worker-test:capture:0003"
    )
    assert "recorded_steps" not in result

    recorded = worker_recording.recording_node(result)
    assert recorded["recorded_steps"][0]["action"] == "click_marker"
    assert (
        recorded["recorded_steps"][0]["decision_capture_id"]
        == "worker-test:capture:0003"
    )
    assert (
        recorded["feedback_episodes"][0]["observation"]["before"]["capture_id"]
        == "worker-test:capture:0003"
    )
    assert len(recorded["feedback_episodes"]) == 1


def test_reflex_transition_executes_input_and_enter_without_recapture(
    monkeypatch,
):
    calls: list[str] = []

    def fake_dispatch(action_name, args, get_bbox, current_url=""):
        calls.append(action_name)
        if args.get("marker_id") is not None:
            get_bbox(args["marker_id"])
        return {
            "action": action_name,
            "status": "success",
            "result": "executed",
        }

    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_ui_action",
        fake_dispatch,
    )
    request = build_action_request(
        "reflex",
        "검색 입력 전이",
        [
            {
                "name": "type_in_marker",
                "args": {
                    "marker_id": 1,
                    "text": "AI 엔지니어",
                },
                "id": "type",
            },
            {
                "name": "press_key",
                "args": {"key": "enter"},
                "id": "submit",
            },
        ],
        metadata={
            "execution_unit": "recipe_transition",
            "recipe_key": "path6#search",
            "transition_index": 0,
            "transition_count": 1,
            "transition_actions": [
                "type_in_marker",
                "press_key",
            ],
            "before_state": {
                "url_template": "example.com/search-overlay",
                "page_role": "search_overlay",
            },
            "expected_after_state": {
                "url_template": "example.com/jobs",
                "page_role": "search",
                "screen_context_signature": {
                    "phash": "f" * 16,
                    "size": [1920, 1080],
                },
            },
        },
    )

    result = worker_execution.execution_node(
        _execution_state(
            request,
            current_markers=[
                {
                    "id": 1,
                    "bbox": [0, 0, 100, 30],
                    "text": "검색어",
                    "type": "input",
                }
            ],
            reflex_trace={
                "recipe_key": "path6#search",
                "tool_calls": {},
            },
        )
    )

    assert calls == ["type_in_marker", "press_key"]
    assert len(result["action_history"]) == 2
    assert result["transition_request"]["action"] == "press_key"
    assert result["transition_request"]["transition_actions"] == [
        "type_in_marker",
        "press_key",
    ]
    assert (
        result["transition_request"]["before_page_role"]
        == "search_overlay"
    )
    assert (
        result["transition_request"]["expected_after_state"][
            "url_template"
        ]
        == "example.com/jobs"
    )


def test_detail_completion_guard_blocks_more_screen_exploration(monkeypatch):
    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_ui_action",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("목록 복귀 전에 상세 화면을 더 탐색하면 안 됩니다.")
        ),
    )
    request = _request(
        "llm",
        [
            {
                "name": "scroll",
                "args": {"direction": "down"},
                "id": "scroll",
            }
        ],
    )
    result = worker_execution.execution_node(
        _execution_state(
            request,
            current_url="https://example.com/jobs/1",
            return_to_job_results={
                "url": "https://example.com/jobs/1",
                "reason": "required_fields_complete",
            },
        )
    )

    assert result["action_history"][0]["status"] == "skipped"
    assert (
        result["action_history"][0]["reason"]
        == "return_to_job_results"
    )
    assert result["error_count"] == 0


def test_failed_ui_dispatch_is_recorded_once(monkeypatch):
    def fail_dispatch(*args, **kwargs):
        raise RuntimeError("physical input failed")

    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_ui_action",
        fail_dispatch,
    )
    request = _request(
        "job_card_queue",
        [
            {
                "name": "click_marker",
                "args": {"marker_id": 1},
                "id": "click",
            }
        ],
    )
    result = worker_execution.execution_node(
        _execution_state(
            request,
            current_markers=[
                {"id": 1, "bbox": [0, 0, 10, 10], "text": "공고"},
            ],
        )
    )

    assert len(result["action_history"]) == 1
    assert result["action_history"][0]["status"] == "error"
    assert result["action_history"][0]["error"] == "physical input failed"
    assert len(result["execution_records"]) == 1
    assert result["error_count"] == 1


def test_returned_ui_error_does_not_create_screen_transition(monkeypatch):
    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_ui_action",
        lambda *args, **kwargs: {
            "action": "click_marker",
            "status": "error",
            "error": "physical input failed",
        },
    )
    request = _request(
        "job_card_queue",
        [
            {
                "name": "click_marker",
                "args": {"marker_id": 1},
                "id": "click",
            }
        ],
    )

    result = worker_execution.execution_node(
        _execution_state(
            request,
            current_markers=[
                {"id": 1, "bbox": [0, 0, 10, 10], "text": "공고"},
            ],
        )
    )

    assert result["action_history"][0]["status"] == "error"
    assert result["error_count"] == 1
    assert not result["transition_request"]


def test_returned_state_error_is_recorded_as_failure(monkeypatch):
    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_state_action",
        lambda *args, **kwargs: StateActionOutcome(
            result={
                "action": "update_extracted_info",
                "status": "error",
                "result": "invalid payload",
            },
            jobs={"jobs": [{"company_name": "기존 회사"}]},
        ),
    )
    request = _request(
        "llm",
        [
            {
                "name": "update_extracted_info",
                "args": {"data_json": "{}"},
                "id": "update",
            }
        ],
    )

    result = worker_execution.execution_node(
        _execution_state(
            request,
            extracted_jd={"jobs": [{"company_name": "원래 회사"}]},
        )
    )

    assert result["action_history"][0]["status"] == "error"
    assert result["action_history"][0]["error"] == "invalid payload"
    assert result["error_count"] == 1
    assert result["extracted_jd"] == {
        "jobs": [{"company_name": "원래 회사"}]
    }


def test_stored_job_card_queue_schedules_first_card(monkeypatch):
    queued_card = {
        "queue_id": "card-1",
        "status": "pending",
        "title": "첫 번째 공고",
        "source_marker_id": 4,
    }

    def fake_dispatch(*args, **kwargs):
        return StateActionOutcome(
            result={
                "action": "set_job_card_queue",
                "status": "success",
                "result": "stored",
            },
            jobs={},
            state_update=StateActionUpdate(
                job_card_queue=[queued_card],
                job_results_memory={"url": "https://example.com/jobs"},
                job_results_availability={},
            ),
        )

    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_state_action",
        fake_dispatch,
    )
    request = _request(
        "llm",
        [
            {
                "name": "set_job_card_queue",
                "args": {
                    "cards": [
                        {"marker_id": 4, "title": "첫 번째 공고"},
                    ]
                },
                "id": "queue",
            }
        ],
    )
    result = worker_execution.execution_node(_execution_state(request))

    follow_up = result["pending_action"]
    assert follow_up.source == "job_card_queue"
    assert follow_up.tool_calls[0].name == "click_marker"
    assert follow_up.tool_calls[0].args["marker_id"] == 4
    assert result["job_card_queue"] == [queued_card]


def test_existing_job_card_queue_finishes_without_opening_detail(monkeypatch):
    existing_cards = [
        {
            "queue_id": "card-1",
            "status": "skipped",
            "title": "기존 공고 1",
            "job_id": 7,
        },
        {
            "queue_id": "card-2",
            "status": "skipped",
            "title": "기존 공고 2",
            "job_id": 8,
        },
    ]

    def fake_dispatch(*args, **kwargs):
        return StateActionOutcome(
            result={
                "action": "set_job_card_queue",
                "status": "success",
                "result": "stored",
            },
            jobs={},
            state_update=StateActionUpdate(
                job_card_queue=existing_cards,
                job_results_memory={"url": "https://example.com/jobs"},
                job_results_availability={},
            ),
        )

    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_state_action",
        fake_dispatch,
    )
    request = _request(
        "llm",
        [
            {
                "name": "set_job_card_queue",
                "args": {"cards": []},
                "id": "queue",
            }
        ],
    )
    result = worker_execution.execution_node(
        _execution_state(
            request,
            recipe_params={
                "count_mode": "explicit",
                "target_count": 2,
            },
        )
    )

    assert result["is_finished"] is True
    assert result["pending_action"] is None
    assert result["job_card_queue"] == existing_cards
    assert result["action_history"][0]["resolved_count"] == 2


def test_graph_custom_event_is_shared_by_metrics_and_sse():
    from agent.application.run_context import run_context
    from agent.observability.graph_events import forward_graph_event

    events = []
    with run_context(run_id="graph-event-test", event_sink=events.append) as (context, _created):
        forward_graph_event(
            {
                "event": "graph_step_started",
                "stage": "capture",
                "component": "graph:capture",
            }
        )
        forward_graph_event(
            {
                "event": "graph_step_finished",
                "stage": "capture",
                "component": "graph:capture",
                "duration_sec": 0.25,
                "success": True,
            }
        )
        snapshot = context.snapshot()

    graph_events = [event for event in events if event.event.startswith("graph_step_")]
    assert [event.event for event in graph_events] == ["graph_step_started", "graph_step_finished"]
    assert {event.run_id for event in graph_events} == {"graph-event-test"}
    assert snapshot["steps"] == [
        {
            "component": "graph:capture",
            "stage": "capture",
            "duration_sec": 0.25,
            "success": True,
        }
    ]


def test_worker_graph_forwards_custom_stream_and_preserves_values(monkeypatch):
    from agent.application import worker_execution_service

    forwarded = []

    class FakeGraph:
        def stream(self, state, config=None, stream_mode=None):
            assert stream_mode == ["values", "custom"]
            yield ("custom", {"event": "graph_step_started", "stage": "capture"})
            yield ("values", {**state, "is_finished": True})

    monkeypatch.setattr(worker_execution_service, "forward_graph_event", forwarded.append)
    state, limited = worker_execution_service.run_graph_with_last_state(
        FakeGraph(),
        {"is_finished": False},
        60,
    )

    assert forwarded == [{"event": "graph_step_started", "stage": "capture"}]
    assert state["is_finished"] is True
    assert limited is False

from agent.graph import worker_execution, worker_observation, worker_recording
from agent.graph.action_request import build_action_request
from agent.graph.workflow import (
    route_after_action,
    route_after_reflex,
    route_after_selection,
    route_after_transition,
)


def _request(source: str, tool_calls: list[dict]):
    return build_action_request(source, "test", tool_calls)


def test_selection_routes_by_action_source_without_hit_flags(monkeypatch):
    monkeypatch.setenv("REFLEX_ENABLED", "1")
    observed = {"ocr_complete": True}
    assert route_after_selection(observed) == "reflex"

    queue_request = _request(
        "card_queue",
        [{"name": "click_marker", "args": {"marker_id": 1}, "id": "queue"}],
    )
    assert route_after_selection({**observed, "pending_action": queue_request}) == "action"

    reflex_request = _request(
        "reflex",
        [{"name": "press_key", "args": {"key": "enter"}, "id": "reflex"}],
    )
    assert route_after_reflex({"pending_action": reflex_request}) == "action"
    assert route_after_reflex({}) == "reasoning"


def test_transition_requires_collection_only_after_ocr():
    assert route_after_transition({"ocr_complete": False}) == "selection"
    assert route_after_transition({"ocr_complete": True}) == "collection"


def test_screen_changing_action_always_routes_to_capture():
    assert route_after_action({"pending_transition": {"action": "click_marker"}}) == "capture"
    assert route_after_action({}) == "reasoning"


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
        "worker_attempt_index": 0,
        "capture_sequence": 0,
        "current_url": "https://example.com",
        "current_url_stale": False,
    }

    first = worker_observation.capture_screen_node(state)
    second = worker_observation.capture_screen_node({**state, **first})
    retry_first = worker_observation.capture_screen_node(
        {
            **state,
            "worker_attempt_index": 1,
        }
    )

    assert first["current_capture_id"] == "worker-test:attempt:00:capture:0001"
    assert first["capture_sequence"] == 1
    assert second["current_capture_id"] == "worker-test:attempt:00:capture:0002"
    assert second["capture_sequence"] == 2
    assert (
        retry_first["current_capture_id"]
        == "worker-test:attempt:01:capture:0001"
    )


def test_atomic_execution_and_recording_are_separate(monkeypatch):
    calls: list[str] = []

    def fake_dispatch(action_name, args, get_bbox, current_url=""):
        calls.append(action_name)
        get_bbox(args["marker_id"])
        return {"action": action_name, "status": "success", "result": "clicked"}

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch)
    request = _request(
        "llm",
        [
            {"name": "click_marker", "args": {"marker_id": 1}, "id": "click"},
        ],
    )
    result = worker_execution.action_node(
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
    assert result["pending_transition"]["action"] == "click_marker"
    assert (
        result["pending_transition"]["from_capture_id"]
        == "worker-test:capture:0003"
    )
    assert (
        result["action_history"][0]["decision_capture_id"]
        == "worker-test:capture:0003"
    )
    assert "recorded_steps" not in result

    recorded = worker_recording.record_execution_node(result)
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

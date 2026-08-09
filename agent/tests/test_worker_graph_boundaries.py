import inspect
import time

import pytest

from agent.graph import (
    worker_execution,
    worker_execution_dispatch,
    worker_observation,
)
from agent.runtime.worker_contracts import action_event_results, build_action_request
from agent.graph.workflow import (
    route_after_execution,
    route_after_start,
    route_after_reasoning,
    route_after_reflex,
    route_after_selection,
)
from agent.runtime.worker_state import current_observation_ready
from agent.graph.worker_execution_dispatch import (
    StateActionOutcome,
    StateActionUpdate,
)
from agent.tests.worker_test_support import (
    apply_update,
    node_runtime,
    worker_state,
)
from shared.schema.collection_intent import CollectionIntent
from shared.schema.jd_schema import JobCapture


def _request(source: str, tool_calls: list[dict]):
    return build_action_request(source, "test", tool_calls)


class _FakeVisionRuntime:
    def __init__(self, perception=None):
        self.perception = perception

    def get_perception(self):
        return self.perception

    def get_action_tools(self):
        return object()

    def check_reasoning_screen(self, *_args, **_kwargs):
        return {
            "checked": True,
            "stale": False,
            "must_refresh": False,
            "reason": "screen_unchanged",
        }


def _execution_state(
    request,
    *,
    current_markers=None,
    current_url="https://example.com/jobs",
    reflex_trace=None,
    job_captures=None,
    collection_intent: CollectionIntent | None = None,
):
    return worker_state(
        goal="테스트",
        request={"collection_intent": collection_intent or CollectionIntent()},
        observation={
            "current_markers": list(current_markers or []),
            "current_url": current_url,
            "current_url_stale": False,
            "observation_id": "worker-test:observation:0003",
            "screen_signature": {},
            "current_screenshot": "",
        },
        decision={"pending_action": request},
        replay={"reflex_trace": dict(reflex_trace or {})},
        collection={
            "job_captures": list(job_captures or []),
        },
    )


def _run_execution(state):
    update = worker_execution.execution_node(
        state,
        node_runtime(_FakeVisionRuntime()),
    )
    return apply_update(state, update)


def _observed_state(**observation):
    return worker_state(
        observation={
            "observation_id": "observation:0001",
            "current_screenshot": "screen.png",
            "ocr_complete": True,
            **observation,
        }
    )


def test_worker_nodes_keep_langgraph_runtime_parameter_name():
    from agent.graph.worker_reasoning import reasoning_node
    from agent.graph.worker_selection import selection_node
    from agent.graph.worker_transition import transition_node
    from agent.recipe.replay_runtime import attempt_reflex_replay

    nodes = [
        worker_observation.capture_node,
        worker_observation.ocr_node,
        transition_node,
        selection_node,
        attempt_reflex_replay,
        reasoning_node,
        worker_execution.execution_node,
    ]

    for node in nodes:
        assert list(inspect.signature(node).parameters)[1] == "runtime"


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


@pytest.mark.parametrize(
    "action_name",
    ["update_extracted_info", "close_browser"],
)
def test_action_request_rejects_removed_worker_tools(action_name):
    with pytest.raises(ValueError, match="허용되지 않은 작업자 도구"):
        _request(
            "llm",
            [{"name": action_name, "args": {}, "id": "removed"}],
        )


def test_job_card_queue_accepts_only_canonical_card_fields():
    request = _request(
        "llm",
        [
            {
                "name": "set_job_card_queue",
                "args": {
                    "cards": [
                        {
                            "marker_id": 7,
                            "title": "AI 엔지니어",
                            "company": "예시회사",
                        }
                    ]
                },
                "id": "queue",
            }
        ],
    )

    assert request.tool_calls[0].args["cards"] == [
        {
            "marker_id": 7,
            "title": "AI 엔지니어",
            "company": "예시회사",
        }
    ]
    with pytest.raises(ValueError, match="target_label"):
        _request(
            "llm",
            [
                {
                    "name": "set_job_card_queue",
                    "args": {
                        "cards": [
                            {
                                "marker_id": 7,
                                "title": "AI 엔지니어",
                                "target_label": "AI 엔지니어",
                            }
                        ]
                    },
                    "id": "legacy-queue",
                }
            ],
        )


def test_selection_routes_by_action_source_without_hit_flags(monkeypatch):
    monkeypatch.setenv("REFLEX_ENABLED", "1")
    observed = _observed_state()
    assert route_after_selection(observed) == "reflex"

    queue_request = _request(
        "job_card_queue",
        [{"name": "click_marker", "args": {"marker_id": 1}, "id": "queue"}],
    )
    assert (
        route_after_selection(
            worker_state(
                observation=observed["observation"],
                decision={"pending_action": queue_request},
            )
        )
        == "execution"
    )

    reflex_request = _request(
        "reflex",
        [{"name": "press_key", "args": {"key": "enter"}, "id": "reflex"}],
    )
    assert (
        route_after_reflex(worker_state(decision={"pending_action": reflex_request}))
        == "execution"
    )
    assert route_after_reflex(worker_state()) == "reasoning"


def test_active_reflex_recipe_keeps_selection_for_reflex(monkeypatch):
    from agent.graph import worker_selection

    monkeypatch.setenv("REFLEX_ENABLED", "1")

    state = worker_state(
        observation={
            "observation_id": "observation:0001",
            "current_screenshot": "screen.png",
            "ocr_complete": True,
        },
        transition={
            "transition_result": {
                "status": "ready",
                "action": "type_in_marker",
            },
        },
        replay={
            "active_reflex_recipe": {
                "recipe_key": "recipe-search-set",
                "current_transition_index": 1,
                "transition_count": 2,
            },
        },
    )
    result = apply_update(
        state,
        worker_selection.selection_node(state, node_runtime()),
    )

    assert route_after_selection(result) == "reflex"


def test_worker_observation_is_ready_only_after_ocr():
    ready = {
        "ocr_complete": True,
        "observation_id": "observation:2",
        "current_screenshot": "screen.png",
    }
    assert current_observation_ready(worker_state(observation=ready)) is True
    assert (
        current_observation_ready(
            worker_state(observation={**ready, "ocr_complete": False})
        )
        is False
    )
    assert current_observation_ready(worker_state(observation={})) is False


def test_worker_start_reuses_only_completed_observation():
    observation = {
        "ocr_complete": False,
        "observation_id": "observation:2",
        "current_markers": [
            {"id": 1, "bbox": [0, 0, 10, 10], "text": "검색"},
        ],
        "current_page_role": "search",
        "current_screenshot": "screen.png",
    }

    assert route_after_start(worker_state(observation=observation)) == "capture"
    assert (
        route_after_start(
            worker_state(
                observation={
                    **observation,
                    "ocr_complete": True,
                }
            )
        )
        == "selection"
    )


def test_go_back_waits_for_cv_change_before_stable_capture(
    monkeypatch,
    tmp_path,
):
    from PIL import Image

    before = tmp_path / "detail.png"
    after = tmp_path / "results.png"
    Image.new("RGB", (80, 60), "black").save(before)
    Image.new("RGB", (80, 60), "white").save(after)
    calls = []

    class CvPerception:
        last_capture_quality = {
            "low_information": False,
            "stable": True,
        }

        def wait_for_transition_change(self, reference_image_path):
            calls.append(("change", reference_image_path))
            return True

        def capture_usable_screen(self):
            calls.append(("stable_capture", str(after)))
            return after

        def get_current_url(self):
            return ""

    state = worker_state(
        observation={
            "observation_id": "observation:0001",
            "ocr_complete": True,
        },
        transition={
            "transition_request": {
                "action": "go_back",
                "before_screenshot": str(before),
                "started_at": time.time(),
            },
        },
    )
    result = apply_update(
        state,
        worker_observation.capture_node(
            state,
            node_runtime(_FakeVisionRuntime(CvPerception())),
        ),
    )

    assert calls == [
        ("change", str(before)),
        ("stable_capture", str(after)),
    ]
    assert result["observation"]["current_screenshot"] == str(after)
    assert result["observation"]["ocr_complete"] is False


def test_screen_changing_action_always_routes_to_capture():
    assert (
        route_after_execution(
            worker_state(transition={"transition_request": {"action": "click_marker"}})
        )
        == "capture"
    )
    assert route_after_execution(worker_state()) == "reasoning"


def test_loading_card_screen_routes_from_reasoning_to_recapture():
    state = worker_state(
        decision={
            "pending_action": None,
            "job_card_selection_trace": {"reason": "screen_loading"},
        }
    )

    assert route_after_reasoning(state) == "capture"
    assert (
        route_after_reasoning(worker_state(decision={"pending_action": object()}))
        == "execution"
    )


def test_capture_screen_assigns_run_scoped_incrementing_observation_id(monkeypatch):
    captures = iter(["screen-1.png", "screen-2.png", "screen-retry-1.png"])

    class FakePerception:
        last_capture_quality = {}

        def capture_usable_screen(self):
            return next(captures)

    state = worker_state(
        request={"worker_run_id": "worker-test"},
        observation={
            "observation_sequence": 0,
            "current_url": "https://example.com",
            "current_url_stale": False,
        },
    )
    runtime = node_runtime(_FakeVisionRuntime(FakePerception()))

    first = apply_update(
        state,
        worker_observation.capture_node(state, runtime),
    )
    second = apply_update(
        first,
        worker_observation.capture_node(first, runtime),
    )
    next_run_state = worker_state(
        request={"worker_run_id": "worker-test-retry"},
        observation=state["observation"],
    )
    next_run_first = apply_update(
        next_run_state,
        worker_observation.capture_node(next_run_state, runtime),
    )

    assert (
        first["observation"]["observation_id"]
        == "worker-test:observation:0001"
    )
    assert first["observation"]["observation_sequence"] == 1
    assert (
        second["observation"]["observation_id"]
        == "worker-test:observation:0002"
    )
    assert second["observation"]["observation_sequence"] == 2
    assert (
        next_run_first["observation"]["observation_id"]
        == "worker-test-retry:observation:0001"
    )


def test_worker_state_factory_matches_the_declared_state_contract():
    from agent.runtime.worker_contracts import WorkerState
    from agent.runtime.worker_contracts import create_worker_state

    state = create_worker_state("테스트 목표")

    assert set(state).issubset(WorkerState.__annotations__)
    assert "worker_attempt_index" not in state
    assert state["request"]["goal"] == "테스트 목표"


def test_execution_records_one_complete_action_event(monkeypatch):
    calls: list[str] = []

    def fake_dispatch(action_name, args, get_bbox, **_kwargs):
        calls.append(action_name)
        get_bbox(args["marker_id"])
        return {"action": action_name, "status": "success", "result": "clicked"}

    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_ui_action",
        fake_dispatch,
    )
    request = _request(
        "llm",
        [
            {"name": "click_marker", "args": {"marker_id": 1}, "id": "click"},
        ],
    )
    result = _run_execution(
        _execution_state(
            request,
            current_markers=[{"id": 1, "bbox": [0, 0, 10, 10], "text": "공고"}],
        )
    )

    assert calls == ["click_marker"]
    action_results = action_event_results(result["transition"]["action_events"])
    assert [item["status"] for item in action_results] == ["success"]
    assert result["transition"]["transition_request"]["action"] == "click_marker"
    assert (
        result["transition"]["transition_request"]["before_observation_id"]
        == "worker-test:observation:0003"
    )
    event = result["transition"]["action_events"][0]
    assert event["observation_id"] == "worker-test:observation:0003"
    assert event["recipe_step"].action == "click_marker"
    assert (
        event["recipe_step"].before_state["observation_id"]
        == "worker-test:observation:0003"
    )
    assert (
        event["feedback_episode"].observation.before["observation_id"]
        == "worker-test:observation:0003"
    )


def test_repeated_no_effect_marker_click_counts_as_error(monkeypatch):
    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_ui_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("효과가 없던 동일 마커를 다시 클릭하면 안 됩니다.")
        ),
    )
    request = _request(
        "llm",
        [
            {
                "name": "click_marker",
                "args": {"marker_id": 1},
                "id": "click",
            }
        ],
    )
    state = _execution_state(
        request,
        current_markers=[
            {"id": 1, "bbox": [0, 0, 10, 10], "text": "공고"},
        ],
    )
    state["transition"]["action_events"] = [
        {
            "seq": 0,
            "result": {"action": "click_marker", "status": "success"},
            "transition": {
                "action": "click_marker",
                "status": "unknown",
                "reason": "no_screen_change",
                "step": {"args": {"marker_id": 1}},
            },
        }
    ]

    result = _run_execution(state)
    action_result = action_event_results(
        result["transition"]["action_events"]
    )[-1]

    assert action_result["status"] == "skipped"
    assert action_result["reason"] == "same_screen_no_effect_action_blocked"
    assert result["transition"]["error_count"] == 1


def test_execution_accumulates_detail_field_evidence(monkeypatch):
    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_ui_action",
        lambda *_args, **_kwargs: {
            "action": "scroll",
            "status": "success",
        },
    )
    request = _request(
        "llm",
        [
            {
                "name": "scroll",
                "args": {
                    "direction": "down",
                    "page_role": "job_detail",
                    "observed_fields": {"requirements": "Python"},
                },
                "id": "scroll",
            }
        ],
    )

    result = _run_execution(
        _execution_state(
            request,
            current_url="https://example.com/jobs/1",
        )
    )

    coverage = result["collection"]["job_detail_coverage"]
    assert coverage["field_evidence"]["requirements"] == "Python"
    assert coverage["field_evidence"]["url"] == ("https://example.com/jobs/1")


def test_successful_execution_resets_consecutive_error_count(monkeypatch):
    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_ui_action",
        lambda *_args, **_kwargs: {
            "action": "scroll",
            "status": "success",
        },
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
    state = _execution_state(request)
    state["transition"]["error_count"] = 2

    result = _run_execution(state)

    assert result["transition"]["error_count"] == 0


def test_reflex_transition_executes_input_and_enter_without_recapture(
    monkeypatch,
):
    calls: list[str] = []

    def fake_dispatch(action_name, args, get_bbox, **_kwargs):
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

    result = _run_execution(
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
    assert len(result["transition"]["action_events"]) == 2
    assert result["transition"]["transition_request"]["action"] == "press_key"
    assert result["transition"]["transition_request"]["transition_actions"] == [
        "type_in_marker",
        "press_key",
    ]
    assert (
        result["transition"]["transition_request"]["before_page_role"]
        == "search_overlay"
    )
    assert (
        result["transition"]["transition_request"]["expected_after_state"][
            "url_template"
        ]
        == "example.com/jobs"
    )


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
    result = _run_execution(
        _execution_state(
            request,
            current_markers=[
                {"id": 1, "bbox": [0, 0, 10, 10], "text": "공고"},
            ],
        )
    )

    action_results = action_event_results(result["transition"]["action_events"])
    assert len(action_results) == 1
    assert action_results[0]["status"] == "error"
    assert action_results[0]["error"] == "physical input failed"
    assert result["transition"]["error_count"] == 1


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

    result = _run_execution(
        _execution_state(
            request,
            current_markers=[
                {"id": 1, "bbox": [0, 0, 10, 10], "text": "공고"},
            ],
        )
    )

    assert (
        action_event_results(result["transition"]["action_events"])[0]["status"]
        == "error"
    )
    assert result["transition"]["error_count"] == 1
    assert not result["transition"]["transition_request"]


def test_returned_state_error_is_recorded_as_failure(monkeypatch):
    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_state_action",
        lambda *args, **kwargs: StateActionOutcome(
            result={
                "action": "set_job_card_queue",
                "status": "error",
                "result": "invalid payload",
            },
            job_captures=[],
        ),
    )
    request = _request(
        "llm",
        [
            {
                "name": "set_job_card_queue",
                "args": {"cards": []},
                "id": "update",
            }
        ],
    )

    result = _run_execution(
        _execution_state(
            request,
            job_captures=[
                JobCapture(
                    url="https://example.com/jobs/original",
                    raw_ocr_text="원래 회사 개발자",
                )
            ],
        )
    )

    action_result = action_event_results(result["transition"]["action_events"])[0]
    assert action_result["status"] == "error"
    assert action_result["error"] == "invalid payload"
    assert result["transition"]["error_count"] == 1
    assert result["collection"]["job_captures"][0].raw_ocr_text == "원래 회사 개발자"


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
            job_captures=[],
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
    result = _run_execution(_execution_state(request))

    follow_up = result["decision"]["pending_action"]
    assert follow_up.source == "job_card_queue"
    assert follow_up.tool_calls[0].name == "click_marker"
    assert follow_up.tool_calls[0].args["marker_id"] == 4
    assert result["collection"]["job_card_queue"] == [queued_card]


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
            job_captures=[],
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
    result = _run_execution(
        _execution_state(
            request,
            collection_intent=CollectionIntent(target_count=2),
        )
    )

    assert result["lifecycle"]["is_finished"] is True
    assert result["decision"]["pending_action"] is None
    assert result["collection"]["job_card_queue"] == existing_cards
    assert (
        action_event_results(result["transition"]["action_events"])[0]["resolved_count"]
        == 2
    )


def test_graph_custom_event_is_shared_by_metrics_and_sse():
    from agent.observability.run_context import run_context
    from agent.observability.graph_events import forward_graph_event

    events = []
    with run_context(run_id="graph-event-test", event_sink=events.append) as (
        context,
        _created,
    ):
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
    assert [event.event for event in graph_events] == [
        "graph_step_started",
        "graph_step_finished",
    ]
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
        def stream(self, state, config=None, context=None, stream_mode=None):
            assert stream_mode == ["values", "custom"]
            assert context.vision is vision_runtime
            yield ("custom", {"event": "graph_step_started", "stage": "capture"})
            yield (
                "values",
                {
                    **state,
                    "lifecycle": {
                        **state["lifecycle"],
                        "is_finished": True,
                    },
                },
            )

    monkeypatch.setattr(
        worker_execution_service, "forward_graph_event", forwarded.append
    )
    vision_runtime = _FakeVisionRuntime()
    state, limited = worker_execution_service.run_graph_with_last_state(
        FakeGraph(),
        worker_state(),
        60,
        worker_runtime=vision_runtime,
    )

    assert forwarded == [{"event": "graph_step_started", "stage": "capture"}]
    assert state["lifecycle"]["is_finished"] is True
    assert limited is False

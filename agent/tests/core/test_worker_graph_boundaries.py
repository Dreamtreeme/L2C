import time

import pytest

from agent.graph import (
    worker_action_guard,
    worker_execution,
    worker_execution_dispatch,
    worker_observation,
    worker_selection,
)
from agent.runtime.worker_contracts import action_event_results, build_action_request
from agent.graph.workflow import (
    route_after_execution,
    route_after_start,
    route_after_reflex,
    route_after_selection,
)
from agent.graph.worker_execution_dispatch import StateActionOutcome
from agent.tests.worker_test_support import (
    apply_update,
    node_runtime,
    worker_data_services,
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

    click_group = build_action_request(
        "reflex",
        "검색어 입력 후 검색 버튼 클릭",
        calls,
        metadata={"execution_unit": "recipe_transition"},
    )
    assert [call.name for call in click_group.tool_calls] == [
        "type_in_marker",
        "click_marker",
    ]

    with pytest.raises(ValueError):
        build_action_request(
            "reflex",
            "두 클릭 사이에는 화면 확인이 필요함",
            [calls[1], calls[1]],
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

    from agent.graph import worker_selection

    active_recipe_state = worker_state(
        observation=observed["observation"],
        transition={
            "transition_result": {"status": "ready", "action": "type_in_marker"}
        },
        replay={
            "replay_session": {
                "recipe_key": "recipe-search-set",
                "current_transition_index": 1,
                "transition_count": 2,
            }
        },
    )
    selected = apply_update(
        active_recipe_state,
        worker_selection.selection_node(active_recipe_state, node_runtime()),
    )
    assert route_after_selection(selected) == "reflex"


def test_duplicate_detail_that_completes_target_ends_after_selection():
    from agent.graph import worker_selection

    state = worker_state(
        request={
            "collection_intent": CollectionIntent(
                site="wanted",
                search_keyword="iOS 개발자",
                target_count=1,
            )
        },
        observation={
            "current_url": "https://www.wanted.co.kr/wd/118",
            "ocr_complete": True,
            "observation_id": "observation:detail",
            "current_screenshot": "detail.png",
        },
        collection={
            "job_card_queue": [
                {
                    "queue_id": "card-1",
                    "status": "active",
                    "title": "iOS 개발자",
                }
            ]
        },
    )
    runtime = node_runtime(
        data=worker_data_services(
            find_existing_job_url=lambda _url, _jobs: {
                "matched": True,
                "job_id": 118,
                "source": "database",
            }
        )
    )

    updated = apply_update(state, worker_selection.selection_node(state, runtime))

    assert updated["lifecycle"]["is_finished"] is True
    assert route_after_selection(updated) == "end"


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


def test_worker_routes_screen_changes_to_recapture():
    assert (
        route_after_execution(
            worker_state(transition={"transition_request": {"action": "click_marker"}})
        )
        == "capture"
    )
    assert route_after_execution(worker_state()) == "reasoning"


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

    assert first["observation"]["observation_id"] == "worker-test:observation:0001"
    assert first["observation"]["observation_sequence"] == 1
    assert second["observation"]["observation_id"] == "worker-test:observation:0002"
    assert second["observation"]["observation_sequence"] == 2
    assert (
        next_run_first["observation"]["observation_id"]
        == "worker-test-retry:observation:0001"
    )


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
    state = _execution_state(
        request,
        current_markers=[{"id": 1, "bbox": [0, 0, 10, 10], "text": "공고"}],
    )
    state["transition"]["error_count"] = 2
    result = _run_execution(state)

    assert calls == ["click_marker"]
    action_results = action_event_results(result["transition"]["action_events"])
    assert [item["status"] for item in action_results] == ["success"]
    assert result["transition"]["transition_request"]["action"] == "click_marker"
    assert (
        result["transition"]["transition_request"]["before_observation_id"]
        == "worker-test:observation:0003"
    )
    event = result["transition"]["action_events"][0]
    assert event.observation_id == "worker-test:observation:0003"
    assert event.candidate_action.action == "click_marker"
    assert event.before_checkpoint.observation_id == "worker-test:observation:0003"
    assert result["transition"]["error_count"] == 0


def test_focus_marker_clicks_component_without_requesting_screen_transition():
    focused_bboxes: list[list[int]] = []

    class FocusActionTools:
        def focus_marker(self, bbox):
            focused_bboxes.append(bbox)
            return {
                "action": "focus_marker",
                "status": "success",
                "result": "focused",
            }

    class FocusVisionRuntime(_FakeVisionRuntime):
        def get_action_tools(self):
            return FocusActionTools()

    request = _request(
        "llm",
        [
            {
                "name": "focus_marker",
                "args": {
                    "marker_id": 7,
                    "target_label": "검색 결과 목록",
                    "target_component": "scroll_panel",
                },
                "id": "focus",
            }
        ],
    )
    state = _execution_state(
        request,
        current_markers=[
            {
                "id": 7,
                "bbox": [10, 20, 110, 220],
                "text": "검색 결과 목록",
            }
        ],
    )

    update = worker_execution.execution_node(
        state,
        node_runtime(FocusVisionRuntime()),
    )
    result = apply_update(state, update)
    action_result = action_event_results(
        result["transition"]["action_events"]
    )[0]

    assert focused_bboxes == [[10, 20, 110, 220]]
    assert action_result["action"] == "focus_marker"
    assert action_result["screen_change_expected"] is False
    assert action_result["target"]["marker_id"] == 7
    assert result["transition"].get("transition_request") is None
    assert route_after_execution(result) == "reasoning"


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
            "result": {
                "action": "click_marker",
                "status": "success",
                "args": {"marker_id": 1},
            },
            "transition": {
                "seq": 0,
                "before": {"observation_id": "observation:0"},
                "actions": [
                    {
                        "source_seq": 0,
                        "action": "click_marker",
                    }
                ],
                "after": {"observation_id": "observation:1"},
                "evidence": {
                    "status": "unknown",
                    "reason": "no_screen_change",
                },
            },
        }
    ]

    result = _run_execution(state)
    action_result = action_event_results(result["transition"]["action_events"])[-1]

    assert action_result["status"] == "skipped"
    assert action_result["reason"] == "same_screen_no_effect_action_blocked"
    assert result["transition"]["error_count"] == 1

    close_request = _request(
        "llm",
        [
            {
                "name": "close_current_tab",
                "args": {"risk_level": "safe_navigation"},
                "id": "close-tab",
            }
        ],
    )
    close_result = _run_execution(_execution_state(close_request))
    close_action = action_event_results(
        close_result["transition"]["action_events"]
    )[-1]

    assert close_action["status"] == "skipped"
    assert close_action["reason"] == "close_tab_requires_failed_go_back"

    monkeypatch.setattr(
        worker_action_guard,
        "latest_no_effect_transition",
        lambda _state: {"action": "go_back"},
    )
    monkeypatch.setattr(
        worker_execution_dispatch,
        "dispatch_ui_action",
        lambda action_name, *_args, **_kwargs: {
            "action": action_name,
            "status": "success",
            "result": "closed",
        },
    )
    close_after_back = _run_execution(_execution_state(close_request))
    allowed_close_action = action_event_results(
        close_after_back["transition"]["action_events"]
    )[-1]

    assert allowed_close_action["status"] == "success"

    stopped = worker_selection.selection_node(
        worker_state(transition={"no_effect_count": 3}),
        node_runtime(),
    )
    assert stopped["decision"]["pending_action"].source == "transition_policy"
    assert stopped["decision"]["pending_action"].tool_calls[0].name == "finish_task"


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
        result["transition"]["transition_request"]["expected_after_state"].url_template
        == "example.com/jobs"
    )


def test_failed_ui_dispatch_is_recorded_once_without_screen_transition(monkeypatch):
    def fail_dispatch(*args, **kwargs):
        raise RuntimeError("physical input failed")

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
    dispatchers = [
        fail_dispatch,
        lambda *args, **kwargs: {
            "action": "click_marker",
            "status": "error",
            "error": "physical input failed",
        },
    ]
    for dispatch in dispatchers:
        monkeypatch.setattr(
            worker_execution_dispatch,
            "dispatch_ui_action",
            dispatch,
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
            state_update={
                "collection": {
                    "job_card_queue": [queued_card],
                    "job_results_memory": {"url": "https://example.com/jobs"},
                    "job_results_availability": {},
                }
            },
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
            state_update={
                "collection": {
                    "job_card_queue": existing_cards,
                    "job_results_memory": {"url": "https://example.com/jobs"},
                    "job_results_availability": {},
                }
            },
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
        context.record_llm_call(
            "vision_reasoning",
            "langchain",
            "gemini-3.6-flash",
            {"input_tokens": 1000, "output_tokens": 100},
            0.5,
        )
        budget = context.llm_budget_usage({"vision_reasoning"})
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
    assert budget["call_count"] == 1
    assert budget["estimated_cost_usd"] > 0
    assert snapshot["llm"]["totals"]["total_tokens"] == 1100
    assert snapshot["llm"]["by_model"]["gemini-3.6-flash"]["total_tokens"] == 1100


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
        data_services=worker_data_services(),
    )

    assert forwarded == [{"event": "graph_step_started", "stage": "capture"}]
    assert state["lifecycle"]["is_finished"] is True
    assert limited is False

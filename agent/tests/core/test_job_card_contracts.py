"""채용공고 카드 선택과 재생 계약 테스트."""

from pathlib import Path
from types import SimpleNamespace

from agent.graph import worker_reasoning
from agent.graph.worker_reasoning_prompt import build_reasoning_messages
from agent.runtime.worker_contracts import build_action_event
from agent.tests.worker_test_support import node_runtime, worker_state
from shared.schema.collection_intent import CollectionIntent


def _state(image_path: Path) -> dict:
    return worker_state(
        observation={
            "current_page_role": "search",
            "current_url": "https://example.com/search?q=ios",
            "marked_image": str(image_path),
            "current_markers": [
                {
                    "id": 10,
                    "type": "text",
                    "text": "iOS 개발자",
                    "bbox": [10, 10, 80, 30],
                },
                {
                    "id": 20,
                    "type": "text",
                    "text": "백엔드 개발자",
                    "bbox": [10, 50, 90, 70],
                },
            ],
        },
        request={
            "collection_intent": CollectionIntent(
                search_keyword="iOS 개발자",
                target_count=2,
            )
        },
    )


def test_selector_builds_queue_only_from_visible_markers(tmp_path, monkeypatch):
    from PIL import Image

    from agent.runtime import job_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.JobCardSelection(
                is_job_results_page=True,
                cards=[
                    selector.VisibleJobCard(
                        marker_id=10,
                        title="iOS 개발자",
                        company="회사 A",
                    ),
                    selector.VisibleJobCard(
                        marker_id=999,
                        title="화면에 없는 공고",
                        company="회사 B",
                    ),
                ],
            )

    monkeypatch.setattr(
        selector,
        "_get_job_card_selector_model",
        lambda: FakeModel(),
    )
    request, trace = selector.select_job_cards(_state(image_path))

    assert trace["reason"] == "cards_selected"
    assert trace["marker_ids"] == [10]
    assert request.tool_calls[0].name == "set_job_card_queue"
    assert len(request.tool_calls[0].args["cards"]) == 1


def test_selector_does_not_refill_an_existing_queue(tmp_path, monkeypatch):
    from agent.runtime import job_card_selector as selector

    state = _state(tmp_path / "unused.jpg")
    state["collection"]["job_card_queue"] = [
        {"queue_id": "card-1", "status": "done", "title": "iOS 개발자"}
    ]
    monkeypatch.setattr(
        selector,
        "_get_job_card_selector_model",
        lambda: (_ for _ in ()).throw(
            AssertionError("기존 큐가 있으면 카드 선택 모델을 다시 호출하면 안 됩니다.")
        ),
    )

    request, trace = selector.select_job_cards(state)

    assert request is None
    assert trace == {"attempted": False, "reason": "selector_not_applicable"}


def test_queue_click_requires_an_exact_queue_id():
    from agent.runtime.job_card_queue import (
        activate_job_card,
        job_card_click_matches_queue,
    )

    queue = [
        {
            "queue_id": "card-1",
            "status": "pending",
            "title": "iOS 개발자",
            "source_marker_id": 10,
        }
    ]

    assert not job_card_click_matches_queue(
        queue,
        {"marker_id": 10, "target_component": "job_card_title"},
    )
    assert not job_card_click_matches_queue(queue, {"queue_id": "card-2"})
    assert job_card_click_matches_queue(queue, {"queue_id": "card-1"})
    assert activate_job_card(queue, {"marker_id": 10}) == queue
    assert activate_job_card(queue, {"queue_id": "card-1"})[0]["status"] == "active"


def test_general_reasoning_converts_model_tool_call(monkeypatch):
    from agent.runtime.tool_schema import model_action_tool_schema
    from agent.runtime.worker_contracts import action_request_from_model_response

    scroll_schema = model_action_tool_schema("scroll")
    scroll_properties = scroll_schema["function"]["parameters"]["properties"]
    assert "amount" not in scroll_properties
    assert scroll_properties["scroll_distance"]["enum"] == ["small", "page"]
    input_schema = model_action_tool_schema("type_in_marker")
    slot_options = input_schema["function"]["parameters"]["properties"][
        "slot_name"
    ]["anyOf"]
    assert slot_options[0]["const"] == "search_keyword"
    normalized_scroll = action_request_from_model_response(
        SimpleNamespace(
            content="조금 아래를 읽습니다.",
            tool_calls=[
                {
                    "id": "scroll-contract",
                    "name": "scroll",
                    "args": {
                        "direction": "down",
                        "scroll_distance": "small",
                    },
                }
            ],
        ),
        allowed_tool_names=("scroll",),
    )
    assert normalized_scroll.tool_calls[0].args["amount"] == "small"

    model_tiers = []
    monkeypatch.setattr(
        worker_reasoning,
        "select_job_cards",
        lambda _state: (None, {"attempted": False}),
    )
    monkeypatch.setattr(
        worker_reasoning,
        "build_reasoning_messages",
        lambda *_args: ["현재 화면"],
    )
    monkeypatch.setattr(
        worker_reasoning,
        "_get_ui_llm_with_tools",
        lambda _runtime, _state, *, tier: model_tiers.append(tier) or object(),
    )
    monkeypatch.setattr(
        worker_reasoning,
        "invoke_with_metrics",
        lambda *_args: SimpleNamespace(
            content="아래 내용을 확인합니다.",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "scroll",
                    "args": {"direction": "down"},
                }
            ],
        ),
    )

    result = worker_reasoning.reasoning_node(
        worker_state(observation={"current_page_role": "search"}),
        node_runtime(),
    )

    assert result["decision"]["pending_action"].tool_calls[0].name == "scroll"
    assert result["decision"]["reasoning_call_count"] == 1
    assert result["replay"]["reflex_trace"]["source"] == "reasoning"
    assert model_tiers == ["lightweight"]
    assert "set_job_card_queue" not in worker_reasoning._reasoning_tool_names(
        worker_state(observation={"current_page_role": "search"})
    )
    assert "finish_detail_reading" in worker_reasoning._reasoning_tool_names(
        worker_state(observation={"current_page_role": "job_detail"})
    )

    recovery = worker_reasoning.reasoning_node(
        worker_state(
            observation={"current_page_role": "job_detail"},
            transition={"transition_result": {"status": "unknown"}},
        ),
        node_runtime(),
    )
    assert recovery["decision"]["pending_action"].tool_calls[0].name == "scroll"
    assert model_tiers[-1] == "primary"

    exhausted = worker_reasoning.reasoning_node(
        worker_state(decision={"reasoning_call_count": 16}),
        node_runtime(),
    )
    assert exhausted["decision"]["pending_action"].source == "reasoning_policy"
    assert exhausted["decision"]["pending_action"].tool_calls[0].name == "finish_task"


def test_reasoning_prompt_reuses_already_compacted_queue_action_args():
    state = worker_state(
        observation={
            "current_page_role": "detail",
            "current_url": "https://www.wanted.co.kr/wd/370091",
        },
        transition={
            "action_events": [
                build_action_event(
                    0,
                    {
                        "action": "set_job_card_queue",
                        "status": "success",
                        "args": {"cards": 1, "titles": ["SRE"]},
                    },
                )
            ]
        },
    )

    messages = build_reasoning_messages(state, "")

    assert len(messages) == 2
    assert "최근 행동 요약" in str(messages[1].content)


def test_resolved_card_count_only_includes_collected_or_database_confirmed():
    from agent.runtime.job_card_queue import (
        job_card_queue_scope_complete,
        resolved_job_card_count,
    )

    queue = [
        {"status": "done"},
        {"status": "skipped", "job_id": 7},
        {"status": "skipped", "skip_reason": "invalid_marker"},
    ]

    assert resolved_job_card_count(queue) == 2
    assert job_card_queue_scope_complete(
        queue,
        count_mode="explicit",
        target_count=2,
    )

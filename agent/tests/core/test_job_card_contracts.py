"""채용공고 카드 선택과 재생 계약 테스트."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.graph import worker_reasoning, worker_selection
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
    state = _state(image_path)
    state["request"]["collection_intent"] = CollectionIntent(
        original_query="신입 iOS 개발자 공고를 정리해줘",
        search_keyword="iOS 개발자",
        target_count=1,
        filters={"experience": "신입"},
    )
    state["observation"]["current_markers"].append(
        {
            "id": 11,
            "type": "text",
            "text": "iOS 앱 개발자",
            "bbox": [10, 35, 90, 48],
        }
    )

    captured = {}

    class FakeModel:
        def invoke(self, inputs, config=None):
            captured["messages"] = inputs
            return selector.JobCardSelection(
                is_job_results_page=True,
                cards=[
                    selector.EvaluatedJobCard(
                        marker_id=10,
                        title="iOS 개발자",
                        company="회사 A",
                        filter_status="match",
                        filter_evidence="신입",
                    ),
                    selector.EvaluatedJobCard(
                        marker_id=11,
                        title="iOS 앱 개발자",
                        company="회사 B",
                        filter_status="conflict",
                        filter_evidence="경력 3년 이상",
                    ),
                    selector.EvaluatedJobCard(
                        marker_id=999,
                        title="화면에 없는 공고",
                        company="회사 C",
                        filter_status="match",
                        filter_evidence="신입",
                    ),
                ],
            )

    monkeypatch.setattr(
        selector,
        "_get_job_card_selector_model",
        lambda: FakeModel(),
    )
    request, trace = selector.select_job_cards(state)

    assert trace["reason"] == "cards_selected"
    assert trace["marker_ids"] == [10]
    assert trace["conflict_count"] == 1
    assert request.tool_calls[0].name == "set_job_card_queue"
    assert len(request.tool_calls[0].args["cards"]) == 1
    payload = json.loads(captured["messages"][-1].content[0]["text"])
    assert payload["original_query"] == "신입 iOS 개발자 공고를 정리해줘"
    assert payload["confirmed_filters"]["experience"] == "신입"


def test_selector_does_not_replace_an_unresolved_queue(tmp_path, monkeypatch):
    from agent.runtime import job_card_selector as selector

    state = _state(tmp_path / "unused.jpg")
    state["collection"]["job_card_queue"] = [
        {"queue_id": "card-1", "status": "pending", "title": "iOS 개발자"}
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


def test_selector_opens_full_results_before_preview_cards(tmp_path, monkeypatch):
    from PIL import Image

    from agent.runtime import job_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    state = _state(image_path)
    state["observation"]["current_markers"].append(
        {
            "id": 18,
            "type": "text",
            "text": "포지션 전체보기",
            "bbox": [90, 10, 180, 30],
        }
    )

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.JobCardSelection(
                is_job_results_page=False,
                open_full_results_marker_id=18,
                open_full_results_label="포지션 전체보기",
                cards=[],
            )

    monkeypatch.setattr(selector, "_get_job_card_selector_model", lambda: FakeModel())

    request, trace = selector.select_job_cards(state)

    assert trace["reason"] == "open_full_results"
    assert request.tool_calls[0].name == "click_marker"
    assert request.tool_calls[0].args["marker_id"] == 18
    assert request.tool_calls[0].args["target_component"] == "full_job_results"


def test_selector_refills_a_resolved_queue_with_only_new_cards(tmp_path, monkeypatch):
    from PIL import Image

    from agent.runtime import job_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    state = _state(image_path)
    state["collection"]["job_card_queue"] = [
        {
            "queue_id": "card-1",
            "status": "done",
            "title": "iOS 개발자",
            "company": "회사 A",
        }
    ]

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.JobCardSelection(
                is_job_results_page=True,
                cards=[
                    selector.EvaluatedJobCard(
                        marker_id=10,
                        title="iOS 개발자",
                        company="회사 A",
                        filter_status="match",
                    ),
                    selector.EvaluatedJobCard(
                        marker_id=20,
                        title="백엔드 개발자",
                        company="회사 B",
                        filter_status="unknown",
                    ),
                ],
            )

    monkeypatch.setattr(selector, "_get_job_card_selector_model", lambda: FakeModel())

    request, trace = selector.select_job_cards(state)

    assert trace["reason"] == "cards_selected"
    assert trace["marker_ids"] == [20]
    assert request.tool_calls[0].args["cards"] == [
        {"marker_id": 20, "title": "백엔드 개발자", "company": "회사 B"}
    ]


def test_selector_continues_when_result_list_has_no_related_card(
    tmp_path,
    monkeypatch,
):
    from PIL import Image

    from agent.runtime import job_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.JobCardSelection(
                is_job_results_page=True,
                cards=[],
            )

    monkeypatch.setattr(
        selector,
        "_get_job_card_selector_model",
        lambda: FakeModel(),
    )

    request, trace = selector.select_job_cards(_state(image_path))

    assert trace["reason"] == "no_valid_card_continue"
    assert request.source == "card_selector"
    assert request.tool_calls[0].name == "scroll"
    assert request.tool_calls[0].args["amount"] == "page"


def test_visible_all_detail_cannot_finish_with_an_active_card():
    visible_scope_with_active_card = worker_state(
        request={
            "collection_intent": CollectionIntent(
                count_mode="visible_all",
            )
        },
        observation={"current_page_role": "job_detail"},
        collection={
            "job_card_queue": [
                {
                    "queue_id": "card-1",
                    "status": "active",
                    "title": "신입 백엔드 엔지니어",
                }
            ]
        },
    )

    assert "finish_task" not in worker_reasoning._reasoning_tool_names(
        visible_scope_with_active_card
    )


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


def test_model_action_schema_normalizes_scroll_contract():
    from agent.runtime.tool_schema import model_action_tool_schema
    from agent.runtime.worker_contracts import action_request_from_model_response

    scroll_schema = model_action_tool_schema("scroll")
    scroll_properties = scroll_schema["function"]["parameters"]["properties"]
    assert "amount" not in scroll_properties
    assert "replay_mode" not in scroll_properties
    assert scroll_properties["scroll_distance"]["enum"] == ["small", "page"]
    input_schema = model_action_tool_schema("type_in_marker")
    assert "replay_mode" not in input_schema["function"]["parameters"]["properties"]
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


def test_general_reasoning_converts_model_tool_call(monkeypatch):
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
    assert "review_job_detail" not in worker_reasoning._reasoning_tool_names(
        worker_state(observation={"current_page_role": "job_detail"})
    )
    incomplete_detail = worker_state(
        request={
            "collection_intent": CollectionIntent(
                target_count=2,
                count_mode="explicit",
            )
        },
        observation={"current_page_role": "job_detail"},
    )
    assert "review_job_detail" not in worker_reasoning._reasoning_tool_names(
        incomplete_detail
    )
    assert "finish_task" not in worker_reasoning._reasoning_tool_names(
        incomplete_detail
    )
    exhausted_detail = worker_state(
        request={
            "collection_intent": CollectionIntent(
                target_count=2,
                count_mode="explicit",
            )
        },
        observation={"current_page_role": "job_detail"},
        collection={
            "job_results_availability": {
                "available_job_count": 0,
                "count_evidence": "검색 결과 0건",
            }
        },
    )
    assert "review_job_detail" not in worker_reasoning._reasoning_tool_names(
        exhausted_detail
    )
    assert "finish_task" in worker_reasoning._reasoning_tool_names(
        exhausted_detail
    )
    review_due_detail = worker_state(
        request={
            "collection_intent": CollectionIntent(
                target_count=2,
                count_mode="explicit",
            )
        },
        observation={"current_page_role": "job_detail"},
        collection={
            "job_detail_buffer": {
                "url": "https://example.com/jobs/current",
                "lines": [{"text": "주요업무"}],
                "stats": {"screen_count": 1},
            }
        },
    )
    assert "review_job_detail" not in worker_reasoning._reasoning_tool_names(
        review_due_detail
    )
    assert "finish_task" not in worker_reasoning._reasoning_tool_names(
        review_due_detail
    )
    direct_review = worker_selection._select_detail_review(review_due_detail)
    assert direct_review is not None
    assert direct_review["decision"]["pending_action"].source == "state_contract"
    assert (
        direct_review["decision"]["pending_action"].tool_calls[0].name
        == "review_job_detail"
    )
    from shared.schema.jd_schema import JobDraft

    reviewed_fingerprint = JobDraft(
        url="https://example.com/jobs/current",
        raw_ocr_text="1. 주요업무",
        required_fields=(
            review_due_detail["request"]["collection_intent"].required_fields
        ),
    ).fingerprint()
    review_due_detail["collection"]["last_job_review"] = SimpleNamespace(
        draft_fingerprint=reviewed_fingerprint,
    )
    assert worker_selection._select_detail_review(review_due_detail) is None
    assert "scroll" in worker_reasoning._reasoning_tool_names(review_due_detail)
    assert "finish_task" not in worker_reasoning._reasoning_tool_names(
        review_due_detail
    )
    review_due_detail["collection"]["job_detail_buffer"]["lines"].append(
        {"text": "자격요건"}
    )
    changed_review = worker_selection._select_detail_review(review_due_detail)
    assert changed_review is not None
    assert changed_review["decision"]["pending_action"].source == "state_contract"
    assert (
        changed_review["decision"]["pending_action"].tool_calls[0].name
        == "review_job_detail"
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


def test_reasoning_rejects_icon_marker_as_text_input_target():
    from agent.runtime.worker_contracts import action_request_from_model_response

    completed = action_request_from_model_response(
        SimpleNamespace(
            content="검색어를 입력합니다.",
            tool_calls=[
                {
                    "name": "type_in_marker",
                    "args": {
                        "marker_id": 18,
                        "text": "데이터 엔지니어",
                        "slot_name": "search_keyword",
                    },
                    "id": "type-search",
                }
            ],
        ),
        allowed_tool_names=("type_in_marker", "press_key"),
    )
    state = worker_state(
        observation={
            "current_markers": [
                {
                    "id": 17,
                    "type": "text",
                    "text": "JOB 검색",
                    "bbox": [200, 100, 500, 150],
                },
                {
                    "id": 18,
                    "type": "icon",
                    "text": "icon",
                    "bbox": [510, 100, 560, 150],
                },
            ]
        }
    )
    assert [call.name for call in completed.tool_calls] == [
        "type_in_marker",
        "press_key",
    ]
    with pytest.raises(
        ValueError,
        match=r"대상 \[18\]은 아이콘 마커.*허용된 텍스트 마커 ID: \[17\]",
    ):
        worker_reasoning._validate_reasoning_target_markers(state, completed)


def test_reasoning_primary_retry_receives_lightweight_validation_error(monkeypatch):
    warnings = []

    def invoke(_state, _runtime, loop_warning, *, tier):
        warnings.append((tier, loop_warning))
        if tier == "lightweight":
            raise ValueError("type_in_marker 대상 [18]은 아이콘 마커입니다.")
        return worker_reasoning.build_action_request(
            "llm",
            "검색어를 입력합니다.",
            [{"name": "press_key", "args": {"key": "Enter"}}],
        )

    monkeypatch.setattr(worker_reasoning, "_invoke_reasoning_model", invoke)

    request, call_count, tier, stop_reason, _ = (
        worker_reasoning._choose_reasoning_action(
            worker_state(),
            node_runtime(),
            "기존 경고",
            initial_tier="lightweight",
            call_count=0,
        )
    )

    assert request is not None
    assert request.tool_calls[0].name == "press_key"
    assert call_count == 2
    assert tier == "primary"
    assert stop_reason == ""
    assert warnings[0] == ("lightweight", "기존 경고")
    assert warnings[1][0] == "primary"
    assert "대상 [18]은 아이콘 마커" in warnings[1][1]


def test_detail_review_request_does_not_repeat_ui_reasoning(monkeypatch):
    tiers = []

    def invoke(_state, _runtime, loop_warning, *, tier):
        tiers.append((tier, loop_warning))
        return worker_reasoning.build_action_request(
            "llm",
            "누적 본문 검토를 요청합니다.",
            [{"name": "review_job_detail", "args": {"reason": "근거 검토"}}],
        )

    monkeypatch.setattr(worker_reasoning, "_invoke_reasoning_model", invoke)

    request, call_count, tier, stop_reason, _ = (
        worker_reasoning._choose_reasoning_action(
            worker_state(observation={"current_page_role": "job_detail"}),
            node_runtime(),
            "",
            initial_tier="lightweight",
            call_count=0,
        )
    )

    assert request is not None
    assert request.tool_calls[0].name == "review_job_detail"
    assert call_count == 1
    assert tier == "lightweight"
    assert stop_reason == ""
    assert [item[0] for item in tiers] == ["lightweight"]


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

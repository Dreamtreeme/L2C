"""채용공고 카드 선택과 재생 계약 테스트."""

from types import SimpleNamespace

from agent.graph import worker_reasoning, worker_selection
from agent.graph.worker_execution_policy import repeats_submitted_input_request
from agent.graph.worker_reasoning_prompt import build_reasoning_messages
from agent.runtime.worker_contracts import build_action_event
from agent.tests.worker_test_support import node_runtime, worker_state
from shared.schema.collection_intent import CollectionIntent


def test_search_reasoning_owns_card_queue_creation(monkeypatch):
    state = worker_state(
        request={
            "collection_intent": CollectionIntent(
                original_query="신입 iOS 개발자 공고를 정리해줘",
                search_keyword="iOS 개발자",
                target_count=1,
                filters={"experience": "신입"},
            )
        },
        observation={
            "observation_id": "search-1",
            "current_page_role": "search",
            "current_screenshot": "search.png",
            "ocr_complete": True,
            "current_markers": [
                {
                    "id": 10,
                    "type": "text",
                    "text": "iOS 개발자",
                    "bbox": [10, 10, 80, 30],
                }
            ],
        },
    )

    def invoke(_state, _runtime, _warning, *, tier):
        return worker_reasoning.build_action_request(
            "llm",
            "현재 화면의 관련 공고를 큐에 저장합니다.",
            [
                {
                    "name": "set_job_card_queue",
                    "args": {
                        "cards": [
                            {
                                "marker_id": 10,
                                "title": "iOS 개발자",
                                "company": "회사 A",
                            }
                        ]
                    },
                }
            ],
        )

    monkeypatch.setattr(worker_reasoning, "_invoke_reasoning_model", invoke)

    result = worker_reasoning.reasoning_node(state, node_runtime())

    request = result["decision"]["pending_action"]
    assert request.source == "llm"
    assert request.tool_calls[0].name == "set_job_card_queue"
    assert result["decision"]["reasoning_call_count"] == 1
    assert "set_job_card_queue" in worker_reasoning._reasoning_tool_names(state)

    from agent.runtime.job_card_queue import normalize_job_card_queue

    state["collection"]["job_card_queue"] = normalize_job_card_queue(
        request.tool_calls[0].args,
        state,
    )
    queued = worker_selection.selection_node(state, node_runtime())
    click_request = queued["decision"]["pending_action"]
    assert click_request.source == "job_card_queue"
    assert click_request.tool_calls[0].name == "click_marker"
    assert click_request.tool_calls[0].args["marker_id"] == 10


def test_unresolved_card_queue_cannot_be_replaced_or_finished():
    state = worker_state(
        observation={"current_page_role": "search"},
        collection={
            "job_card_queue": [
                {"queue_id": "card-1", "status": "pending", "title": "iOS 개발자"}
            ]
        },
    )

    tools = worker_reasoning._reasoning_tool_names(state)

    assert "set_job_card_queue" not in tools
    assert "finish_task" not in tools


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
    slot_options = input_schema["function"]["parameters"]["properties"]["slot_name"][
        "anyOf"
    ]
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
    assert normalized_scroll.tool_calls[0].args["risk_level"] == "safe_read"

    normalized_input = action_request_from_model_response(
        SimpleNamespace(
            content="검색어를 입력합니다.",
            tool_calls=[
                {
                    "id": "search-input",
                    "name": "type_in_marker",
                    "args": {
                        "marker_id": 20,
                        "text": "백엔드",
                        "slot_name": "search_keyword",
                    },
                }
            ],
        ),
        allowed_tool_names=("type_in_marker", "press_key"),
    )
    assert [call.name for call in normalized_input.tool_calls] == [
        "type_in_marker",
        "press_key",
    ]
    assert all(
        call.args["risk_level"] == "safe_navigation"
        for call in normalized_input.tool_calls
    )


def test_general_reasoning_converts_model_tool_call(monkeypatch):
    model_tiers = []
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
    marker_ready = worker_state(
        observation={
            "observation_id": "search-2",
            "current_page_role": "home",
            "current_screenshot": "search.png",
            "ocr_complete": True,
            "current_markers": [
                {
                    "id": 10,
                    "type": "text",
                    "text": "iOS 개발자",
                    "bbox": [10, 10, 80, 30],
                }
            ],
        }
    )
    assert "set_job_card_queue" in worker_reasoning._reasoning_tool_names(
        marker_ready
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
    review_due_detail = worker_state(
        request={
            "collection_intent": CollectionIntent(
                target_count=2,
                count_mode="explicit",
            )
        },
        observation={
            "current_page_role": "search",
            "current_url": "https://example.com/jobs/current",
        },
        collection={
            "job_detail_buffer": {
                "url": "https://example.com/jobs/current",
                "lines": [{"text": "주요업무"}],
                "stats": {"screen_count": 1},
            }
        },
    )
    assert "review_job_detail" in worker_reasoning._reasoning_tool_names(
        review_due_detail
    )
    assert "finish_task" not in worker_reasoning._reasoning_tool_names(
        review_due_detail
    )
    assert "scroll" in worker_reasoning._reasoning_tool_names(review_due_detail)

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
        worker_state(
            decision={
                "reasoning_call_count": 16,
                "reasoning_stage": "navigation",
                "reasoning_stage_call_count": 16,
            }
        ),
        node_runtime(),
    )
    assert exhausted["decision"]["pending_action"].source == "reasoning_policy"
    assert exhausted["decision"]["pending_action"].tool_calls[0].name == "finish_task"

    detail_after_navigation = worker_reasoning.reasoning_node(
        worker_state(
            observation={"current_page_role": "job_detail"},
            decision={
                "reasoning_call_count": 16,
                "reasoning_stage": "navigation",
                "reasoning_stage_call_count": 16,
            },
            progress={"stage": "detail"},
        ),
        node_runtime(),
    )
    assert detail_after_navigation["decision"]["pending_action"].source == "llm"
    assert detail_after_navigation["decision"]["reasoning_call_count"] == 17
    assert detail_after_navigation["decision"]["reasoning_stage"] == "detail"
    assert detail_after_navigation["decision"]["reasoning_stage_call_count"] == 1


def test_reasoning_primary_retry_receives_lightweight_validation_error(monkeypatch):
    warnings = []

    def invoke(_state, _runtime, loop_warning, *, tier):
        warnings.append((tier, loop_warning))
        if tier == "lightweight":
            raise ValueError("도구 호출 인자가 유효하지 않습니다.")
        return worker_reasoning.build_action_request(
            "llm",
            "검색어를 입력합니다.",
            [{"name": "press_key", "args": {"key": "Enter"}}],
        )

    monkeypatch.setattr(worker_reasoning, "_invoke_reasoning_model", invoke)

    usage = worker_reasoning._reasoning_usage(worker_state())
    request, tier, stop_reason = worker_reasoning._choose_reasoning_action(
        worker_state(),
        node_runtime(),
        "기존 경고",
        initial_tier="lightweight",
        usage=usage,
    )

    assert request is not None
    assert request.tool_calls[0].name == "press_key"
    assert usage.total_call_count == 2
    assert usage.stage_call_count == 2
    assert tier == "primary"
    assert stop_reason == ""
    assert warnings[0] == ("lightweight", "기존 경고")
    assert warnings[1][0] == "primary"
    assert "도구 호출 인자가 유효하지 않습니다" in warnings[1][1]


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

    detail_state = worker_state(
        observation={"current_page_role": "job_detail"},
        progress={"stage": "detail"},
    )
    usage = worker_reasoning._reasoning_usage(detail_state)
    request, tier, stop_reason = worker_reasoning._choose_reasoning_action(
        detail_state,
        node_runtime(),
        "",
        initial_tier="lightweight",
        usage=usage,
    )

    assert request is not None
    assert request.tool_calls[0].name == "review_job_detail"
    assert usage.total_call_count == 1
    assert usage.stage_call_count == 1
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


def test_search_prompt_assigns_card_selection_to_reasoning():
    state = worker_state(
        goal="백엔드 개발자 공고 1개 수집",
        observation={"current_page_role": "search"},
        transition={
            "action_events": [
                {
                    "seq": 0,
                    "result": {
                        "action": "type_in_marker",
                        "status": "success",
                    },
                    "transition": {
                        "seq": 0,
                        "before": {"observation_id": "before-search"},
                        "actions": [
                            {
                                "source_seq": 0,
                                "action": "type_in_marker",
                                "param": {
                                    "text": "백엔드 개발자",
                                    "slot_name": "search_keyword",
                                },
                            },
                            {
                                "source_seq": 1,
                                "action": "press_key",
                                "param": {"key": "enter"},
                            },
                        ],
                        "after": {"observation_id": "after-search"},
                        "evidence": {
                            "result_status": "success",
                            "status": "ready",
                        },
                    },
                }
            ],
            "transition_result": {
                "status": "unknown",
                "reason": "no_screen_change",
                "needs_ocr": False,
            },
        },
    )

    messages = build_reasoning_messages(state, "")
    prompt = str(messages[1].content)

    assert "사용자 요청과 직접 관련된 미방문 공고만 선택" in prompt
    assert "set_job_card_queue" in prompt
    assert "공고를 직접 클릭하지 말고" in prompt
    assert "이미 입력하고 제출한 검색어: 백엔드 개발자" in prompt
    assert "finish_task" in worker_reasoning._reasoning_tool_names(state)

    same_query = worker_reasoning.build_action_request(
        "llm",
        "같은 검색어 재입력",
        [
            {
                "name": "type_in_marker",
                "args": {
                    "marker_id": 20,
                    "text": "백엔드 개발자",
                    "slot_name": "search_keyword",
                },
            },
            {"name": "press_key", "args": {"key": "enter"}},
        ],
    )
    different_query = same_query.model_copy(deep=True)
    different_query.tool_calls[0].args["text"] = "서버 개발자"
    assert repeats_submitted_input_request(state, same_query)
    assert not repeats_submitted_input_request(state, different_query)


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

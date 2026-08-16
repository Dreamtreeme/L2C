import time

from agent.graph import (
    worker_selection,
    worker_transition,
)
from agent.graph.workflow import route_after_execution, route_after_selection
from agent.tests.worker_test_support import (
    apply_update,
    node_runtime,
    worker_data_services,
    worker_state,
)
from shared.schema.collection_intent import CollectionIntent
from shared.schema.jd_schema import JobCapture
from shared.schema.experience_rule_schema import ExpectedEffect


def test_completed_detail_uses_deterministic_results_navigation():
    state = worker_state(
        observation={
            "current_url": "https://www.wanted.co.kr/wd/1",
            "current_page_role": "job_detail",
            "ocr_complete": True,
        },
        transition={
            "transition_result": {
                "status": "ready",
                "action": "scroll",
                "source": "llm",
            },
        },
        collection={
            "job_captures": [
                JobCapture(
                    url="https://www.wanted.co.kr/wd/1",
                    raw_ocr_text="첫 번째 공고 상세 원문",
                )
            ],
            "job_card_queue": [
                {
                    "queue_id": "card-2",
                    "status": "pending",
                    "title": "두 번째 iOS 개발자",
                },
            ]
        },
    )

    assert route_after_execution(state) == "selection"
    back = worker_selection.selection_node(state, node_runtime())
    back_request = back["decision"]["pending_action"]
    assert back_request.source == "job_results_navigation"
    assert back_request.tool_calls[0].name == "go_back"

    failed_back = worker_state(
        observation={
            "current_url": "https://www.wanted.co.kr/wd/1",
            "current_page_role": "job_detail",
            "ocr_complete": True,
        },
        transition={
            "transition_result": {
                "status": "unknown",
                "action": "go_back",
                "reason": "no_screen_change",
                "source": "job_results_navigation",
            },
        },
        collection=state["collection"],
    )
    close = worker_selection.selection_node(failed_back, node_runtime())
    close_request = close["decision"]["pending_action"]
    assert close_request.source == "job_results_navigation"
    assert close_request.tool_calls[0].name == "close_current_tab"


def test_completed_detail_without_queue_returns_for_more_results():
    state = worker_state(
        request={"collection_intent": CollectionIntent(target_count=2)},
        observation={
            "current_url": "https://www.rallit.com/positions/example-posting",
            "current_page_role": "job_detail",
            "ocr_complete": True,
        },
        transition={
            "transition_result": {
                "status": "ready",
                "action": "scroll",
                "source": "llm",
            },
        },
        collection={
            "job_captures": [
                JobCapture(
                    url="https://www.rallit.com/positions/example-posting",
                    raw_ocr_text="첫 번째 공고 상세 원문",
                )
            ],
            "job_card_queue": [],
        },
    )

    assert route_after_execution(state) == "selection"
    selected = worker_selection.selection_node(state, node_runtime())
    request = selected["decision"]["pending_action"]
    assert request.source == "job_results_navigation"
    assert request.tool_calls[0].name == "go_back"


def test_detail_click_does_not_return_before_detail_reading_finishes():
    state = worker_state(
        observation={
            "current_url": "https://www.wanted.co.kr/wd/1",
            "current_page_role": "job_detail",
            "ocr_complete": False,
        },
        transition={
            "transition_result": {
                "status": "ready",
                "action": "click_marker",
                "source": "llm",
            },
        },
        collection={
            "job_card_queue": [
                {
                    "queue_id": "card-2",
                    "status": "pending",
                    "title": "두 번째 iOS 개발자",
                },
            ]
        },
    )

    selected = worker_selection.selection_node(state, node_runtime())

    assert selected == {}


def test_returned_results_selects_next_card_from_current_ocr():
    state = worker_state(
        observation={
            "current_url": "https://www.wanted.co.kr/search",
            "current_page_role": "search",
            "ocr_complete": True,
            "current_markers": [
                {
                    "id": 29,
                    "bbox": [300, 400, 500, 450],
                    "text": "두 번째 iOS 개발자 ~ 08/20(목)",
                    "type": "text",
                }
            ],
        },
        transition={
            "transition_result": {
                "status": "ready",
                "action": "close_current_tab",
                "source": "job_results_navigation",
            },
        },
        collection={
            "job_card_queue": [
                {
                    "queue_id": "card-2",
                    "status": "pending",
                    "title": "두 번째 iOS 개발자 - 08/20(목)",
                }
            ],
        },
    )

    selected = worker_selection.selection_node(state, node_runtime())
    request = selected["decision"]["pending_action"]
    assert request.source == "job_card_queue"
    assert request.tool_calls[0].name == "click_marker"
    assert request.tool_calls[0].args["marker_id"] == 29


def test_queue_click_without_screen_change_keeps_active_card(monkeypatch):
    monkeypatch.setattr(
        worker_transition,
        "transition_has_visual_change",
        lambda *_args: (False, 0.0),
    )
    state = worker_state(
        observation={
            "current_url": "https://www.saramin.co.kr/zf_user/search",
            "current_page_role": "search",
            "current_screenshot": "same-search.png",
            "ocr_complete": True,
            "current_markers": [{"id": 28, "text": "백엔드 개발자"}],
        },
        transition={
            "transition_request": {
                "action": "click_marker",
                "action_seq": 3,
                "source": "job_card_queue",
                "before_url": "https://www.saramin.co.kr/zf_user/search",
                "before_screenshot": "same-search.png",
                "started_at": time.time(),
            }
        },
        collection={
            "job_card_queue": [
                {
                    "queue_id": "card-1",
                    "status": "active",
                    "title": "백엔드 개발자",
                },
                {
                    "queue_id": "card-2",
                    "status": "pending",
                    "title": "백엔드 개발자 팀원",
                },
            ]
        },
    )

    updated = apply_update(
        state,
        worker_transition.transition_node(state, node_runtime()),
    )

    assert updated["transition"]["transition_result"]["status"] == "unknown"
    assert (
        updated["transition"]["transition_result"]["reason"]
        == "job_card_detail_not_reached"
    )
    assert updated["collection"]["job_card_queue"][0]["status"] == "active"
    assert worker_selection.selection_node(updated, node_runtime()) == {}
    assert route_after_selection(updated) == "reasoning"


def test_text_input_refreshes_ocr_after_small_screen_change(monkeypatch):
    monkeypatch.setattr(
        worker_transition,
        "transition_has_visual_change",
        lambda *_args: (False, 0.025),
    )
    request = {
        "action": "type_in_marker",
        "action_seq": 11,
        "source": "autonomous",
        "before_observation_id": "observation:0010",
        "before_screenshot": "search-overlay.png",
        "before_url": "https://www.wanted.co.kr/",
        "input_text": "iOS 개발자",
        "started_at": time.time(),
    }
    result = worker_transition.transition_node(
        worker_state(
            observation={
                "current_screenshot": "search-suggestions.png",
                "ocr_complete": False,
            },
            transition={"transition_request": request},
        ),
        node_runtime(),
    )

    transition = result["transition"]["transition_result"]
    assert transition["status"] == "needs_ocr"
    assert transition["reason"] == "input_ocr_required"
    assert transition["needs_ocr"] is True

    verified = worker_transition.transition_node(
        worker_state(
            observation={
                "observation_id": "observation:0011",
                "current_screenshot": "search-suggestions.png",
                "current_url": "https://www.wanted.co.kr/",
                "current_markers": [
                    {"id": 1, "text": "iOS 개발자"},
                    {"id": 10, "text": "iOS 개발자 · 직무"},
                ],
                "ocr_complete": True,
                "previous_observation": {
                    "observation_id": "observation:0010",
                    "screenshot": "search-overlay.png",
                    "current_url": "https://www.wanted.co.kr/",
                    "markers": [{"id": 1, "text": "검색어를 입력해 주세요"}],
                },
            },
            transition={"transition_request": request},
        ),
        node_runtime(),
    )

    verified_transition = verified["transition"]["transition_result"]
    assert verified_transition["status"] == "ready"
    assert verified_transition["reason"] == "input_text_ocr_matched"


def test_reflex_transition_rejects_change_without_saved_after_state():
    replay_results = []
    request = {
        "action": "press_key",
        "action_seq": 2,
        "source": "reflex",
        "recipe_key": "experience-rule10#search",
        "before_url": "https://www.wanted.co.kr/search",
        "started_at": time.time(),
    }
    transition = worker_transition.transition_node(
        worker_state(
            transition={"transition_request": request},
            observation={
                "current_url": "https://www.wanted.co.kr/search",
                "observation_id": "observation:0002",
                "current_markers": [{"id": 1, "text": "검색 결과"}],
                "screen_signature": {"phash": "f" * 16},
                "ocr_complete": True,
            },
            replay={
                "replay_session": {
                    "recipe_key": "experience-rule10#search",
                    "current_step_index": 0,
                    "pending_step_index": 0,
                    "step_count": 1,
                }
            },
        ),
        node_runtime(
            data=worker_data_services(
                record_recipe_replay=lambda recipe_key, succeeded: (
                    replay_results.append((recipe_key, succeeded)) or True
                ),
            )
        ),
    )
    assert transition["transition"]["transition_request"] is None
    assert transition["transition"]["transition_result"]["status"] == "unknown"
    assert (
        transition["transition"]["transition_result"]["reason"]
        == "rule_expected_effect_missing"
    )
    assert transition["replay"]["reflex_blocked_recipe_keys"] == [
        "experience-rule10#search"
    ]
    assert replay_results == [("experience-rule10#search", False)]


def test_reflex_transition_accepts_changed_url_with_dynamic_content(
    monkeypatch,
):
    monkeypatch.setattr(
        worker_transition,
        "transition_has_visual_change",
        lambda *_args: (True, 0.5),
    )
    result = worker_transition.transition_node(
        worker_state(
            transition={
                "transition_request": {
                    "source": "reflex",
                    "before_page_role": "search_overlay",
                    "before_url": "https://www.wanted.co.kr/",
                    "expected_effect": ExpectedEffect(
                        kind="url_change",
                        description="검색 결과로 이동한다",
                        expected_url_template="wanted.co.kr/search?query",
                        expected_page_role="search",
                    ),
                }
            },
            observation={
                "current_url": "https://www.wanted.co.kr/search?query=ios",
                "current_page_role": "search",
                "current_markers": [{"id": 1, "text": "검색 결과"}],
                "screen_signature": {
                    "phash": "f" * 16,
                    "size": [1920, 1080],
                },
                "ocr_complete": True,
            },
        ),
        node_runtime(),
    )

    transition = result["transition"]["transition_result"]
    assert transition["status"] == "ready"
    assert transition["reason"] == "rule_url_change_verified"


def test_reflex_url_effect_requests_ocr_before_completing(monkeypatch):
    replay_results = []
    monkeypatch.setattr(
        worker_transition,
        "transition_has_visual_change",
        lambda *_args: (True, 0.5),
    )
    result = worker_transition.transition_node(
        worker_state(
            transition={
                "transition_request": {
                    "source": "reflex",
                    "recipe_key": "experience-rule10#search",
                    "before_page_role": "home",
                    "before_url": "https://www.saramin.co.kr/zf_user/",
                    "expected_effect": ExpectedEffect(
                        kind="url_change",
                        description="검색 결과로 이동한다",
                        expected_url_template="saramin.co.kr/zf_user/search?searchword",
                    ),
                    "recipe_step_index": 1,
                    "recipe_step_count": 2,
                }
            },
            observation={
                "current_url": "https://www.saramin.co.kr/zf_user/search?searchword=AI",
                "current_screenshot": "result.png",
                "raw_screen_signature": {
                    "phash": "a" * 16,
                    "size": [1920, 1080],
                },
                "ocr_complete": False,
            },
            replay={
                "replay_session": {
                    "recipe_key": "experience-rule10#search",
                    "current_step_index": 1,
                    "pending_step_index": 1,
                    "step_count": 2,
                }
            },
        ),
        node_runtime(
            data=worker_data_services(
                record_recipe_replay=lambda recipe_key, succeeded: (
                    replay_results.append((recipe_key, succeeded)) or True
                )
            )
        ),
    )

    transition = result["transition"]["transition_result"]
    assert transition["status"] == "needs_ocr"
    assert transition["reason"] == "rule_effect_ocr_required"
    assert transition["needs_ocr"] is True
    assert result.get("replay") is None
    assert replay_results == []

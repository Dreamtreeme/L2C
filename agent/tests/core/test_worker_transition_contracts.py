import time

from agent.graph import (
    worker_selection,
    worker_transition,
)
from agent.graph.workflow import route_after_execution
from agent.tests.worker_test_support import (
    node_runtime,
    worker_data_services,
    worker_state,
)
from shared.schema.recipe_schema import ScreenCheckpoint


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
                "action": "finish_detail_reading",
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
            },
        },
        collection=state["collection"],
    )
    close = worker_selection.selection_node(failed_back, node_runtime())
    close_request = close["decision"]["pending_action"]
    assert close_request.source == "job_results_navigation"
    assert close_request.tool_calls[0].name == "close_current_tab"


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
        "recipe_key": "path6#search",
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
                    "recipe_key": "path6#search",
                    "current_transition_index": 0,
                    "pending_transition_index": 0,
                    "transition_count": 1,
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
        == "recipe_after_state_missing"
    )
    assert transition["replay"]["reflex_blocked_recipe_keys"] == ["path6#search"]
    assert replay_results == [("path6#search", False)]


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
                    "expected_after_state": ScreenCheckpoint(
                        url_template="wanted.co.kr/search?query",
                        page_role="search_results",
                        screen_context_signature={
                            "phash": "0" * 16,
                            "size": [1920, 1080],
                        },
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
    assert transition["reason"] == "recipe_after_url_matched"


def test_reflex_final_transition_accepts_url_before_ocr(monkeypatch):
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
                    "recipe_key": "experience8#search",
                    "before_page_role": "home",
                    "before_url": "https://www.saramin.co.kr/zf_user/",
                    "expected_after_state": ScreenCheckpoint(
                        url_template="saramin.co.kr/zf_user/search?searchword",
                        page_role="search_results",
                    ),
                    "recipe_transition_index": 1,
                    "recipe_transition_count": 2,
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
                    "recipe_key": "experience8#search",
                    "current_transition_index": 1,
                    "pending_transition_index": 1,
                    "transition_count": 2,
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
    assert transition["status"] == "ready"
    assert transition["reason"] == "recipe_after_url_matched"
    assert transition["needs_ocr"] is True
    assert result["replay"]["replay_session"] is None
    assert replay_results == [("experience8#search", True)]


def test_reflex_transition_does_not_accept_page_role_without_screen_match(
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
                    "before_url": "https://www.wanted.co.kr/search",
                    "expected_after_state": ScreenCheckpoint(
                        url_template="wanted.co.kr/search",
                        page_role="search_results",
                        screen_context_signature={
                            "phash": "0" * 16,
                            "size": [1920, 1080],
                        },
                    ),
                }
            },
            observation={
                "current_url": "https://www.wanted.co.kr/search",
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
    assert transition["status"] == "unknown"
    assert transition["reason"] == "screen_context_phash_distance"


def test_reflex_transition_keeps_phash_check_within_same_page_role(monkeypatch):
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
                    "before_page_role": "job_detail",
                    "expected_after_state": ScreenCheckpoint(
                        page_role="job_detail",
                        screen_context_signature={
                            "phash": "0" * 16,
                            "size": [1920, 1080],
                        },
                    ),
                }
            },
            observation={
                "current_page_role": "job_detail",
                "current_markers": [{"id": 1, "text": "상세"}],
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
    assert transition["status"] == "unknown"
    assert transition["reason"] == "screen_context_phash_distance"

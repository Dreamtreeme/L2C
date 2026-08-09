import time

from agent.graph import (
    worker_selection,
    worker_transition,
)
from agent.runtime.worker_contracts import action_event_transitions
from agent.graph.workflow import route_after_selection
from agent.tests.worker_test_support import (
    apply_update,
    node_runtime,
    worker_state,
)


def test_queue_phash_match_records_results_transition(monkeypatch):
    from agent.graph import worker_selection

    result = worker_selection.selection_node(
        worker_state(
            observation={
                "observation_id": "observation:0009",
                "current_screenshot": "returned-list.png",
                "current_url": "https://www.wanted.co.kr/search",
                "ocr_complete": False,
                "raw_screen_signature": {
                    "phash": "0" * 16,
                    "size": [1000, 1000],
                },
            },
            transition={
                "transition_result": {
                    "status": "needs_ocr",
                    "action": "go_back",
                    "action_seq": 9,
                    "source": "autonomous",
                    "started_at": time.time(),
                    "step": {
                        "seq": 9,
                        "action": "go_back",
                        "page_role": "job_detail",
                        "args": {"page_role": "job_detail"},
                    },
                },
                "action_events": [
                    {
                        "seq": 9,
                        "result": {
                            "action": "go_back",
                            "status": "success",
                        },
                    }
                ],
            },
            collection={
                "job_card_queue": [
                    {
                        "queue_id": "card-2",
                        "status": "pending",
                        "title": "두 번째 iOS 개발자",
                        "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                        "center_ratio": [0.4, 0.425],
                    }
                ],
                "job_results_memory": {
                    "screen_signature": {
                        "phash": "0" * 16,
                        "size": [1000, 1000],
                        "anchors": ["검색 결과", "두 번째 iOS 개발자"],
                    },
                },
            },
        ),
        node_runtime(),
    )

    assert result["decision"]["pending_action"].source == "job_card_queue"
    record = action_event_transitions(result["transition"]["action_events"])[0]
    assert isinstance(record, dict)
    assert record["action_seq"] == 9
    assert record["action"] == "go_back"
    assert record["status"] == "ready"
    assert record["reason"] == "queue_results_phash_match"
    assert record["marker_texts"] == [
        "검색 결과",
        "두 번째 iOS 개발자",
    ]


def test_queue_phash_mismatch_falls_through_to_ocr():
    transition_result = {
        "status": "needs_ocr",
        "action": "go_back",
        "action_seq": 9,
        "source": "page_policy",
        "needs_ocr": True,
        "started_at": time.time(),
        "step": {
            "seq": 9,
            "action": "go_back",
            "page_role": "job_detail",
        },
    }
    state = worker_state(
        observation={
            "observation_id": "observation:0009",
            "current_url": "https://www.wanted.co.kr/search",
            "ocr_complete": False,
            "raw_screen_signature": {
                "phash": "f" * 16,
                "size": [1000, 1000],
            },
        },
        transition={
            "transition_request": {},
            "transition_result": transition_result,
        },
        collection={
            "job_card_queue": [
                {
                    "queue_id": "card-2",
                    "status": "pending",
                    "title": "두 번째 iOS 개발자",
                    "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                },
            ],
            "job_results_memory": {
                "screen_signature": {
                    "phash": "0" * 16,
                    "size": [1000, 1000],
                    "anchors": ["검색 결과", "두 번째 iOS 개발자"],
                },
            },
        },
    )

    result = worker_selection.selection_node(state, node_runtime())

    assert "decision" not in result
    assert "transition" not in result
    assert route_after_selection(apply_update(state, result)) == "ocr"


def test_queue_click_without_screen_change_refreshes_ocr(monkeypatch):
    monkeypatch.setattr(
        worker_transition,
        "transition_has_visual_change",
        lambda *_args: (False, 0.0),
    )
    state = worker_state(
        observation={
            "observation_id": "observation:0010",
            "current_screenshot": "same-list.png",
            "ocr_complete": False,
        },
        transition={
            "transition_request": {
                "action": "click_marker",
                "action_seq": 10,
                "source": "job_card_queue",
                "started_at": time.time(),
            },
        },
        collection={
            "job_card_queue": [
                {
                    "queue_id": "card-2",
                    "status": "active",
                    "title": "두 번째 공고",
                }
            ]
        },
    )

    result = worker_transition.transition_node(state, node_runtime())
    updated = apply_update(state, result)

    assert updated["transition"]["transition_request"]["action"] == "click_marker"
    assert updated["transition"]["transition_result"]["needs_ocr"] is True
    assert (
        updated["transition"]["transition_result"]["reason"]
        == "queue_click_no_screen_change"
    )
    assert updated["collection"]["job_card_queue"][0]["status"] == "pending"
    assert worker_selection.selection_node(updated, node_runtime()) == {}
    assert route_after_selection(updated) == "ocr"


def test_queue_click_retry_without_change_returns_to_reasoning(monkeypatch):
    monkeypatch.setattr(
        worker_transition,
        "transition_has_visual_change",
        lambda *_args: (False, 0.0),
    )
    state = worker_state(
        observation={
            "observation_id": "observation:0012",
            "current_screenshot": "same-list-again.png",
            "ocr_complete": False,
        },
        transition={
            "transition_request": {
                "action": "click_marker",
                "action_seq": 12,
                "source": "job_card_queue",
                "started_at": time.time(),
            },
            "action_events": [
                {
                    "seq": 11,
                    "result": {"action": "click_marker", "status": "success"},
                    "transition": {
                        "action": "click_marker",
                        "source": "job_card_queue",
                        "status": "unknown",
                        "reason": "no_screen_change",
                    },
                },
                {
                    "seq": 12,
                    "result": {"action": "click_marker", "status": "success"},
                },
            ],
        },
        collection={
            "job_card_queue": [
                {
                    "queue_id": "card-2",
                    "status": "active",
                    "title": "두 번째 공고",
                }
            ]
        },
    )

    result = worker_transition.transition_node(state, node_runtime())
    updated = apply_update(state, result)

    assert updated["transition"]["transition_request"] == {}
    assert (
        updated["transition"]["transition_result"]["reason"]
        == "queue_retry_no_screen_change"
    )
    assert updated["collection"]["job_card_queue"][0]["status"] == "pending"
    assert worker_selection.selection_node(updated, node_runtime()) == {}
    assert route_after_selection(updated) == "reasoning"


def test_reflex_transition_rejects_change_without_saved_after_state():
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
        ),
        node_runtime(),
    )
    assert transition["transition"]["transition_request"] == {}
    assert transition["transition"]["transition_result"]["status"] == "unknown"
    assert (
        transition["transition"]["transition_result"]["reason"]
        == "recipe_after_state_missing"
    )
    assert transition["replay"]["reflex_blocked_recipe_keys"] == ["path6#search"]


def test_reflex_transition_accepts_changed_page_role_with_dynamic_content(
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
                    "expected_after_state": {
                        "url_template": "wanted.co.kr/search?query",
                        "page_role": "search_results",
                        "screen_context_signature": {
                            "phash": "0" * 16,
                            "size": [1920, 1080],
                        },
                    },
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
    evidence = transition["after_state_match"]
    assert transition["status"] == "ready"
    assert transition["reason"] == "recipe_after_page_role_matched"
    assert evidence["expected_page_role"] == "search"


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
                    "expected_after_state": {
                        "page_role": "job_detail",
                        "screen_context_signature": {
                            "phash": "0" * 16,
                            "size": [1920, 1080],
                        },
                    },
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

import time

from agent.graph import (
    worker_execution_dispatch,
    worker_selection,
    worker_transition,
)
from agent.graph.workflow import route_after_selection
from agent.graph.worker_reflex import reflex_node
from agent.runtime.job_card_queue import replay_job_card_after_return


def test_queue_phash_match_records_return_transition(monkeypatch):
    from agent.graph import worker_selection

    result = worker_selection.selection_node(
        {
            "current_capture_id": "capture:0009",
            "current_screenshot": "returned-list.png",
            "current_url": "https://www.wanted.co.kr/search",
            "ocr_complete": False,
            "raw_screen_signature": {
                "phash": "0" * 16,
                "size": [1000, 1000],
            },
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
            "active_job_card": {},
        }
    )

    assert result["pending_action"].source == "job_card_queue"
    record = result["transition_records"][0]
    assert record["action_seq"] == 9
    assert record["action"] == "go_back"
    assert record["status"] == "ready"
    assert record["reason"] == "queue_return_phash_match"
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
    state = {
        "current_capture_id": "capture:0009",
        "ocr_capture_id": "",
        "current_url": "https://www.wanted.co.kr/search",
        "ocr_complete": False,
        "raw_screen_signature": {
            "phash": "f" * 16,
            "size": [1000, 1000],
        },
        "transition_request": {},
        "transition_result": transition_result,
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
        "active_job_card": {},
    }

    result = worker_selection.selection_node(state)

    assert "pending_action" not in result
    assert "transition_request" not in result
    assert route_after_selection({**state, **result}) == "ocr"


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
        {
            "transition_request": request,
            "current_url": "https://www.wanted.co.kr/search",
            "current_capture_id": "capture:0002",
            "current_markers": [{"id": 1, "text": "검색 결과"}],
            "screen_signature": {"phash": "f" * 16},
            "ocr_complete": True,
        }
    )
    assert transition["transition_request"] == {}
    assert transition["transition_result"]["status"] == "unknown"
    assert (
        transition["transition_result"]["reason"]
        == "recipe_after_state_missing"
    )
    assert transition["reflex_blocked_recipe_keys"] == [
        "path6#search"
    ]


def test_reflex_transition_accepts_changed_page_role_with_dynamic_content():
    from agent.graph.worker_transition_policy import verify_reflex_after_state

    matched, reason, evidence = verify_reflex_after_state(
        {
            "before_page_role": "search_overlay",
            "expected_after_state": {
                "url_template": "wanted.co.kr/search?query",
                "page_role": "search_results",
                "screen_context_signature": {
                    "phash": "0" * 16,
                    "size": [1920, 1080],
                },
            },
        },
        {
            "current_url": "https://www.wanted.co.kr/search?query=ios",
            "current_page_role": "search",
            "screen_signature": {
                "phash": "f" * 16,
                "size": [1920, 1080],
            },
        },
    )

    assert matched is True
    assert reason == "recipe_after_page_role_matched"
    assert evidence["expected_page_role"] == "search"


def test_reflex_transition_keeps_phash_check_within_same_page_role():
    from agent.graph.worker_transition_policy import verify_reflex_after_state

    matched, reason, _ = verify_reflex_after_state(
        {
            "before_page_role": "job_detail",
            "expected_after_state": {
                "page_role": "job_detail",
                "screen_context_signature": {
                    "phash": "0" * 16,
                    "size": [1920, 1080],
                },
            },
        },
        {
            "current_page_role": "job_detail",
            "screen_signature": {
                "phash": "f" * 16,
                "size": [1920, 1080],
            },
        },
    )

    assert matched is False
    assert reason == "screen_context_phash_distance"

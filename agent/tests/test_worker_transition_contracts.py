import time

from agent.graph import (
    worker_execution_dispatch,
    worker_observation,
    worker_selection,
    worker_transition,
)
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


def test_queue_return_waits_for_saved_phash_before_ocr(monkeypatch):
    from agent.graph import worker_selection

    transition_request = {
        "status": "needs_ocr",
        "action": "go_back",
        "action_seq": 9,
        "source": "page_policy",
        "started_at": time.time(),
        "step": {
            "seq": 9,
            "action": "go_back",
            "page_role": "job_detail",
        },
    }
    result = worker_selection.selection_node(
        {
            "current_url": "https://www.wanted.co.kr/search",
            "ocr_complete": False,
            "raw_screen_signature": {
                "phash": "f" * 16,
                "size": [1000, 1000],
            },
            "transition_request": transition_request,
            "transition_result": transition_request,
            "job_card_queue": [
                {
                    "queue_id": "card-2",
                    "status": "pending",
                    "title": "두 번째 iOS 개발자",
                    "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
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

    assert "pending_action" not in result
    assert result["transition_result"]["status"] == "pending"
    assert result["transition_result"]["needs_ocr"] is False
    assert (
        result["transition_request"]["pending_target_phash"]
        == "0" * 16
    )


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


def test_target_phash_wait_skips_capture_until_match(monkeypatch):
    class FakePerception:
        def wait_for_transition_phash_match(
            self,
            _target_phash,
            *,
            max_distance,
            max_wait_sec=None,
        ):
            assert max_distance == 9
            return False

        def capture_usable_screen(self, **_kwargs):
            raise AssertionError("목표 pHash 전에는 파일 캡처를 하면 안 됩니다.")

    monkeypatch.setattr(
        worker_observation,
        "_perception_engine",
        lambda: FakePerception(),
    )
    capture = worker_observation.capture_node(
        {
            "transition_request": {
                "action": "go_back",
                "pending_target_phash": "0" * 16,
                "pending_target_max_distance": 9,
                "started_at": time.time(),
            },
            "ocr_complete": False,
        }
    )

    assert capture == {
        "ocr_complete": False,
        "ocr_capture_id": "",
        "transition_probe_unchanged": True,
    }


def test_target_phash_timeout_falls_back_to_ocr_once():
    request = {
        "action": "go_back",
        "source": "llm",
        "pending_target_phash": "0" * 16,
        "pending_target_max_distance": 9,
        "started_at": time.time() - 20.0,
        "attempts": 2,
    }

    transition = worker_transition.transition_node(
        {
            "transition_request": request,
            "transition_probe_unchanged": True,
            "ocr_complete": False,
        }
    )
    assert transition["transition_request"] == {}
    assert transition["transition_result"]["status"] == "unknown"
    assert transition["transition_result"]["needs_ocr"] is True

    selection = worker_selection.selection_node(
        {
            "current_url": "https://www.wanted.co.kr/search",
            "ocr_complete": False,
            "raw_screen_signature": {
                "phash": "f" * 16,
                "size": [1000, 1000],
            },
            "transition_request": {},
            "transition_result": transition["transition_result"],
            "job_card_queue": [
                {
                    "queue_id": "card-2",
                    "status": "pending",
                    "title": "두 번째 iOS 개발자",
                    "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                }
            ],
            "job_results_memory": {
                "screen_signature": {
                    "phash": "0" * 16,
                    "size": [1000, 1000],
                },
            },
            "active_job_card": {},
        }
    )

    assert "transition_request" not in selection

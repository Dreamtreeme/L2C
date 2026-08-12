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
    worker_data_services,
    worker_state,
)
from shared.schema.recipe_schema import ScreenCheckpoint


def test_queue_phash_match_records_results_transition(monkeypatch):
    from agent.graph import worker_selection

    monkeypatch.setattr(
        "agent.runtime.job_card_queue.roi_signature_match",
        lambda *_args, **_kwargs: {"matched": True, "reason": "roi_matched"},
    )

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
                },
                "action_events": [
                    {
                        "seq": 9,
                        "result": {
                            "action": "go_back",
                            "status": "success",
                        },
                        "candidate_action": {
                            "source_seq": 9,
                            "action": "go_back",
                        },
                        "before_checkpoint": {
                            "observation_id": "observation:0008",
                            "url_template": "wanted.co.kr/wd/{id}",
                            "page_role": "job_detail",
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
                        "roi_signature": {"phash": "0" * 16},
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
    assert record.seq == 9
    assert record.actions[0].action == "go_back"
    assert record.evidence.status == "ready"
    assert record.evidence.reason == "queue_results_phash_match"
    assert record.evidence.after_marker_texts == [
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


def test_cached_queue_click_without_screen_change_refreshes_ocr():
    state = worker_state(
        observation={
            "observation_id": "observation:0010",
            "current_screenshot": "same-list.png",
            "ocr_complete": False,
            "raw_screen_signature": {
                "phash": "0" * 16,
                "size": [1000, 1000],
            },
            "previous_observation": {
                "observation_id": "observation:0009",
                "screenshot": "same-list.png",
                "current_url": "https://www.wanted.co.kr/search",
                "markers": [
                    {
                        "id": 0,
                        "bbox": [300, 400, 500, 450],
                        "text": "두 번째 공고",
                        "type": "queue_cached_card",
                    }
                ],
                "screen_signature": {
                    "phash": "0" * 16,
                    "size": [1000, 1000],
                },
                "page_role": "search",
            },
        },
        transition={
            "transition_request": {
                "action": "click_marker",
                "action_seq": 10,
                "source": "job_card_queue",
                "started_at": time.time(),
                "target_marker_id": 0,
            },
        },
        collection={
            "job_card_queue": [
                {
                    "queue_id": "card-2",
                    "status": "active",
                    "title": "두 번째 공고",
                    "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                    "center_ratio": [0.4, 0.425],
                }
            ],
            "job_results_memory": {
                "screen_signature": {
                    "phash": "0" * 16,
                    "size": [1000, 1000],
                    "anchors": ["두 번째 공고"],
                }
            },
        },
    )

    result = worker_transition.transition_node(state, node_runtime())
    updated = apply_update(state, result)

    assert updated["transition"]["transition_result"]["reason"] == (
        "queue_cached_marker_refresh_required"
    )
    assert updated["transition"]["transition_result"]["needs_ocr"] is True
    assert updated["collection"]["job_card_queue"][0]["status"] == "pending"
    assert worker_selection.selection_node(updated, node_runtime()) == {}
    assert route_after_selection(updated) == "ocr"

    refreshed = apply_update(
        updated,
        {
            "observation": {
                "current_url": "https://www.wanted.co.kr/search",
                "current_screenshot": "same-list.png",
                "current_markers": [
                    {
                        "id": 29,
                        "bbox": [300, 400, 500, 450],
                        "text": "두 번째 공고",
                        "type": "text",
                    }
                ],
                "screen_signature": {
                    "phash": "0" * 16,
                    "size": [1000, 1000],
                    "anchors": ["두 번째 공고"],
                },
                "current_page_role": "search",
                "ocr_complete": True,
            }
        },
    )
    verified = apply_update(
        refreshed,
        worker_transition.transition_node(refreshed, node_runtime()),
    )
    replay = worker_selection.selection_node(verified, node_runtime())

    assert verified["transition"]["transition_result"]["status"] == "unknown"
    assert verified["transition"]["transition_result"]["queue_marker_refresh"] is True
    assert replay["decision"]["pending_action"].source == "job_card_queue"
    assert replay["decision"]["pending_action"].tool_calls[0].args["marker_id"] == 29


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

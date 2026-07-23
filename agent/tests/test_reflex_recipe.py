import sqlite3

from agent.graph import (
    worker_execution,
    worker_observation,
    worker_reasoning,
    worker_recording,
    worker_selection,
    worker_transition,
)
from agent.graph.worker_observation_context import build_ui_context
from agent.runtime.reflex_runtime import reflex_node
from agent.runtime.result_card_queue import (
    complete_active_result_card,
    mark_result_card_active,
    queue_replay_after_return,
    result_card_click_matches_queue,
)


def _action_request(*, content="", tool_calls=None, source="llm"):
    from agent.graph.action_request import build_action_request

    return build_action_request(source, str(content or ""), list(tool_calls or []))


def test_record_ui_step_stays_in_marker_text_space():
    from agent.recipe.record import record_ui_step

    steps = []
    state = {
        "goal": "지원하기",
        "current_url": "https://www.wanted.co.kr/wd/12345",
        "screen_signature": {
            "phash": "0" * 16,
            "size": [200, 200],
            "anchors": ["지원하기", "공유하기"],
        },
        "current_markers": [
            {"id": 1, "bbox": [20, 20, 120, 60], "text": "지원하기"},
            {"id": 2, "bbox": [20, 80, 120, 120], "text": "공유하기"},
        ],
    }

    record_ui_step(
        steps,
        state,
        "click_marker",
        {
            "marker_id": 1,
            "reason": "open apply flow",
            "target_role": "apply_button",
            "target_component": "job_detail_header",
            "expected_after": "application modal opens",
        },
        0,
    )

    assert "state_key" not in steps[0]
    assert "state_anchors" not in steps[0]
    assert "screen_signature" not in steps[0]
    assert steps[0]["target"] == {
        "text": "지원하기",
        "region": "top-left",
        "ordinal": 0,
        "evidence_texts": ["공유하기"],
        "bbox_ratio": [0.1, 0.1, 0.6, 0.3],
        "center_ratio": [0.35, 0.2],
    }
    assert steps[0]["intent"] == "open apply flow"
    assert steps[0]["target_role"] == "apply_button"
    assert steps[0]["component"] == "job_detail_header"
    assert steps[0]["expected_after"] == "application modal opens"
    assert "bbox" not in steps[0]["target"]


def test_phash_replay_rejects_step_without_roi_signature():
    from agent.recipe.phash_replay import match_step_by_screen_signature

    current_signature = {
        "phash": "f0f0f0f0f0f0f0f0",
        "size": [1000, 1000],
        "anchors": ["검색", "채용"],
    }
    step = {
        "screen_signature": {
            "phash": "f0f0f0f0f0f0f0f0",
            "size": [1000, 1000],
            "anchors": ["검색", "채용"],
        },
        "target": {
            "text": "검색",
            "bbox_ratio": [0.79, 0.08, 0.83, 0.12],
            "center_ratio": [0.81, 0.10],
        },
    }
    markers = [
        {"id": 99, "bbox": [790, 80, 830, 120], "text": "검색"},
        {"id": 3, "bbox": [100, 100, 140, 140], "text": "검색"},
    ]

    marker_id, result = match_step_by_screen_signature(step, current_signature, markers)

    assert marker_id is None
    assert result["matched"] is False
    assert result["reason"] == "roi_signature_missing"
    assert result["mode"] == "roi_phash"


def test_record_ui_step_stores_target_roi_signature(tmp_path):
    from PIL import Image, ImageDraw
    from agent.recipe.record import record_ui_step

    screenshot = tmp_path / "screen.png"
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([150, 20, 170, 40], fill="black")
    image.save(screenshot)

    steps = []
    state = {
        "goal": "검색",
        "current_url": "https://www.wanted.co.kr",
        "current_page_role": "home",
        "screen_signature": {
            "phash": "0" * 16,
            "size": [200, 200],
            "anchors": ["검색", "채용"],
        },
        "recent_images": [screenshot],
        "current_markers": [
            {"id": 1, "bbox": [150, 20, 170, 40], "text": "검색"},
            {"id": 2, "bbox": [20, 20, 60, 40], "text": "채용"},
        ],
    }

    record_ui_step(
        steps,
        state,
        "click_marker",
        {
            "marker_id": 1,
            "reason": "검색 아이콘 클릭",
            "target_role": "button",
            "target_component": "search_button",
        },
        0,
    )

    roi_signature = steps[0]["roi_signature"]
    crop_rect = roi_signature["crop_rect_ratio"]

    assert steps[0]["page_role"] == "home"
    assert roi_signature["algorithm"] == "roi-phash-dct64-v2"
    assert len(roi_signature["phash"]) == 16
    assert crop_rect[0] <= steps[0]["target"]["bbox_ratio"][0]
    assert crop_rect[2] >= steps[0]["target"]["bbox_ratio"][2]
    assert roi_signature["target_center_ratio"] == [0.8, 0.15]


def test_marker_geometry_ratio_helpers_are_consistent():
    from agent.vision.marker_geometry import (
        bbox_from_ratio,
        bbox_to_ratio,
        center_ratio_from_bbox,
        marker_bbox,
        marker_center,
        marker_center_ratio,
        screen_size_from_signature,
    )

    marker = {"bbox": ["10", "20", "30", "60"]}
    size = [100, 200]

    assert marker_bbox(marker) == [10, 20, 30, 60]
    assert marker_center(marker) == (20, 40)
    assert bbox_to_ratio(marker_bbox(marker), size) == [0.1, 0.1, 0.3, 0.3]
    assert center_ratio_from_bbox(marker_bbox(marker), size) == [0.2, 0.2]
    assert marker_center_ratio(marker, size) == [0.2, 0.2]
    assert bbox_from_ratio([0.1, 0.1, 0.3, 0.3], size) == [10, 20, 30, 60]
    assert screen_size_from_signature({"size": ["100", "200"]}) == [100, 200]


def test_roi_phash_replay_matches_when_full_phash_differs(tmp_path):
    from PIL import Image, ImageDraw
    from agent.recipe.phash_replay import match_step_by_screen_signature
    from agent.vision.screen_signature import compute_target_roi_signature

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        if path == current:
            draw.rectangle([0, 120, 200, 200], fill="black")
        image.save(path)

    roi_signature = compute_target_roi_signature(saved, [150, 20, 170, 40], [200, 200])
    step = {
        "screen_signature": {
            "phash": "f" * 16,
            "size": [200, 200],
        },
        "roi_signature": roi_signature,
        "target": {
            "text": "검색",
            "bbox_ratio": [0.75, 0.1, 0.85, 0.2],
            "center_ratio": [0.8, 0.15],
        },
    }
    current_signature = {
        "phash": "0" * 16,
        "size": [200, 200],
    }
    markers = [{"id": 7, "bbox": [150, 20, 170, 40], "text": "검색"}]

    marker_id, result = match_step_by_screen_signature(
        step,
        current_signature,
        markers,
        current_image_path=str(current),
    )

    assert marker_id == 7
    assert result["matched"] is True
    assert result["mode"] == "roi_phash"
    assert result["reason"] == "roi_matched"


def test_roi_phash_replay_rejects_different_capture_size(tmp_path):
    from PIL import Image, ImageDraw
    from agent.recipe.phash_replay import match_step_by_screen_signature
    from agent.vision.screen_signature import compute_target_roi_signature

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).rectangle([150, 20, 170, 40], fill="black")
    image.save(saved)
    Image.new("RGB", (300, 200), "white").save(current)

    roi_signature = compute_target_roi_signature(saved, [150, 20, 170, 40], [200, 200])
    marker_id, result = match_step_by_screen_signature(
        {
            "roi_signature": roi_signature,
            "target": {"text": "검색", "center_ratio": [0.8, 0.15]},
        },
        {"size": [300, 200]},
        [{"id": 7, "bbox": [225, 20, 255, 40], "text": "검색"}],
        current_image_path=str(current),
    )

    assert marker_id is None
    assert result["reason"] == "capture_size_mismatch"
    assert result["saved_size"] == [200, 200]
    assert result["current_size"] == [300, 200]


def test_roi_phash_replay_rejects_when_saved_roi_changes(tmp_path):
    from PIL import Image, ImageDraw
    from agent.recipe.phash_replay import match_step_by_screen_signature
    from agent.vision.screen_signature import compute_target_roi_signature

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    saved_image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(saved_image).rectangle([150, 20, 170, 40], fill="black")
    saved_image.save(saved)
    current_image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(current_image).rectangle([125, 20, 145, 40], fill="black")
    current_image.save(current)

    roi_signature = compute_target_roi_signature(saved, [150, 20, 170, 40], [200, 200])
    marker_id, result = match_step_by_screen_signature(
        {
            "roi_signature": roi_signature,
            "target": {"text": "검색", "center_ratio": [0.8, 0.15]},
        },
        {"size": [200, 200]},
        [{"id": 9, "bbox": [125, 20, 145, 40], "text": "검색"}],
        current_image_path=str(current),
    )

    assert marker_id is None
    assert result["matched"] is False
    assert result["reason"] == "roi_phash_distance"


def test_roi_phash_replay_uses_closest_marker_without_text_matching(tmp_path):
    from PIL import Image, ImageDraw
    from agent.recipe.phash_replay import match_step_by_screen_signature
    from agent.vision.screen_signature import compute_target_roi_signature

    image_path = tmp_path / "screen.png"
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).rectangle([150, 20, 170, 40], fill="black")
    image.save(image_path)
    signature = compute_target_roi_signature(image_path, [150, 20, 170, 40], [200, 200])
    marker_id, result = match_step_by_screen_signature(
        {
            "roi_signature": signature,
            "target": {
                "center_ratio": [0.8, 0.15],
                "text": "저장 당시 글자",
            },
        },
        {"size": [200, 200]},
        [
            {"id": 7, "bbox": [150, 20, 170, 40], "text": "현재는 다른 글자"},
            {"id": 9, "bbox": [140, 20, 160, 40], "text": "저장 당시 글자"},
        ],
        current_image_path=str(image_path),
    )

    assert marker_id == 7
    assert result["matched"] is True
    assert result["mode"] == "roi_phash"


def test_page_role_match_requires_exact_screen_role():
    from agent.recipe.page_context import page_role_matches

    assert page_role_matches("home", "home") is True
    assert page_role_matches("search_overlay", "search_overlay") is True
    assert page_role_matches("search_overlay", "home") is False


def test_text_utils_normalizes_marker_noise_and_url_template():
    from agent.recipe.text_utils import normalize_text, recipe_url_scope_matches, url_template

    assert normalize_text("[id: 3] 검색어를 입력해주세요") == "검색어를 입력해주세요"
    assert url_template("https://www.wanted.co.kr/wd/12345?query=ios&tab=position") == (
        "wanted.co.kr/wd/{id}?query,tab"
    )
    assert recipe_url_scope_matches("wanted.co.kr/wd/{id}", "https://www.wanted.co.kr/wd/98765") is True
    assert recipe_url_scope_matches("wanted.co.kr/", "https://social.wanted.co.kr/community") is False
    assert recipe_url_scope_matches("wanted.co.kr/search?query", "https://www.wanted.co.kr/search?query=ios") is True


def test_transition_contract_waits_for_known_result_outcomes():
    from agent.recipe.transition import evaluate_transition

    contract = {
        "common_ready_cues": [
            {"kind": "slot_text", "slot": "query"},
            {"kind": "text_any", "values": ["포지션", "회사"]},
        ],
        "outcomes": [
            {"name": "results_found", "cues": [{"kind": "text_all", "values": ["Android 개발자", "마크노바"]}]},
            {"name": "results_empty", "cues": [{"kind": "text_any", "values": ["검색 결과 없음", "0건"]}]},
        ],
        "loading_cues": [{"kind": "text_any", "values": ["포지션(0)"]}],
        "timeout_sec": 5,
    }
    skeleton = [
        {"text": "android 개발자"},
        {"text": "포지션(0)"},
        {"text": "회사(0)"},
    ]
    found = skeleton + [{"text": "Android 개발자"}, {"text": "마크노바"}]
    empty = skeleton + [{"text": "검색 결과 없음"}]

    assert evaluate_transition(contract, skeleton, {"query": "android 개발자"}, 1)["status"] == "pending"
    assert evaluate_transition(contract, found, {"query": "android 개발자"}, 2)["outcome"] == "results_found"
    assert evaluate_transition(contract, empty, {"query": "android 개발자"}, 2)["outcome"] == "results_empty"
    assert evaluate_transition(contract, skeleton, {"query": "android 개발자"}, 6)["status"] == "unknown"
    assert evaluate_transition({}, skeleton, {"query": "android 개발자"}, 0)["reason"] == "transition_contract_missing"
    assert evaluate_transition({"timeout_sec": 12}, skeleton, {"query": "android 개발자"}, 0)["reason"] == "transition_contract_empty"


def test_perception_node_records_and_resolves_pending_transition(monkeypatch, tmp_path):
    import time
    from PIL import Image

    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        def capture_screen(self):
            return screenshot

        def analyze_ui(self, _path):
            return {
                "markers": [
                    {"id": 1, "bbox": [10, 150, 200, 180], "text": "android 개발자"},
                    {"id": 2, "bbox": [10, 200, 200, 230], "text": "포지션"},
                    {"id": 3, "bbox": [10, 250, 300, 280], "text": "Android App 개발자"},
                ],
                "marked_image": str(screenshot),
            }

    monkeypatch.setattr(worker_observation, "_perception_engine", lambda: FakePerception())
    result = worker_observation.observe_screen_cycle(
        {
            "worker_run_id": "worker-transition",
            "worker_attempt_index": 0,
            "capture_sequence": 7,
            "current_url": "https://www.wanted.co.kr",
            "current_url_stale": False,
            "pending_transition": {
                "action_seq": 3,
                "action": "press_key",
                "from_capture_id": "worker-transition:attempt:00:capture:0007",
                "expected_after": "검색 결과가 나타남",
                "source": "reflex",
                "started_at": time.time(),
                "attempts": 0,
                "params": {"query": "android 개발자"},
                "contract": {
                    "common_ready_cues": [
                        {"kind": "slot_text", "slot": "query"},
                        {"kind": "text_any", "values": ["포지션"]},
                    ],
                    "outcomes": [
                        {"name": "results_found", "cues": [{"kind": "text_any", "values": ["Android App 개발자"]}]}
                    ],
                },
            },
        }
    )

    assert result["transition_status"] == "ready"
    assert result["transition_outcome"] == "results_found"
    assert result["pending_transition"] == {}
    assert result["transition_observations"][0]["action_seq"] == 3
    assert (
        result["transition_observations"][0]["from_capture_id"]
        == "worker-transition:attempt:00:capture:0007"
    )
    assert (
        result["transition_observations"][0]["to_capture_id"]
        == "worker-transition:attempt:00:capture:0008"
    )
    assert "Android App 개발자" in result["transition_observations"][0]["marker_texts"]


def test_perception_node_blocks_reflex_recipe_after_unknown_transition(monkeypatch, tmp_path):
    import time
    from PIL import Image

    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        def capture_screen(self):
            return screenshot

        def analyze_ui(self, _path):
            return {
                "markers": [{"id": 1, "bbox": [10, 150, 200, 180], "text": "검색"}],
                "marked_image": str(screenshot),
            }

    monkeypatch.setattr(worker_observation, "_perception_engine", lambda: FakePerception())
    result = worker_observation.observe_screen_cycle(
        {
            "current_url": "https://www.wanted.co.kr",
            "current_url_stale": False,
            "reflex_blocked_recipe_keys": [],
            "pending_transition": {
                "action_seq": 3,
                "action": "click_marker",
                "expected_after": "검색 결과가 나타남",
                "source": "reflex",
                "recipe_key": "roi#bad",
                "started_at": time.time(),
                "attempts": 0,
                "params": {},
                "contract": {},
            },
        }
    )

    assert result["transition_status"] == "unknown"
    assert result["pending_transition"] == {}
    assert result["transition_observations"][0]["recipe_key"] == "roi#bad"
    assert result["reflex_blocked_recipe_keys"] == ["roi#bad"]


def test_perception_node_blocks_reflex_click_when_screen_does_not_change(monkeypatch, tmp_path):
    import time
    from PIL import Image

    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        def capture_screen(self):
            return screenshot

        def analyze_ui(self, _path):
            raise AssertionError("pHash no-effect precheck should skip OCR")

    monkeypatch.setattr(worker_observation, "_perception_engine", lambda: FakePerception())
    monkeypatch.setattr(
        worker_observation,
        "raw_screen_phash_signature",
        lambda _image_path: {"phash": "0" * 16, "size": [800, 600]},
    )
    result = worker_observation.observe_screen_cycle(
        {
            "current_url": "https://www.wanted.co.kr/search?query=ios",
            "current_url_stale": False,
            "current_markers": [{"id": 1, "bbox": [10, 150, 200, 180], "text": "포지션"}],
            "marked_image": str(screenshot),
            "ui_context": "포지션",
            "reflex_blocked_recipe_keys": [],
            "pending_transition": {
                "action_seq": 3,
                "action": "click_marker",
                "expected_after": "포지션 결과가 보임",
                "source": "reflex",
                "recipe_key": "roi#tab",
                "step": {"seq": 3, "action": "click_marker", "marker_id": 1},
                "before_url": "https://www.wanted.co.kr/search?query=ios",
                "before_phash": "0" * 16,
                "started_at": time.time(),
                "attempts": 0,
                "params": {},
                "contract": {
                    "common_ready_cues": [{"kind": "text_any", "values": ["포지션"]}],
                },
            },
        }
    )

    assert result["transition_status"] == "unknown"
    assert result["transition_observations"][0]["reason"] == "reflex_no_screen_change"
    assert result["transition_observations"][0]["phash_distance"] == 0
    assert result["transition_observations"][0]["ocr_skipped"] is True
    assert result["transition_observations"][0]["step"]["action"] == "click_marker"
    assert result["reflex_blocked_recipe_keys"] == ["roi#tab"]


def test_perception_node_accepts_tab_visual_change_when_ocr_cue_is_pending(monkeypatch, tmp_path):
    import time
    from PIL import Image

    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        def capture_screen(self):
            return screenshot

        def analyze_ui(self, _path):
            return {
                "markers": [{"id": 1, "bbox": [10, 150, 200, 180], "text": "아직 다른 문구"}],
                "marked_image": str(screenshot),
            }

    monkeypatch.setattr(worker_observation, "_perception_engine", lambda: FakePerception())

    result = worker_observation.observe_screen_cycle(
        {
            "current_url": "https://www.wanted.co.kr/search?query=ios",
            "current_url_stale": False,
            "reflex_blocked_recipe_keys": [],
            "pending_transition": {
                "action_seq": 3,
                "action": "click_marker",
                "expected_after": "포지션 결과가 보임",
                "source": "reflex",
                "recipe_key": "roi#tab",
                "step": {"seq": 3, "action": "click_marker", "component": "tab_button"},
                "before_url": "https://www.wanted.co.kr/search?query=ios",
                "before_phash": "0" * 16,
                "started_at": time.time(),
                "attempts": 0,
                "params": {},
                "contract": {
                    "common_ready_cues": [{"kind": "text_any", "values": ["포지션"]}],
                    "timeout_sec": 12,
                },
            },
        }
    )

    assert result["transition_status"] == "ready"
    assert result["pending_transition"] == {}
    assert result["transition_observations"][0]["reason"] == "screen_change_phash_matched"
    assert result["transition_observations"][0]["phash_distance"] > 2


def test_perception_node_accepts_ready_cue_when_pixels_changed_but_phash_is_same(monkeypatch, tmp_path):
    import time
    from PIL import Image, ImageDraw

    before = tmp_path / "before.png"
    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (800, 600), "white").save(before)
    current = Image.new("RGB", (800, 600), "white")
    ImageDraw.Draw(current).rectangle((40, 160, 760, 500), fill="black")
    current.save(screenshot)

    class FakePerception:
        def capture_screen(self):
            return screenshot

        def analyze_ui(self, _path):
            return {
                "markers": [{"id": 1, "bbox": [10, 150, 200, 180], "text": "포지션"}],
                "marked_image": str(screenshot),
            }

    monkeypatch.setattr(worker_observation, "_perception_engine", lambda: FakePerception())
    monkeypatch.setattr(
        worker_observation,
        "raw_screen_phash_signature",
        lambda _image_path: {"phash": "0" * 16, "size": [800, 600]},
    )
    result = worker_observation.observe_screen_cycle(
        {
            "current_url": "https://www.wanted.co.kr/search?query=ios",
            "current_url_stale": False,
            "reflex_blocked_recipe_keys": [],
            "pending_transition": {
                "action_seq": 3,
                "action": "click_marker",
                "expected_after": "포지션 결과가 보임",
                "source": "reflex",
                "recipe_key": "roi#tab",
                "step": {"seq": 3, "action": "click_marker", "component": "tab_button"},
                "before_url": "https://www.wanted.co.kr/search?query=ios",
                "before_phash": "0" * 16,
                "before_screenshot": str(before),
                "started_at": time.time(),
                "attempts": 0,
                "params": {},
                "contract": {
                    "common_ready_cues": [{"kind": "text_any", "values": ["포지션"]}],
                    "timeout_sec": 12,
                },
            },
        }
    )

    assert result["transition_status"] == "ready"
    assert result["pending_transition"] == {}
    assert result["transition_observations"][0]["reason"] == "common_ready_cues_matched"
    assert result["transition_observations"][0]["visual_change_ratio"] >= 0.03


def test_set_result_card_queue_stores_visible_card_ratios():

    state = {
        "current_markers": [
            {"id": 10, "bbox": [100, 200, 300, 240], "text": "iOS 개발자"},
            {"id": 11, "bbox": [100, 245, 260, 270], "text": "보이저엑스"},
        ],
        "screen_signature": {
            "phash": "0" * 16,
            "size": [1000, 1000],
            "anchors": ["iOS 개발자", "보이저엑스"],
        },
        "recent_images": ["screen.png"],
        "marked_image": "marked.png",
        "recipe_params": {"target_count": 2},
        "extracted_jd": {},
    }

    result, _jd = worker_execution._dispatch_state(
        "set_result_card_queue",
        {"cards": [{"marker_id": 10, "title": "iOS 개발자", "company": "보이저엑스"}]},
        {},
        current_url="https://www.wanted.co.kr/search?query=ios",
        state=state,
    )

    assert result["status"] == "success"
    queue = result["_result_card_queue"]
    assert queue[0]["title"] == "iOS 개발자"
    assert queue[0]["company"] == "보이저엑스"
    assert queue[0]["bbox_ratio"] == [0.1, 0.2, 0.3, 0.24]
    assert "state_key" not in result["_result_page_memory"]


def test_set_result_card_queue_accepts_title_fallback():

    state = {
        "current_markers": [
            {"id": 10, "bbox": [100, 200, 300, 240], "text": "iOS 개발자"},
            {"id": 11, "bbox": [100, 260, 360, 300], "text": "Backend Engineer"},
            {"id": 12, "bbox": [100, 320, 360, 360], "text": "Android 개발자"},
        ],
        "screen_signature": {
            "phash": "0" * 16,
            "size": [1000, 1000],
            "anchors": ["iOS 개발자", "Backend Engineer", "Android 개발자"],
        },
        "recent_images": ["screen.png"],
        "marked_image": "marked.png",
        "recipe_params": {"target_count": 2},
        "extracted_jd": {},
    }

    result, _jd = worker_execution._dispatch_state(
        "set_result_card_queue",
        {
            "cards": 4,
            "titles": ["iOS 개발자", "Backend Engineer", "Android 개발자"],
            "companies": ["보이저엑스", "샘플"],
        },
        {},
        current_url="https://www.wanted.co.kr/search?query=ios",
        state=state,
    )

    assert result["status"] == "success"
    assert result["queued_titles"] == ["iOS 개발자", "Backend Engineer"]
    queue = result["_result_card_queue"]
    assert queue[0]["source_marker_id"] == 10
    assert queue[0]["bbox_ratio"] == [0.1, 0.2, 0.3, 0.24]
    assert queue[1]["source_marker_id"] == 11
    assert queue[1]["company"] == "샘플"


def test_set_result_card_queue_skips_title_without_visible_marker():

    state = {
        "current_markers": [
            {"id": 10, "bbox": [100, 200, 300, 240], "text": "iOS 개발자"},
        ],
        "screen_signature": {
            "phash": "0" * 16,
            "size": [1000, 1000],
            "anchors": ["iOS 개발자"],
        },
        "recipe_params": {"target_count": 1},
        "extracted_jd": {},
    }

    result, _jd = worker_execution._dispatch_state(
        "set_result_card_queue",
        {"cards": 1, "titles": ["화면에 없는 공고"]},
        {},
        current_url="https://www.wanted.co.kr/search?query=ios",
        state=state,
    )

    assert result["status"] == "skipped"
    assert result["queued_count"] == 0
    assert result["_result_card_queue"] == []


def test_existing_result_cards_match_exact_company_title_within_site(tmp_path):
    import sqlite3

    from agent.runtime.duplicate_job_policy import mark_existing_result_cards

    db_path = tmp_path / "jobs.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
                id INTEGER PRIMARY KEY,
                company_name TEXT,
                position TEXT,
                url TEXT,
                source_platform TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
            [
                (7, "에너자이", "ML Engineer Researcher", "https://www.saramin.co.kr/zf_user/search", "Saramin"),
                (8, "(주)에너자이", "ML Engineer/Researcher", "https://www.jobkorea.co.kr/Recruit/GI_Read/1", "JobKorea"),
            ],
        )

    queue, traces = mark_existing_result_cards(
        [
            {
                "queue_id": "card-1",
                "status": "pending",
                "company": "(주)에너자이",
                "title": "ML Engineer/Researcher",
            },
            {
                "queue_id": "card-2",
                "status": "pending",
                "company": "다른 회사",
                "title": "ML Engineer/Researcher",
            },
            {
                "queue_id": "card-3",
                "status": "pending",
                "company": "",
                "title": "ML Engineer/Researcher",
            },
        ],
        "https://www.saramin.co.kr/zf_user/search?searchword=ml",
        db_path=db_path,
    )

    assert queue[0]["status"] == "skipped"
    assert queue[0]["job_id"] == 7
    assert queue[1]["status"] == "pending"
    assert queue[2]["status"] == "pending"
    assert traces == [
        {
            "queue_id": "card-1",
            "company": "(주)에너자이",
            "title": "ML Engineer/Researcher",
            "job_id": 7,
        }
    ]


def test_action_node_finishes_when_selected_card_already_exists(monkeypatch):

    monkeypatch.setattr(
        worker_execution,
        "_mark_existing_result_cards",
        lambda queue, _url: (
            [{**queue[0], "status": "skipped", "job_id": 91}],
            [{"queue_id": queue[0]["queue_id"], "job_id": 91}],
        ),
    )

    result = worker_execution.action_node(
        {
            "goal": "사람인 머신러닝 엔지니어 공고 1개",
            "current_markers": [
                {
                    "id": 10,
                    "bbox": [100, 200, 400, 240],
                    "text": "ML Engineer/Researcher",
                }
            ],
            "current_url": "https://www.saramin.co.kr/zf_user/search?searchword=ml",
            "current_url_stale": False,
            "current_page_role": "search",
            "screen_signature": {"phash": "0" * 16, "size": [1000, 1000]},
            "recipe_params": {"target_count": 1, "count_mode": "explicit"},
            "extracted_jd": {},
            "is_finished": False,
            "collected_data": [],
            "error_count": 0,
            "recorded_steps": [],
            "pending_action": _action_request(
                content="",
                tool_calls=[
                    {
                        "name": "set_result_card_queue",
                        "args": {
                            "cards": [
                                {
                                    "marker_id": 10,
                                    "title": "ML Engineer/Researcher",
                                    "company": "(주)에너자이",
                                }
                            ]
                        },
                        "id": "queue-existing",
                    }
                ],
            ),
        }
    )

    assert result["is_finished"] is True
    assert result["result_card_queue"][0]["status"] == "skipped"
    assert result["result_card_queue"][0]["job_id"] == 91
    assert result["action_history"][0]["auto_finished"] is True


def test_card_queue_replay_after_go_back_uses_cached_bbox():

    state = {
        "result_card_queue": [
            {
                "queue_id": "card-2",
                "status": "pending",
                "title": "두번째 iOS 개발자",
                "company": "넛지",
                "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                "center_ratio": [0.4, 0.425],
                "target": {
                    "text": "두번째 iOS 개발자",
                    "semantic_label": "두번째 iOS 개발자",
                    "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                    "center_ratio": [0.4, 0.425],
                },
        }
    ],
    "result_page_memory": {
            "screen_signature": {
                "phash": "0" * 16,
                "anchors": ["두번째 iOS 개발자"],
                "size": [1000, 1000],
            },
        },
    }

    msg, markers, trace = queue_replay_after_return(
        state,
        {"action": "go_back"},
        "https://www.wanted.co.kr/search?query=ios",
        [],
        {"phash": "0" * 16, "anchors": ["두번째 iOS 개발자"], "size": [1000, 1000]},
    )

    assert msg is not None
    assert trace["hit"] is True
    call = msg.tool_calls[0]
    assert call.name == "click_marker"
    assert call.metadata["queue_id"] == "card-2"
    assert markers[0]["bbox"] == [300, 400, 500, 450]


def test_card_queue_replay_waits_until_active_card_is_done():

    state = {
        "active_result_card": {"queue_id": "card-1", "status": "active", "title": "첫번째 iOS 개발자"},
        "result_card_queue": [
            {"queue_id": "card-1", "status": "active", "title": "첫번째 iOS 개발자"},
            {
                "queue_id": "card-2",
                "status": "pending",
                "title": "두번째 iOS 개발자",
                "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
            },
        ],
        "result_page_memory": {
            "screen_signature": {"phash": "0" * 16, "anchors": ["두번째 iOS 개발자"], "size": [1000, 1000]},
        },
    }

    msg, _markers, trace = queue_replay_after_return(
        state,
        {"action": "go_back"},
        "https://www.wanted.co.kr/search?query=ios",
        [],
        {"phash": "0" * 16, "anchors": ["두번째 iOS 개발자"], "size": [1000, 1000]},
    )

    assert msg is None
    assert trace["reason"] == "active_card_not_completed"


def test_card_queue_marks_active_and_done():

    queue = [{"queue_id": "card-1", "status": "pending", "title": "A"}]

    queue, active = mark_result_card_active(queue, {"queue_id": "card-1"})
    assert queue[0]["status"] == "active"
    assert active["queue_id"] == "card-1"

    queue, active = complete_active_result_card(queue, active)
    assert queue[0]["status"] == "done"
    assert active == {}


def test_existing_job_url_trace_checks_current_run_and_database(tmp_path, monkeypatch):
    import sqlite3

    from agent.runtime.duplicate_job_policy import existing_job_url_trace

    monkeypatch.setenv("VISION_SKIP_EXISTING_JOB_DETAILS", "1")
    db_path = tmp_path / "jobs.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, url TEXT NOT NULL)")
        connection.execute("INSERT INTO jobs(url) VALUES (?)", ("https://www.wanted.co.kr/wd/2",))

    current_run = existing_job_url_trace(
        "https://www.wanted.co.kr/wd/1",
        {"공고목록": [{"url": "https://www.wanted.co.kr/wd/1"}]},
        db_path=db_path,
    )
    database = existing_job_url_trace(
        "https://www.wanted.co.kr/wd/2/",
        {},
        db_path=db_path,
    )
    missing = existing_job_url_trace(
        "https://www.wanted.co.kr/wd/3",
        {},
        db_path=db_path,
    )

    assert current_run["source"] == "current_run"
    assert database["source"] == "database"
    assert missing["matched"] is False


def test_perception_skips_existing_active_job_before_ocr(monkeypatch, tmp_path):
    from PIL import Image


    screenshot = tmp_path / "detail.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        last_capture_quality = {}

        def capture_screen(self):
            return screenshot

        def get_current_url(self):
            return "https://www.wanted.co.kr/wd/123"

        def analyze_ui(self, _path):
            raise AssertionError("중복 상세에서는 전체 OCR을 실행하면 안 됩니다.")

    monkeypatch.setattr(worker_observation, "_perception_engine", lambda: FakePerception())
    monkeypatch.setattr(
        worker_selection,
        "existing_job_url_trace",
        lambda _url, _extracted: {"matched": True, "source": "database", "job_id": 7},
    )
    monkeypatch.setattr(
        worker_observation,
        "raw_screen_phash_signature",
        lambda _path: {"phash": "a" * 16, "size": [800, 600]},
    )
    monkeypatch.setattr(worker_transition, "transition_has_visual_change", lambda *_args: (True, 0.5))

    result = worker_observation.observe_screen_cycle(
        {
            "current_url": "https://www.wanted.co.kr/search?query=ai",
            "current_url_stale": True,
            "pending_transition": {"action": "click_marker", "source": "card_queue"},
            "active_result_card": {"queue_id": "card-1", "title": "AI 엔지니어"},
            "result_card_queue": [
                {"queue_id": "card-1", "status": "active", "title": "AI 엔지니어"},
                {"queue_id": "card-2", "status": "pending", "title": "ML 엔지니어"},
            ],
            "extracted_jd": {},
        }
    )

    assert result["pending_action"].source == "duplicate_job_policy"
    assert result["page_policy_trace"]["policy"] == "skip_existing_job_detail"
    assert result["ocr_complete"] is False
    assert result["result_card_queue"][0]["status"] == "skipped"
    assert result["result_card_queue"][0]["job_id"] == 7
    assert result["active_result_card"] == {}
    assert result["pending_action"].tool_calls[0].name == "go_back"


def test_perception_finishes_when_last_visible_card_is_existing(monkeypatch, tmp_path):
    from PIL import Image


    screenshot = tmp_path / "last-duplicate.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        last_capture_quality = {}

        def capture_screen(self):
            return screenshot

        def get_current_url(self):
            return "https://www.wanted.co.kr/wd/123"

        def analyze_ui(self, _path):
            raise AssertionError("마지막 중복 상세에서는 전체 OCR을 실행하면 안 됩니다.")

    monkeypatch.setattr(worker_observation, "_perception_engine", lambda: FakePerception())
    monkeypatch.setattr(
        worker_selection,
        "existing_job_url_trace",
        lambda _url, _extracted: {"matched": True, "source": "database", "job_id": 7},
    )
    monkeypatch.setattr(
        worker_observation,
        "raw_screen_phash_signature",
        lambda _path: {"phash": "a" * 16, "size": [800, 600]},
    )
    monkeypatch.setattr(worker_transition, "transition_has_visual_change", lambda *_args: (True, 0.5))

    result = worker_observation.observe_screen_cycle(
        {
            "current_url": "https://www.wanted.co.kr/search?query=ai",
            "current_url_stale": True,
            "pending_transition": {"action": "click_marker", "source": "card_queue"},
            "active_result_card": {"queue_id": "card-1", "title": "AI 엔지니어"},
            "result_card_queue": [
                {"queue_id": "card-1", "status": "active", "title": "AI 엔지니어"},
            ],
            "recipe_params": {"target_count": 0, "count_mode": "visible_all"},
            "extracted_jd": {},
        }
    )

    assert result["page_policy_trace"]["policy"] == "finish_existing_job_queue"
    assert result["result_card_queue"][0]["status"] == "skipped"
    assert result["pending_action"].tool_calls[0].name == "finish_task"
    assert result["pending_action"].tool_calls[0].args["result"]


def test_perception_replays_next_card_after_go_back_without_ocr(monkeypatch, tmp_path):
    from PIL import Image


    screenshot = tmp_path / "returned-list.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        last_capture_quality = {}

        def capture_screen(self):
            return screenshot

        def get_current_url(self):
            return "https://www.wanted.co.kr/search?query=ai"

        def analyze_ui(self, _path):
            raise AssertionError("pHash가 일치한 목록 복귀에서는 전체 OCR을 실행하면 안 됩니다.")

    monkeypatch.setattr(worker_observation, "_perception_engine", lambda: FakePerception())
    monkeypatch.setattr(
        worker_observation,
        "raw_screen_phash_signature",
        lambda _path: {"phash": "0" * 16, "size": [800, 600]},
    )
    monkeypatch.setattr(worker_transition, "transition_has_visual_change", lambda *_args: (True, 0.5))

    result = worker_observation.observe_screen_cycle(
        {
            "current_url": "https://www.wanted.co.kr/wd/1",
            "current_url_stale": True,
            "pending_transition": {
                "action": "go_back",
                "source": "page_policy",
                "before_url": "https://www.wanted.co.kr/wd/1",
                "before_phash": "f" * 16,
            },
            "active_result_card": {},
            "result_card_queue": [
                {
                    "queue_id": "card-2",
                    "status": "pending",
                    "title": "두 번째 AI 엔지니어",
                    "bbox_ratio": [0.2, 0.3, 0.5, 0.4],
                    "center_ratio": [0.35, 0.35],
                },
            ],
            "result_page_memory": {
                "marked_image": "cached-marked.png",
                "screen_signature": {
                    "phash": "0" * 16,
                    "anchors": ["두 번째 AI 엔지니어"],
                    "size": [800, 600],
                },
            },
            "detail_ocr_buffer": {},
        }
    )

    assert result["pending_action"].source == "card_queue"
    assert result["ocr_complete"] is False
    assert result["transition_outcome"] == "queue_return_phash_match"
    assert result["pending_action"].tool_calls[0].metadata["queue_id"] == "card-2"
    assert result["current_markers"][0]["bbox"] == [160, 180, 400, 240]


def test_result_card_selector_can_refill_exhausted_queue():
    from agent.runtime.result_card_selector import should_select_result_cards

    state = {
        "current_page_role": "search",
        "recipe_params": {"target_count": 2},
        "extracted_jd": {},
        "result_card_queue": [
            {"queue_id": "card-1", "status": "skipped", "title": "이미 본 공고"},
        ],
        "active_result_card": {},
        "current_markers": [{"id": 1, "text": "새 공고"}],
        "marked_image": "marked.png",
    }

    assert should_select_result_cards(state) is True


def test_result_card_selector_stops_when_target_sized_queue_is_all_database_duplicates():
    from agent.runtime.result_card_selector import should_select_result_cards

    state = {
        "current_page_role": "search",
        "recipe_params": {"target_count": 2},
        "extracted_jd": {},
        "result_card_queue": [
            {
                "queue_id": "card-1",
                "status": "skipped",
                "title": "첫 번째 중복 공고",
                "job_id": 7,
            },
            {
                "queue_id": "card-2",
                "status": "skipped",
                "title": "두 번째 중복 공고",
                "job_id": 8,
            },
        ],
        "active_result_card": {},
        "current_markers": [{"id": 1, "text": "새 공고"}],
        "marked_image": "marked.png",
    }

    assert should_select_result_cards(state) is False


def test_perception_finishes_when_explicit_target_is_all_database_duplicates(monkeypatch, tmp_path):
    from PIL import Image


    screenshot = tmp_path / "explicit-last-duplicate.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        last_capture_quality = {}

        def capture_screen(self):
            return screenshot

        def get_current_url(self):
            return "https://www.wanted.co.kr/wd/456"

        def analyze_ui(self, _path):
            raise AssertionError("목표를 채운 중복 상세에서는 전체 OCR을 실행하면 안 됩니다.")

    monkeypatch.setattr(worker_observation, "_perception_engine", lambda: FakePerception())
    monkeypatch.setattr(
        worker_selection,
        "existing_job_url_trace",
        lambda _url, _extracted: {"matched": True, "source": "database", "job_id": 8},
    )
    monkeypatch.setattr(
        worker_observation,
        "raw_screen_phash_signature",
        lambda _path: {"phash": "a" * 16, "size": [800, 600]},
    )
    monkeypatch.setattr(worker_transition, "transition_has_visual_change", lambda *_args: (True, 0.5))

    result = worker_observation.observe_screen_cycle(
        {
            "current_url": "https://www.wanted.co.kr/search?query=ios",
            "current_url_stale": True,
            "pending_transition": {"action": "click_marker", "source": "card_queue"},
            "active_result_card": {"queue_id": "card-2", "title": "두 번째 iOS 개발자"},
            "result_card_queue": [
                {
                    "queue_id": "card-1",
                    "status": "skipped",
                    "title": "첫 번째 iOS 개발자",
                    "job_id": 7,
                },
                {"queue_id": "card-2", "status": "active", "title": "두 번째 iOS 개발자"},
            ],
            "recipe_params": {"target_count": 2, "count_mode": "explicit"},
            "extracted_jd": {},
        }
    )

    assert result["page_policy_trace"]["policy"] == "finish_existing_job_queue"
    assert result["result_card_queue"][1]["job_id"] == 8
    assert result["pending_action"].tool_calls[0].name == "finish_task"


def test_result_card_selector_does_not_refill_completed_visible_queue():
    from agent.runtime.result_card_selector import should_select_result_cards

    state = {
        "current_page_role": "search",
        "recipe_params": {"target_count": 0, "count_mode": "visible_all"},
        "extracted_jd": {},
        "result_card_queue": [
            {"queue_id": "card-1", "status": "skipped", "title": "이미 본 공고"},
            {"queue_id": "card-2", "status": "done", "title": "새로 수집한 공고"},
        ],
        "active_result_card": {},
        "current_markers": [{"id": 1, "text": "화면 아래 공고"}],
        "marked_image": "marked.png",
    }

    assert should_select_result_cards(state) is False


def test_card_queue_marks_active_when_card_click_uses_title_label():

    queue = [{"queue_id": "card-1", "status": "pending", "title": "iOS 핵심 시스템 CTO 및 PM급 엔지니어"}]
    args = {
        "marker_id": 170,
        "target_component": "job_card",
        "target_role": "link",
        "target_label": "iOS 핵심 시스템 CTO 및 PM급 엔지니어",
    }

    assert result_card_click_matches_queue(queue, args) is True
    queue, active = mark_result_card_active(queue, args)
    assert queue[0]["status"] == "active"
    assert active["queue_id"] == "card-1"


def test_recipe_store_commits_and_reads_by_recipe_key(tmp_path):
    from agent.recipe.store import RecipeStore

    db_path = tmp_path / "recipes.db"
    store = RecipeStore(db_path)
    saved = store.commit_recipe(
        "wanted.co.kr",
        "goal",
        [
            {
                "seq": 0,
                "decision_capture_id": "worker-run-1:attempt:00:capture:0001",
                "url_template": "wanted.co.kr",
                "page_role": "home",
                "action": "click_marker",
                "target": {"text": "검색", "region": "top-left", "ordinal": 0},
                "roi_signature": {"phash": "0" * 16, "target_center_ratio": [0.8, 0.1]},
                "param": {},
                "transition_contract": {
                    "common_ready_cues": [{"kind": "text_any", "values": ["검색 결과"]}],
                    "outcomes": [],
                },
            }
        ],
    )

    assert saved == 1
    rows = store.get_by_site("wanted.co.kr")
    assert "decision_capture_id" not in rows[0]["steps"][0]
    recipe = store.get_recipe(rows[0]["recipe_key"])
    assert recipe is not None
    assert recipe.steps[0].action == "click_marker"
    assert recipe.steps[0].transition_contract.common_ready_cues[0].values == ["검색 결과"]

    conn = sqlite3.connect(db_path)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(recipes)").fetchall()]
    conn.close()
    assert "recipe_key" in columns
    assert "metadata_json" in columns
    assert "state_key" not in columns


def test_recipe_key_ignores_layout_measurements_within_same_page_family():
    from agent.recipe.store import RecipeStore

    base_step = {
        "url_template": "wanted.co.kr",
        "page_role": "home",
        "action": "click_marker",
        "component": "search_button",
        "target_role": "button",
        "target": {"region": "top-right", "center_ratio": [0.8, 0.1]},
        "roi_signature": {"phash": "0" * 16},
    }
    moved_step = {
        **base_step,
        "page_role": "search_overlay",
        "target": {"region": "top-right", "center_ratio": [0.7, 0.1]},
        "roi_signature": {"phash": "f" * 16},
    }

    first = RecipeStore._recipe_key_for_step("wanted", base_step, {"task_category": "검색"})
    second = RecipeStore._recipe_key_for_step("wanted", moved_step, {"task_category": "검색"})

    assert first.startswith("roi2#")
    assert first == second


def test_recipe_store_recreates_old_state_key_schema_without_legacy_tables(tmp_path):
    from agent.recipe.store import RecipeStore

    db_path = tmp_path / "recipes.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE recipes (
            state_key TEXT PRIMARY KEY,
            site TEXT NOT NULL,
            goal TEXT,
            steps_json TEXT NOT NULL,
            success_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO recipes (
            state_key, site, goal, steps_json, success_count, created_at, updated_at
        ) VALUES ('old-state', 'wanted', 'old goal', '[]', 1, 'old', 'old')
        """
    )
    conn.execute("CREATE TABLE recipes_legacy_20260101000000 (state_key TEXT)")
    conn.commit()
    conn.close()

    RecipeStore(db_path)

    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    columns = [row[1] for row in conn.execute("PRAGMA table_info(recipes)").fetchall()]
    count = conn.execute("SELECT count(*) FROM recipes").fetchone()[0]
    conn.close()

    assert "recipes" in tables
    assert not any(name.startswith("recipes_legacy_") for name in tables)
    assert "recipe_key" in columns
    assert "state_key" not in columns
    assert count == 0


def test_recipe_store_rejects_target_steps_without_page_role(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "recipes.db")
    saved = store.commit_recipe(
        "wanted",
        "goal",
        [
            {
                "seq": 0,
                "action": "click_marker",
                "target": {"text": "검색", "center_ratio": [0.8, 0.1]},
                "roi_signature": {"phash": "0" * 16, "target_center_ratio": [0.8, 0.1]},
            }
        ],
        metadata={"task_category": "검색"},
    )

    assert saved == 0
    assert store.get_by_site("wanted") == []


def test_recipe_store_stores_each_roi_step_as_recipe(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "recipes.db")
    store.commit_recipe(
        "wanted.co.kr",
        "goal",
        [
            {
                "seq": 0,
                "page_role": "home",
                "action": "type_in_marker",
                "target": {"text": "검색", "region": "top-left", "ordinal": 0},
                "roi_signature": {"phash": "1" * 16, "target_center_ratio": [0.2, 0.2]},
                "param": {"text": "ai 엔지니어"},
            },
            {
                "seq": 1,
                "page_role": "home",
                "action": "click_marker",
                "target": {"text": "공고", "region": "middle-left", "ordinal": 0},
                "roi_signature": {"phash": "2" * 16, "target_center_ratio": [0.4, 0.4]},
                "param": {},
            },
        ],
    )

    recipes = store.get_site_recipes("wanted.co.kr")

    assert len(recipes) == 2
    assert sorted(recipe.steps[0].action for _key, recipe in recipes) == ["click_marker", "type_in_marker"]


def test_recipe_store_filters_recipe_by_site_and_task_category(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "recipes.db")
    store.commit_recipe(
        "wanted",
        "goal",
        [
            {
                "seq": 0,
                "page_role": "home",
                "action": "click_marker",
                "target": {"text": "검색", "center_ratio": [0.8, 0.1]},
                "roi_signature": {"phash": "0" * 16, "target_center_ratio": [0.8, 0.1]},
            }
        ],
        metadata={"task_category": "검색"},
    )

    assert len(store.get_site_recipes("wanted", task_category="검색")) == 1
    assert store.get_site_recipes("wanted", task_category="로그인") == []


def test_reflex_node_builds_action_tool_call(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    saved = tmp_path / "saved-search.png"
    current = tmp_path / "current-search.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([10, 10, 70, 40], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [10, 10, 70, 40], [200, 120])

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            return [
                (
                    "recipe-search-input",
                    SiteRecipe(
                        site="wanted",
                        goal="goal",
                        steps=[
                            RecipeStep(
                                seq=0,
                                action="type_in_marker",
                                page_role="home",
                                replay_mode="parameterized",
                                roi_signature=roi_signature,
                                target={
                                    "text": "검색",
                                    "region": "top-left",
                                    "ordinal": 0,
                                    "bbox_ratio": [0.05, 0.0833, 0.35, 0.3333],
                                    "center_ratio": [0.2, 0.2083],
                                },
                                param={"text": "ai 엔지니어"},
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = reflex_node(
        {
            "goal": "ai 엔지니어 공고 찾아줘",
            "current_url": "https://www.wanted.co.kr",
            "current_page_role": "home",
            "screen_signature": {"size": [200, 120]},
            "recent_images": [current],
            "current_markers": [{"id": 7, "bbox": [10, 10, 70, 40], "text": "검색"}],
            "recipe_params": {"site": "wanted"},
        }
    )

    msg = result["pending_action"]
    assert result["reflex_trace"]["hit"] is True
    assert msg.tool_calls[0].name == "type_in_marker"
    assert msg.tool_calls[0].args == {"marker_id": 7, "text": "ai 엔지니어", "page_role": "home"}
    assert len(msg.tool_calls) == 1


def test_reflex_node_appends_enter_for_search_input_compound(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    saved = tmp_path / "saved-search-input.png"
    current = tmp_path / "current-search-input.png"
    for path in [saved, current]:
        image = Image.new("RGB", (240, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([20, 20, 190, 50], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [20, 20, 190, 50], [240, 120])

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            return [
                (
                    "recipe-search-input",
                    SiteRecipe(
                        site="wanted",
                        goal="goal",
                        steps=[
                            RecipeStep(
                                seq=0,
                                action="type_in_marker",
                                page_role="search_overlay",
                                replay_mode="parameterized",
                                roi_signature=roi_signature,
                                target={
                                    "text": "검색어를 입력해주세요",
                                    "bbox_ratio": [0.0833, 0.1667, 0.7917, 0.4167],
                                    "center_ratio": [0.4375, 0.2917],
                                },
                                param={"text": "old query", "slot_name": "query"},
                                slot_refs=["query"],
                                target_role="search_input",
                                component="search_input_field",
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = reflex_node(
        {
            "goal": "ios 개발자 공고 2개",
            "current_url": "https://www.wanted.co.kr",
            "current_page_role": "search_overlay",
            "screen_signature": {"size": [240, 120]},
            "recent_images": [current],
            "current_markers": [{"id": 24, "bbox": [20, 20, 190, 50], "text": "검색어를 입력해주세요"}],
            "recipe_params": {"site": "wanted", "query": "ios 개발자", "task_category": "검색"},
        }
    )

    msg = result["pending_action"]
    assert result["reflex_trace"]["hit"] is True
    assert [call.name for call in msg.tool_calls] == ["type_in_marker", "press_key"]
    assert msg.tool_calls[0].args["text"] == "ios 개발자"
    assert msg.tool_calls[1].args["key"] == "enter"
    assert msg.tool_calls[1].metadata["transition_source"] == "reflex_compound"
    assert result["reflex_trace"]["actions"] == ["type_in_marker", "press_key"]


def test_reflex_node_uses_roi_signature_when_available(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [150, 20, 170, 40], [200, 200])

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            return [
                (
                    "recipe-search-button",
                    SiteRecipe(
                        site="wanted",
                        goal="goal",
                        steps=[
                            RecipeStep(
                                seq=0,
                                action="click_marker",
                                page_role="home",
                                replay_mode="fixed",
                                roi_signature=roi_signature,
                                target={
                                    "text": "검색",
                                    "bbox_ratio": [0.75, 0.1, 0.85, 0.2],
                                    "center_ratio": [0.8, 0.15],
                                },
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = reflex_node(
        {
            "goal": "ai 엔지니어 공고 찾아줘",
            "current_url": "https://www.wanted.co.kr",
            "current_page_role": "home",
            "screen_signature": {
                "phash": "0" * 16,
                "size": [200, 200],
                "anchors": ["검색"],
            },
            "recent_images": [current],
            "current_markers": [{"id": 77, "bbox": [150, 20, 170, 40], "text": "검색"}],
            "recipe_params": {"site": "wanted"},
        }
    )

    assert result["reflex_trace"]["hit"] is True
    assert result["pending_action"].tool_calls[0].args == {"marker_id": 77, "page_role": "home"}
    assert result["reflex_trace"]["hit"] is True
    call_id = result["pending_action"].tool_calls[0].id
    assert result["reflex_trace"]["tool_calls"][call_id]["match_mode"] == "roi_phash"
    assert result["reflex_trace"]["tool_calls"][call_id]["phash"]["distance"] == 0


def test_reflex_node_rejects_page_role_mismatch(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [150, 20, 170, 40], [200, 200])

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            return [
                (
                    "recipe-search-button",
                    SiteRecipe(
                        site="wanted",
                        goal="goal",
                        steps=[
                            RecipeStep(
                                seq=0,
                                action="click_marker",
                                page_role="home",
                                replay_mode="fixed",
                                roi_signature=roi_signature,
                                target={"text": "검색", "center_ratio": [0.8, 0.15]},
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = reflex_node(
        {
            "goal": "ai 엔지니어 공고 찾아줘",
            "current_page_role": "job_detail",
            "screen_signature": {"size": [200, 200]},
            "recent_images": [current],
            "current_markers": [{"id": 77, "bbox": [150, 20, 170, 40], "text": "검색"}],
            "recipe_params": {"site": "wanted"},
        }
    )

    assert result["reflex_trace"]["hit"] is False
    assert result["reflex_trace"]["last_reason"] == "page_role_mismatch"


def test_reflex_node_rejects_recipe_from_different_url_scope(monkeypatch):
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            return [
                (
                    "recipe-social-search",
                    SiteRecipe(
                        site="wanted",
                        steps=[
                            RecipeStep(
                                seq=0,
                                url_template="social.wanted.co.kr/community",
                                action="click_marker",
                                page_role="home",
                                replay_mode="fixed",
                                roi_signature={"phash": "0" * 16, "crop_rect_ratio": [0, 0, 1, 1]},
                                target={"text": "검색", "center_ratio": [0.8, 0.15]},
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())
    monkeypatch.setattr(
        "agent.recipe.phash_replay.match_step_by_screen_signature",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("URL 범위 검사보다 먼저 ROI를 검사함")),
    )

    result = reflex_node(
        {
            "goal": "ios 개발자 공고 2개",
            "current_url": "https://www.wanted.co.kr/",
            "current_page_role": "home",
            "screen_signature": {"size": [200, 200]},
            "recent_images": ["screen.png"],
            "current_markers": [{"id": 7, "bbox": [150, 20, 170, 40], "text": "검색"}],
            "recipe_params": {"site": "wanted", "task_category": "검색"},
        }
    )

    assert result["reflex_trace"]["hit"] is False
    assert result["reflex_trace"]["last_reason"] == "url_scope_mismatch"
    assert result["reflex_trace"]["candidate_rejections"][0]["current_url_template"] == "wanted.co.kr/"


def test_search_button_transition_accepts_verified_visual_change():
    from agent.runtime.transition_runtime import transition_accepts_visual_change

    assert transition_accepts_visual_change(
        {
            "source": "reflex",
            "action": "click_marker",
            "step": {"component": "search_button"},
        }
    ) is True


def test_reflex_node_skips_blocked_recipe_key(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [150, 20, 170, 40], [200, 200])

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            return [
                (
                    "recipe-search-button",
                    SiteRecipe(
                        site="wanted",
                        goal="goal",
                        steps=[
                            RecipeStep(
                                seq=0,
                                action="click_marker",
                                page_role="home",
                                replay_mode="fixed",
                                roi_signature=roi_signature,
                                target={"text": "검색", "center_ratio": [0.8, 0.15]},
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = reflex_node(
        {
            "goal": "ai 엔지니어 공고 찾아줘",
            "current_page_role": "home",
            "screen_signature": {"size": [200, 200]},
            "recent_images": [current],
            "current_markers": [{"id": 77, "bbox": [150, 20, 170, 40], "text": "검색"}],
            "recipe_params": {"site": "wanted"},
            "reflex_blocked_recipe_keys": ["recipe-search-button"],
        }
    )

    assert result["reflex_trace"]["hit"] is False
    assert result["reflex_trace"]["last_reason"] == "recipe_blocked_after_transition_failure"


def test_reflex_node_skips_idempotent_recipe_used_on_same_url(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    saved = tmp_path / "saved-tab.png"
    current = tmp_path / "current-tab.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([80, 50, 130, 80], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [80, 50, 130, 80], [200, 200])

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            return [
                (
                    "recipe-tab",
                    SiteRecipe(
                        site="wanted",
                        steps=[
                            RecipeStep(
                                seq=0,
                                action="click_marker",
                                page_role="search",
                                replay_mode="fixed",
                                roi_signature=roi_signature,
                                target={"text": "포지션", "center_ratio": [0.525, 0.325]},
                                component="tab_button",
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = reflex_node(
        {
            "goal": "ios 개발자 공고 2개",
            "current_url": "https://www.wanted.co.kr/search?query=ios&tab=position",
            "current_page_role": "search",
            "screen_signature": {"size": [200, 200]},
            "recent_images": [current],
            "current_markers": [{"id": 9, "bbox": [80, 50, 130, 80], "text": "포지션"}],
            "recipe_params": {"site": "wanted", "task_category": "검색"},
            "action_history": [
                {
                    "status": "success",
                    "action": "click_marker",
                    "before_url": "https://www.wanted.co.kr/search?query=ios",
                    "reflex_recipe_key": "recipe-tab",
                    "args": {"target_component": "tab_button"},
                }
            ],
        }
    )

    assert result["reflex_trace"]["hit"] is False
    assert result["reflex_trace"]["last_reason"] == "recipe_already_used_on_page"


def test_reflex_node_rejects_signed_step_when_roi_missing(monkeypatch):
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            return [
                (
                    "recipe-without-roi",
                    SiteRecipe(
                        site="wanted",
                        goal="goal",
                        steps=[
                            RecipeStep(
                                seq=0,
                                action="click_marker",
                                page_role="home",
                                replay_mode="fixed",
                                target={"text": "검색", "center_ratio": [0.81, 0.10]},
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = reflex_node(
        {
            "goal": "ai 엔지니어 공고 찾아줘",
            "current_url": "https://www.wanted.co.kr",
            "current_page_role": "home",
            "screen_signature": {
                "phash": "0000000000000000",
                "size": [1000, 1000],
                "anchors": ["검색"],
            },
            "current_markers": [{"id": 77, "bbox": [790, 80, 830, 120], "text": "검색"}],
            "recipe_params": {"site": "wanted"},
        }
    )

    assert result["reflex_trace"]["hit"] is False
    assert result["reflex_trace"]["reason"] == "no_candidate_passed"
    assert result["reflex_trace"]["last_reason"] == "roi_signature_missing"


def test_reflex_node_replaces_type_input_slot(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe
    from shared.schema.skill_schema import RecipeSkillMetadata, SkillInputSlot

    saved = tmp_path / "slot-saved.png"
    current = tmp_path / "slot-current.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([10, 10, 70, 40], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [10, 10, 70, 40], [200, 120])

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            return [
                (
                    "recipe-slot-input",
                    SiteRecipe(
                        site="wanted",
                        goal="old goal",
                        skill_metadata=RecipeSkillMetadata(
                            inputs=[SkillInputSlot(name="query", required=True)]
                        ),
                        steps=[
                            RecipeStep(
                                seq=0,
                                action="type_in_marker",
                                page_role="home",
                                replay_mode="parameterized",
                                roi_signature=roi_signature,
                                target={
                                    "text": "Search",
                                    "region": "top-left",
                                    "ordinal": 0,
                                    "bbox_ratio": [0.05, 0.0833, 0.35, 0.3333],
                                    "center_ratio": [0.2, 0.2083],
                                },
                                param={"text": "old query", "slot_name": "query"},
                                slot_refs=["query"],
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = reflex_node(
        {
            "goal": "find android jobs",
            "current_url": "https://www.wanted.co.kr",
            "current_page_role": "home",
            "screen_signature": {"size": [200, 120]},
            "recent_images": [current],
            "current_markers": [{"id": 7, "bbox": [10, 10, 70, 40], "text": "Search"}],
            "recipe_params": {"site": "wanted", "query": "android developer"},
        }
    )

    assert result["reflex_trace"]["hit"] is True
    assert result["pending_action"].tool_calls[0].args == {
        "marker_id": 7,
        "text": "android developer",
        "page_role": "home",
        "slot_name": "query",
    }


def test_reflex_node_uses_site_candidates_without_state_lookup(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe
    current = tmp_path / "site-current.png"
    saved = tmp_path / "site-saved.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [150, 20, 170, 40], [200, 200])

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            assert site == "wanted"
            assert task_category == "검색"
            return [
                (
                    "recipe-search",
                    SiteRecipe(
                        site="wanted",
                        steps=[
                            RecipeStep(
                                seq=0,
                                action="click_marker",
                                page_role="home",
                                replay_mode="fixed",
                                roi_signature=roi_signature,
                                target={"text": "검색", "center_ratio": [0.8, 0.15]},
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = reflex_node(
        {
            "goal": "android 개발자 공고 찾아줘",
            "current_page_role": "home",
            "screen_signature": {"size": [200, 200]},
            "recent_images": [current],
            "current_markers": [{"id": 7, "bbox": [150, 20, 170, 40], "text": "검색"}],
            "recipe_params": {"site": "wanted", "query": "android 개발자", "task_category": "검색"},
        }
    )

    assert result["reflex_trace"]["hit"] is True
    assert result["pending_action"].tool_calls[0].args == {"marker_id": 7, "page_role": "home"}


def test_reflex_node_skips_recipe_when_task_category_mismatches(monkeypatch):

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            assert task_category == "검색"
            return []

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = reflex_node(
        {
            "goal": "android 개발자 공고 찾아줘",
            "screen_signature": {"size": [200, 200]},
            "current_markers": [{"id": 7, "bbox": [150, 20, 170, 40], "text": "검색"}],
            "recipe_params": {"site": "wanted", "query": "android 개발자", "task_category": "검색"},
        }
    )

    assert result["reflex_trace"]["hit"] is False
    assert result["reflex_trace"]["reason"] == "no_recipe"
    assert result["reflex_trace"]["candidate_count"] == 0


def test_action_node_commits_accumulated_recorded_steps(monkeypatch):
    seen = {}

    class FakeTools:
        def finish_task(self, result):
            return {"status": "success", "action": "finish_task", "result": result}

    monkeypatch.setattr(worker_execution, "_get_action_tools", lambda: FakeTools())
    monkeypatch.setattr(
        worker_recording,
        "commit_if_finished",
        lambda steps, state, current_url: seen.update(steps=steps, current_url=current_url),
    )

    prior_steps = [
        {
            "seq": 0,
            "state_key": "state-a",
            "url_template": "wanted.co.kr",
            "action": "click_marker",
            "target": {"text": "검색", "region": "top-left", "ordinal": 0},
            "param": {},
            "transition_contract": None,
        }
    ]

    result = worker_execution.action_node(
        {
            "current_markers": [],
            "current_url": "https://www.wanted.co.kr",
            "current_url_stale": False,
            "extracted_jd": {},
            "is_finished": False,
            "collected_data": [],
            "error_count": 0,
            "recorded_steps": prior_steps,
            "pending_action": _action_request(
                content="",
                tool_calls=[{"name": "finish_task", "args": {"result": "done"}, "id": "1"}],
            ),
        }
    )
    worker_recording.record_execution_node(
        {
            **result,
            "recorded_steps": prior_steps,
        }
    )

    assert result["is_finished"] is True
    assert seen["steps"] == prior_steps
    assert seen["current_url"] == "https://www.wanted.co.kr"


def test_action_node_skips_stale_reasoning_screen_click(monkeypatch):

    dispatched = []

    class FakeTools:
        def click_marker(self, bbox):
            dispatched.append(bbox)
            return {"status": "success", "action": "click_marker", "result": "clicked"}

    monkeypatch.setattr(worker_execution, "_get_action_tools", lambda: FakeTools())
    monkeypatch.setattr(
        worker_execution,
        "_check_current_reasoning_screen",
        lambda state: {
            "checked": True,
            "stale": True,
            "reason": "screen_changed_during_reasoning",
            "distance": 22,
            "max_distance": 10,
        },
    )

    result = worker_execution.action_node(
        {
            "goal": "ios 개발자 공고 2개",
            "current_markers": [{"id": 7, "bbox": [500, 500, 560, 540], "text": "포지션"}],
            "current_url": "https://www.wanted.co.kr/search?query=ios",
            "current_url_stale": False,
            "current_page_role": "search",
            "screen_signature": {"phash": "0" * 16, "size": [1000, 1000]},
            "pending_action": _action_request(
                content="",
                tool_calls=[{"name": "click_marker", "args": {"marker_id": 7}, "id": "stale-click"}],
            ),
        }
    )

    action = result["action_history"][0]
    assert dispatched == []
    assert action["status"] == "skipped"
    assert action["reason"] == "screen_changed_during_reasoning"
    assert action["guard"]["distance"] == 22
    assert result["current_url_stale"] is True


def test_action_node_bundles_reflex_action_as_transition_step(monkeypatch):

    class FakeTools:
        def click_marker(self, bbox):
            return {"status": "success", "action": "click_marker", "result": f"clicked {bbox}"}

    monkeypatch.setattr(worker_execution, "_get_action_tools", lambda: FakeTools())

    result = worker_execution.action_node(
        {
            "goal": "ios 개발자 공고 2개",
            "current_markers": [{"id": 7, "bbox": [500, 500, 560, 540], "text": "포지션"}],
            "current_url": "https://www.wanted.co.kr/search?query=ios",
            "current_url_stale": False,
            "current_page_role": "search",
            "screen_signature": {"phash": "0" * 16, "size": [1000, 1000]},
            "extracted_jd": {},
            "is_finished": False,
            "collected_data": [],
            "error_count": 0,
            "recorded_steps": [],
            "reflex_trace": {
                "recipe_key": "roi#tab",
                "tool_calls": {
                    "reflex_call": {
                        "seq": 2,
                        "replay_mode": "fixed",
                        "match_mode": "roi_phash",
                        "target_text": "포지션",
                        "marker_id": 7,
                        "phash": {"distance": 0, "mode": "roi_phash"},
                    }
                },
            },
            "reflex_transition_contracts": {
                "reflex_call": {"common_ready_cues": [{"kind": "text_any", "values": ["포지션"]}]}
            },
            "pending_action": _action_request(
                content="[reflex]",
                source="reflex",
                tool_calls=[
                    {
                        "name": "click_marker",
                        "args": {"marker_id": 7, "page_role": "search", "target_component": "tab_button"},
                        "id": "reflex_call",
                    }
                ],
            ),
        }
    )

    pending = result["pending_transition"]
    assert pending["action"] == "click_marker"
    assert pending["recipe_key"] == "roi#tab"
    assert pending["step"]["action"] == "click_marker"
    assert pending["step"]["recipe_key"] == "roi#tab"
    assert pending["step"]["target_text"] == "포지션"
    assert pending["step"]["phash"]["distance"] == 0


def test_reflex_routing_respects_flag_and_validation(monkeypatch):
    from agent.config import clear_settings_cache
    from agent.graph.workflow import route_after_reflex, route_after_selection, route_after_start

    monkeypatch.delenv("REFLEX_ENABLED", raising=False)
    clear_settings_cache()
    observed = {"ocr_complete": True}
    assert route_after_selection(observed) == "reflex"
    assert route_after_start(observed) == "selection"
    assert route_after_start({}) == "reasoning"
    assert route_after_selection({"low_information_screen": True}) == "reasoning"
    assert route_after_start({"low_information_screen": True}) == "reasoning"

    monkeypatch.setenv("REFLEX_ENABLED", "0")
    clear_settings_cache()
    assert route_after_selection(observed) == "reasoning"
    assert route_after_start(observed) == "selection"

    monkeypatch.setenv("REFLEX_ENABLED", "1")
    clear_settings_cache()
    assert route_after_selection(observed) == "reflex"
    assert route_after_selection(
        {"ocr_complete": True, "transition_status": "pending"}
    ) == "capture"
    assert route_after_selection(
        {"ocr_complete": False, "ocr_required": True}
    ) == "ocr"
    queued = _action_request(
        source="card_queue",
        tool_calls=[{"name": "click_marker", "args": {"marker_id": 1}, "id": "queue"}],
    )
    assert route_after_selection({"pending_action": queued}) == "action"

    reflex_request = _action_request(
        source="reflex",
        tool_calls=[{"name": "press_key", "args": {"key": "enter"}, "id": "reflex"}],
    )
    assert route_after_reflex({"pending_action": reflex_request}) == "action"
    assert route_after_reflex({}) == "reasoning"


def test_detail_ui_context_compacts_ocr_markers_into_ordered_lines(monkeypatch):

    monkeypatch.setenv("VISION_DETAIL_SECTION_MIN_TEXT_MARKERS", "1")
    markers = [
        {"id": 1, "bbox": [80, 80, 130, 110], "text": "채용"},
        {"id": 2, "bbox": [100, 200, 210, 235], "text": "모바일"},
        {"id": 3, "bbox": [220, 200, 360, 235], "text": "엔지니어"},
        {"id": 10, "bbox": [100, 400, 210, 435], "text": "주요업무"},
        {"id": 11, "bbox": [100, 450, 135, 480], "text": "iOS"},
        {"id": 12, "bbox": [150, 450, 180, 480], "text": "앱"},
        {"id": 13, "bbox": [195, 450, 250, 480], "text": "개발"},
        {"id": 20, "bbox": [100, 540, 220, 575], "text": "자격요건"},
        {"id": 21, "bbox": [100, 590, 170, 620], "text": "Swift"},
        {"id": 22, "bbox": [190, 590, 250, 620], "text": "경험"},
        {"id": 30, "bbox": [100, 700, 290, 735], "text": "상세 정보 더 보기"},
        {"id": 31, "bbox": [820, 210, 870, 260], "text": "상호작용 가능한 요소 (button)"},
    ]

    context = build_ui_context(markers, current_url="https://www.wanted.co.kr/wd/1")

    assert "상세 페이지 OCR 본문" in context
    assert "식별된 텍스트 요소" not in context
    assert "[공고 소개]" not in context
    assert "모바일 엔지니어" in context
    assert "[주요업무]" not in context
    assert "주요업무" in context
    assert "iOS 앱 개발" in context
    assert "[자격요건]" not in context
    assert "자격요건" in context
    assert "Swift 경험" in context
    assert "상세 정보 더 보기" in context
    assert "수집 진행용 클릭 후보" not in context
    assert "[id: 30]" not in context
    assert "1. 채용" in context


def test_detail_ui_context_keeps_heading_lines_for_llm_judgment(monkeypatch):

    monkeypatch.setenv("VISION_DETAIL_SECTION_MIN_TEXT_MARKERS", "1")
    markers = [
        {"id": 1, "bbox": [100, 200, 210, 235], "text": "기술스택"},
        {"id": 2, "bbox": [100, 250, 170, 280], "text": "Flutter"},
        {"id": 3, "bbox": [190, 250, 260, 280], "text": "WebRTC"},
        {"id": 4, "bbox": [100, 340, 160, 375], "text": "태그"},
        {"id": 5, "bbox": [100, 390, 190, 420], "text": "식대지원"},
        {"id": 6, "bbox": [210, 390, 320, 420], "text": "장비지원"},
    ]

    context = build_ui_context(markers, current_url="https://www.wanted.co.kr/wd/1")

    assert "[기술스택]" not in context
    assert "[태그/혜택]" not in context
    assert "기술스택" in context
    assert "태그" in context
    assert "Flutter WebRTC" in context
    assert "식대지원 장비지원" in context


def test_non_detail_ui_context_keeps_raw_marker_list(monkeypatch):

    monkeypatch.setenv("VISION_DETAIL_SECTION_MIN_TEXT_MARKERS", "1")
    markers = [
        {"id": 1, "bbox": [100, 200, 180, 230], "text": "검색"},
        {"id": 2, "bbox": [100, 260, 220, 290], "text": "iOS 개발자"},
    ]

    context = build_ui_context(markers, current_url="https://www.wanted.co.kr/search?query=ios")

    assert "상세 페이지 OCR 섹션 요약" not in context
    assert "식별된 텍스트 요소" in context
    assert "[id: 2] iOS 개발자" in context


def test_detail_lightweight_marked_image_draws_action_candidates(tmp_path, monkeypatch):
    from PIL import Image

    from agent.runtime.detail_runtime import build_detail_lightweight_marked_image

    monkeypatch.setenv("VISION_DETAIL_LIGHTWEIGHT_MARKED_IMAGE_ENABLED", "1")
    image_path = tmp_path / "screen_detail.png"
    Image.new("RGB", (420, 360), "white").save(image_path)
    markers = [
        {"id": 1, "bbox": [40, 40, 180, 80], "text": "채용"},
        {"id": 30, "bbox": [100, 270, 260, 310], "text": "상세 정보 더 보기"},
    ]

    output_path = build_detail_lightweight_marked_image(
        image_path,
        markers,
        "https://www.wanted.co.kr/wd/1",
    )

    assert output_path
    assert output_path.endswith(".jpg")
    assert (tmp_path / "light_marked_screen_detail.jpg").exists()


def test_detail_lightweight_marked_image_skips_non_detail_page(tmp_path, monkeypatch):
    from PIL import Image

    from agent.runtime.detail_runtime import build_detail_lightweight_marked_image

    monkeypatch.setenv("VISION_DETAIL_LIGHTWEIGHT_MARKED_IMAGE_ENABLED", "1")
    image_path = tmp_path / "screen_search.png"
    Image.new("RGB", (420, 360), "white").save(image_path)

    output_path = build_detail_lightweight_marked_image(
        image_path,
        [{"id": 30, "bbox": [100, 270, 260, 310], "text": "상세 정보 더 보기"}],
        "https://www.wanted.co.kr/search?query=ios",
    )

    assert output_path == ""


def test_detail_ocr_buffer_accumulates_unique_detail_lines(monkeypatch):
    from agent.runtime.detail_runtime import update_detail_ocr_buffer

    monkeypatch.setenv("VISION_DETAIL_OCR_BUFFER_ENABLED", "1")
    markers = [
        {"id": 1, "bbox": [100, 200, 210, 235], "text": "주요업무"},
        {"id": 2, "bbox": [100, 250, 160, 280], "text": "iOS"},
        {"id": 3, "bbox": [180, 250, 250, 280], "text": "개발"},
        {"id": 4, "bbox": [100, 320, 180, 350], "text": "자격요건"},
        {"id": 5, "bbox": [100, 370, 180, 400], "text": "Swift"},
    ]

    first = update_detail_ocr_buffer(
        {},
        markers,
        "https://www.wanted.co.kr/wd/1",
        "screen_a.png",
    )
    second = update_detail_ocr_buffer(
        first,
        markers,
        "https://www.wanted.co.kr/wd/1",
        "screen_b.png",
    )

    assert first["stats"]["added_lines_last_screen"] == 4
    assert len(first["lines"]) == 4
    assert second["stats"]["added_lines_last_screen"] == 0
    assert second["stats"]["duplicate_lines_last_screen"] == 4
    assert len(second["lines"]) == 4


def test_detail_ocr_buffer_resets_for_another_card_on_same_url(monkeypatch):
    from agent.runtime.detail_runtime import update_detail_ocr_buffer

    monkeypatch.setenv("VISION_DETAIL_OCR_BUFFER_ENABLED", "1")
    url = "https://www.rocketpunch.com/jobs"
    first = update_detail_ocr_buffer(
        {},
        [{"id": 1, "bbox": [100, 220, 260, 250], "text": "첫 번째 공고 업무"}],
        url,
        "screen_a.png",
        page_role="side_panel_detail",
        detail_key="card-a",
    )
    second = update_detail_ocr_buffer(
        first,
        [{"id": 2, "bbox": [100, 220, 260, 250], "text": "두 번째 공고 업무"}],
        url,
        "screen_b.png",
        page_role="side_panel_detail",
        detail_key="card-b",
    )

    assert second["detail_key"] == "card-b"
    assert [line["text"] for line in second["lines"]] == ["두 번째 공고 업무"]
    assert second["stats"]["screen_count"] == 1


def test_detail_ocr_buffer_context_guides_finish_detail_reading(monkeypatch):

    monkeypatch.setenv("VISION_DETAIL_OCR_BUFFER_ENABLED", "1")
    state = {
        "detail_ocr_buffer": {
            "url": "https://www.wanted.co.kr/wd/1",
            "lines": [{"text": "주요업무 iOS 개발"}, {"text": "자격요건 Swift"}],
            "stats": {
                "screen_count": 2,
                "added_lines_last_screen": 1,
                "duplicate_lines_last_screen": 1,
            },
        }
    }

    context = worker_reasoning._compact_detail_ocr_buffer_context(state, "https://www.wanted.co.kr/wd/1")

    assert "상세 OCR 누적 상태" in context
    assert "finish_detail_reading" in context
    assert "중간 DB 추출" in context


def test_finish_detail_reading_merges_buffer_extraction_and_clears_buffer(monkeypatch):

    monkeypatch.setattr(
        worker_execution,
        "_extract_job_from_detail_ocr_buffer",
        lambda state, current_url: {
            "company_name": "보이저엑스",
            "position": "iOS 개발자",
            "url": current_url,
            "requirements": ["Swift"],
        },
    )

    result, current_jd = worker_execution._dispatch_state(
        "finish_detail_reading",
        {"page_role": "job_detail", "detail_complete": True},
        {},
        current_url="https://www.wanted.co.kr/wd/1",
        state={
            "detail_ocr_buffer": {
                "url": "https://www.wanted.co.kr/wd/1",
                "lines": [{"text": "자격요건 Swift"}],
            }
        },
    )

    assert result["status"] == "success"
    assert result["_detail_ocr_buffer"] == {}
    assert current_jd["공고목록"][0]["position"] == "iOS 개발자"


def test_finish_detail_reading_keeps_buffer_when_actual_job_content_is_missing(monkeypatch):

    url = "https://example.com/jobs/intermediary"
    buffer = {
        "url": url,
        "lines": [
            {"text": "중계회사"},
            {"text": "백엔드 개발자"},
            {"text": "원문 공고로 이동"},
        ],
    }
    monkeypatch.setattr(
        worker_execution,
        "_extract_job_from_detail_ocr_buffer",
        lambda _state, current_url: {
            "company_name": "중계회사",
            "position": "백엔드 개발자",
            "url": current_url,
        },
    )

    result, current_jd = worker_execution._dispatch_state(
        "finish_detail_reading",
        {"page_role": "job_detail", "detail_complete": True},
        {},
        current_url=url,
        state={"detail_ocr_buffer": buffer},
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "detail_content_incomplete"
    assert result["_detail_ocr_buffer"] == buffer
    assert result["_detail_followup_required"]["url"] == url
    assert current_jd == {}


def test_detail_followup_routes_to_reasoning_before_reflex():
    from agent.graph.workflow import route_after_selection

    result = route_after_selection(
        {
            "detail_followup_required": {
                "url": "https://example.com/jobs/intermediary",
                "reason": "detail_content_incomplete",
            }
        }
    )

    assert result == "reasoning"


def test_detail_followup_waits_for_required_ocr_before_reasoning():
    from agent.graph.workflow import route_after_selection

    result = route_after_selection(
        {
            "ocr_complete": False,
            "ocr_required": True,
            "detail_followup_required": {
                "url": "https://example.com/jobs/intermediary",
                "reason": "detail_content_incomplete",
            },
        }
    )

    assert result == "ocr"


def test_marker_ordinal_ignores_browser_chrome_and_uses_region():
    from agent.recipe.matcher import marker_ordinal

    markers = [
        {"id": 1, "bbox": [900, 140, 960, 170], "text": "상호작용 가능한 요소 (icon)"},
        {"id": 2, "bbox": [1200, 140, 1260, 170], "text": "상호작용 가능한 요소 (icon)"},
        {"id": 3, "bbox": [1500, 190, 1560, 250], "text": "상호작용 가능한 요소 (icon)"},
        {"id": 4, "bbox": [1600, 190, 1660, 250], "text": "상호작용 가능한 요소 (icon)"},
        {"id": 5, "bbox": [100, 500, 160, 560], "text": "상호작용 가능한 요소 (icon)"},
    ]

    assert marker_ordinal(markers[2], markers) == 0
    assert marker_ordinal(markers[3], markers) == 1


def test_record_ui_step_preserves_llm_selected_card_title():
    from agent.recipe.record import record_ui_step

    steps = []
    state = {
        "goal": "collect jobs",
        "current_url": "https://www.wanted.co.kr/search?query=iOS",
        "current_markers": [
            {"id": 1, "bbox": [10, 10, 120, 40], "text": "Reward 100"},
            {"id": 2, "bbox": [10, 50, 260, 85], "text": "Senior iOS Developer"},
            {"id": 3, "bbox": [10, 90, 220, 120], "text": "Example Company"},
        ],
    }

    record_ui_step(
        steps,
        state,
        "click_marker",
        {"marker_id": 1, "target_label": "Senior iOS Developer"},
        0,
    )

    assert steps[0]["target"]["text"] == "Reward 100"
    assert steps[0]["target"]["semantic_label"] == "Senior iOS Developer"


def test_feedback_episode_records_parameter_candidate_and_observation():
    from agent.recipe.feedback import record_action_episode

    episodes = []
    state = {
        "goal": "AI 엔지니어 채용공고 찾아줘",
        "current_url": "https://www.wanted.co.kr",
        "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "검색"}],
    }
    enriched = {
        "action": "type_in_marker",
        "status": "success",
        "result": "ok",
        "target": {"marker_id": 1, "text": "검색"},
    }

    record_action_episode(
        episodes,
        state,
        _action_request(content="검색어를 입력한다"),
        "type_in_marker",
        {
            "marker_id": 1,
            "text": "AI 엔지니어",
            "reason": "enter search keyword",
            "target_role": "search_input",
            "target_component": "site_search",
            "expected_after": "search keyword is entered",
        },
        enriched,
        {"url": "https://www.wanted.co.kr", "screenshot": "s.png", "marked_image": "m.png"},
        {"current_url": "https://www.wanted.co.kr", "current_url_stale": True, "screen_changed": True, "extracted_jd": {}, "is_finished": False},
        0,
    )

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["proposal"]["action"] == "type_in_marker"
    assert episode["proposal"]["llm_thought"] == "검색어를 입력한다"
    assert episode["proposal"]["expected_after"] == "search keyword is entered"
    assert episode["proposal"]["parameter_candidates"][0]["slot_candidate"] == "query"
    assert "state_key" not in episode["observation"]["before"]
    assert episode["observation"]["after"]["screen_changed"] is True
    assert episode["feedback"]["label"] == "partial"


def test_feedback_episode_does_not_infer_site_slot_from_open_url():
    from agent.recipe.feedback import record_action_episode

    episodes = []
    state = {
        "goal": "collect jobs",
        "current_url": "",
        "current_markers": [],
    }

    record_action_episode(
        episodes,
        state,
        _action_request(content="open the site home page"),
        "open_browser",
        {
            "url": "https://www.wanted.co.kr",
            "reason": "start from home page",
            "expected_after": "site home page is visible",
        },
        {"action": "open_browser", "status": "success", "result": {"opened": True}},
        {"state_key": "state-empty", "url": "", "screenshot": "s.png", "marked_image": "m.png"},
        {"current_url": "https://www.wanted.co.kr", "current_url_stale": True, "screen_changed": True, "extracted_jd": {}, "is_finished": False},
        0,
    )

    assert episodes[0]["proposal"]["parameter_candidates"] == []
    assert episodes[0]["proposal"]["expected_after"] == "site home page is visible"


def _sample_feedback_episode(seq=0):
    return {
        "seq": seq,
        "goal": "AI 엔지니어 채용공고 찾아줘",
        "site": "wanted.co.kr",
        "page_state_key": "state-home",
        "proposal": {
            "action": "type_in_marker",
            "args": {"marker_id": 1, "text": "AI 엔지니어"},
            "expected_after": "search results are visible",
            "parameter_candidates": [{"slot_candidate": "query", "value": "AI 엔지니어", "confidence": 0.45}],
        },
        "observation": {
            "before": {
                "state_key": "state-home",
                "capture_id": "worker-run:attempt:00:capture:0001",
                "url": "https://www.wanted.co.kr",
            },
            "after": {"screen_changed": True},
            "result": {"status": "success", "action": "type_in_marker"},
        },
        "feedback": {"label": "partial", "reason": "screen-changing action executed", "confidence": 0.45},
    }


def test_feedback_store_commits_and_reads_recent(tmp_path):
    from agent.recipe.feedback_store import FeedbackStore

    store = FeedbackStore(tmp_path / "feedback.db")
    saved = store.commit_episodes(
        [_sample_feedback_episode()],
        run_id="run-1",
        run_status="finished",
        source="test",
        review_attempt=0,
    )
    retry_saved = store.commit_episodes(
        [_sample_feedback_episode()],
        run_id="run-1",
        run_status="finished",
        source="test",
        review_attempt=1,
    )

    assert saved == 1
    assert retry_saved == 1
    rows = store.list_recent(limit=5)
    assert len(rows) == 2
    assert rows[0]["run_id"] == "run-1"
    assert {row["review_attempt"] for row in rows} == {0, 1}
    assert {
        row["episode_id"] for row in rows
    } == {
        "run-1:attempt:00:action:0000",
        "run-1:attempt:01:action:0000",
    }
    assert rows[0]["run_status"] == "finished"
    assert rows[0]["site"] == "wanted.co.kr"
    assert rows[0]["action"] == "type_in_marker"
    assert rows[0]["feedback_label"] == "partial"
    assert "page_state_key" not in rows[0]["payload"]
    assert "state_key" not in rows[0]["payload"]["observation"]["before"]
    assert (
        rows[0]["payload"]["observation"]["before"]["capture_id"]
        == "worker-run:attempt:00:capture:0001"
    )
    assert rows[0]["payload"]["proposal"]["parameter_candidates"][0]["slot_candidate"] == "query"


def test_database_initializes_feedback_episode_table(tmp_path):
    from shared.db.database import Database

    db_path = tmp_path / "jobs.db"
    Database(db_path)

    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    columns = [row[1] for row in conn.execute("PRAGMA table_info(feedback_episodes)").fetchall()]
    submission_columns = [row[1] for row in conn.execute("PRAGMA table_info(worker_submissions)").fetchall()]
    candidate_columns = [row[1] for row in conn.execute("PRAGMA table_info(recipe_candidates)").fetchall()]
    recipe_columns = [row[1] for row in conn.execute("PRAGMA table_info(recipes)").fetchall()]
    conn.close()

    assert "feedback_episodes" in tables
    assert "worker_submissions" in tables
    assert "recipe_candidates" in tables
    assert "episode_id" in columns
    assert "review_attempt" in columns
    assert "feedback_label" in columns
    assert "submission_id" in submission_columns
    assert "review_decision" in submission_columns
    assert "candidate_id" in candidate_columns
    assert "steps_json" in candidate_columns
    assert "validation_json" in candidate_columns
    assert "review_attempts" in candidate_columns
    assert "review_started_at" in candidate_columns
    assert "next_review_at" in candidate_columns
    assert "review_error" in candidate_columns
    assert "metadata_json" in recipe_columns


def test_database_migrates_feedback_episode_review_attempt(tmp_path):
    from shared.db.database import Database

    db_path = tmp_path / "legacy-feedback.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE feedback_episodes (
            episode_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            run_status TEXT,
            source TEXT,
            site TEXT,
            goal TEXT,
            action TEXT,
            feedback_label TEXT,
            feedback_reason TEXT,
            feedback_confidence REAL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    Database(db_path)

    conn = sqlite3.connect(db_path)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(feedback_episodes)")
    }
    indexes = {
        row[1] for row in conn.execute("PRAGMA index_list(feedback_episodes)")
    }
    conn.close()

    assert "review_attempt" in columns
    assert "idx_feedback_run_attempt" in indexes


def test_database_migrates_existing_recipe_candidate_review_queue(tmp_path):
    from shared.db.database import Database

    db_path = tmp_path / "legacy-candidates.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE recipe_candidates (
            candidate_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            submission_id TEXT NOT NULL,
            source TEXT,
            site TEXT,
            goal TEXT,
            keyword TEXT,
            status TEXT NOT NULL DEFAULT 'pending_replay',
            review_confidence REAL,
            steps_json TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            review_json TEXT,
            validation_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    Database(db_path)

    conn = sqlite3.connect(db_path)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(recipe_candidates)").fetchall()
    }
    indexes = {
        row[1] for row in conn.execute("PRAGMA index_list(recipe_candidates)").fetchall()
    }
    conn.close()
    assert {"review_attempts", "review_started_at", "next_review_at", "review_error"} <= columns
    assert "idx_recipe_candidates_review_queue" in indexes


def test_realtime_scraping_commits_feedback_episodes_with_run_status(monkeypatch):
    from agent.tools.realtime_scraping import _commit_feedback_episodes

    seen = {}

    class FakeStore:
        def commit_episodes(
            self,
            episodes,
            run_id=None,
            run_status="",
            source="",
            review_attempt=0,
        ):
            seen["episodes"] = episodes
            seen["run_id"] = run_id
            seen["run_status"] = run_status
            seen["source"] = source
            seen["review_attempt"] = review_attempt
            return len(episodes)

    monkeypatch.setattr("agent.recipe.feedback_store.FeedbackStore", lambda: FakeStore())

    saved = _commit_feedback_episodes(
        {"feedback_episodes": [_sample_feedback_episode()]},
        True,
        False,
        run_id="worker-run-1",
        review_attempt=2,
    )

    assert saved == 1
    assert seen["run_id"] == "worker-run-1"
    assert seen["run_status"] == "recursion_limit"
    assert seen["source"] == "realtime_scraping"
    assert seen["review_attempt"] == 2


def test_worker_submission_shape_review_requests_revision():
    from agent.recipe.reviewer import build_worker_submission, review_worker_submission

    submission = build_worker_submission(
        {
            "goal": "collect AI engineer jobs",
            "current_url": "https://www.wanted.co.kr/search?query=ai",
            "extracted_jd": {},
            "recorded_steps": [],
            "feedback_episodes": [],
        },
        site="wanted",
        keyword="ai engineer",
        run_status="stopped",
    )

    review = review_worker_submission(submission)

    assert review["decision"] == "revise"
    assert review["recipe_candidate"] is False
    assert "extracted_summary" in review["feedback_to_worker"]


def test_worker_submission_accepts_existing_database_job_evidence(monkeypatch):
    from agent.recipe.reviewer import build_worker_submission, review_worker_submission

    monkeypatch.setenv("VISION_WORKER_SUMMARY_MODE", "off")
    monkeypatch.setenv("VISION_WORKER_REVIEW_MODE", "shape")
    submission = build_worker_submission(
        {
            "goal": "collect two iOS jobs",
            "current_url": "https://www.wanted.co.kr/wd/2",
            "is_finished": True,
            "extracted_jd": {},
            "result_card_queue": [
                {"queue_id": "card-1", "status": "skipped", "job_id": 7},
                {"queue_id": "card-2", "status": "skipped", "job_id": 8},
            ],
            "recorded_steps": [],
            "feedback_episodes": [],
        },
        site="wanted",
        keyword="iOS 개발자",
        run_status="finished",
        target_count=2,
    )

    review = review_worker_submission(submission)

    assert submission["observed_job_ids"] == [7, 8]
    assert submission["collected_count"] == 0
    assert review["decision"] == "accept"
    assert review["recipe_candidate"] is False


def test_worker_submission_review_accepts_structured_data(monkeypatch):
    from agent.recipe.reviewer import build_worker_submission, review_worker_submission

    monkeypatch.setenv("VISION_WORKER_SUMMARY_MODE", "off")
    monkeypatch.setenv("VISION_WORKER_REVIEW_MODE", "shape")
    submission = build_worker_submission(
        {
            "goal": "collect AI engineer jobs",
            "current_url": "https://www.wanted.co.kr/wd/1",
            "extracted_jd": {
                "jobs": [
                    {
                        "company_name": "Acme",
                        "position": "AI Engineer",
                        "url": "https://www.wanted.co.kr/wd/1",
                    }
                ]
            },
            "recorded_steps": [
                {
                    "seq": 0,
                    "state_key": "state-a",
                    "action": "click_marker",
                    "target": {"text": "AI Engineer"},
                    "intent": "open the selected job card",
                    "target_role": "job_card_title",
                    "component": "search_result_card",
                    "expected_after": "job detail page is visible",
                }
            ],
            "feedback_episodes": [_sample_feedback_episode()],
            "transition_observations": [
                {"action_seq": 0, "status": "unknown", "marker_texts": ["AI Engineer", "주요업무"]}
            ],
        },
        site="wanted",
        keyword="ai engineer",
        run_status="finished",
    )

    review = review_worker_submission(submission)

    assert review["decision"] == "accept"
    assert review["accept_collected_data"] is True
    assert review["continue_collection"] is False
    assert review["recipe_candidate"] is True
    assert submission["task_category"] == ""
    assert submission["skill_metadata_evidence"]["site"] == "wanted"
    assert submission["skill_metadata_evidence"]["task_category"] == ""
    assert submission["skill_metadata_evidence"]["actions"] == ["click_marker"]
    assert submission["skill_metadata_evidence"]["step_intents"][0]["expected_after"] == "job detail page is visible"
    assert submission["transition_observations"][0]["action_seq"] == 0


def test_worker_review_payload_excludes_repeated_screen_trace():
    import json

    from agent.recipe.reviewer import build_worker_review_payload

    repeated_goal = "긴 작업 목표 " * 500
    submission = {
        "goal": repeated_goal,
        "site": "wanted",
        "keyword": "AI 에이전트",
        "target_count": 2,
        "run_status": "finished",
        "is_finished": True,
        "collected_count": 1,
        "collection_intent": {
            "original_query": "AI 에이전트 공고를 찾아줘",
            "search_keyword": "AI 에이전트",
        },
        "semantic_evidence": [
            {
                "company_name": "예시회사",
                "position": "AI Agent Engineer",
                "url": "https://example.com/jobs/1",
            }
        ],
        "extracted_summary": {
            "jobs": [{"company": "예시회사", "position": "AI Agent Engineer"}],
            "result_availability": {
                "available_result_count": 1,
                "count_evidence": "포지션 1",
                "count_confidence": 0.95,
            },
        },
        "recorded_steps": [{"seq": 1, "action": "click_marker", "roi_signature": "ROI_SECRET"}],
        "feedback_episodes": [
            {
                "seq": index,
                "goal": repeated_goal,
                "proposal": {"action": "click_marker"},
                "observation": {
                    "before": {"marker_texts": ["OCR_SECRET"] * 100},
                    "after": {},
                    "result": {},
                },
                "feedback": {"label": "success", "reason": "완료"},
            }
            for index in range(20)
        ],
        "transition_observations": [
            {"action_seq": 1, "status": "success", "marker_texts": ["OCR_SECRET"] * 100}
        ],
    }

    payload = build_worker_review_payload(submission, [])
    encoded = json.dumps(payload, ensure_ascii=False)

    assert "OCR_SECRET" not in encoded
    assert "ROI_SECRET" not in encoded
    assert repeated_goal not in encoded
    assert len(encoded) < 10000


def test_worker_submission_defaults_to_shape_review(monkeypatch):
    from agent.recipe import reviewer

    monkeypatch.setenv("VISION_WORKER_SUMMARY_MODE", "off")
    monkeypatch.delenv("VISION_WORKER_REVIEW_MODE", raising=False)
    submission = reviewer.build_worker_submission(
        {
            "goal": "collect recent AI engineer jobs",
            "current_url": "https://www.wanted.co.kr/wd/1",
            "recipe_params": {
                "collection_intent": {
                    "search_keyword": "AI Engineer",
                    "freshness_required": True,
                }
            },
            "extracted_jd": {
                "jobs": [
                    {
                        "company_name": "Acme",
                        "position": "AI Engineer",
                        "url": "https://www.wanted.co.kr/wd/1",
                        "posted_at": "2026-07-13",
                        "requirements": ["Python"],
                    }
                ]
            },
            "recorded_steps": [{"seq": 0, "action": "click_marker"}],
            "feedback_episodes": [_sample_feedback_episode()],
        },
        site="wanted",
        keyword="AI Engineer",
        run_status="finished",
    )
    def fail_llm_review(value, issues, fallback):
        raise AssertionError("기본 검토 경로에서 LLM을 호출하면 안 됩니다.")

    monkeypatch.setattr(reviewer, "_llm_review", fail_llm_review)

    review = reviewer.review_worker_submission(submission)

    assert review["decision"] == "accept"
    assert review["recipe_candidate"] is True
    assert submission["collection_intent"]["freshness_required"] is True
    assert submission["semantic_evidence"][0]["posted_at"] == "2026-07-13"


def test_worker_submission_report_summary_uses_llm(monkeypatch):
    import agent.recipe.reviewer as reviewer
    from agent.recipe.reviewer import build_worker_submission

    class FakeStructuredLLM:
        def invoke(self, messages):
            assert "raw_job" in messages[-1].content
            return reviewer.ReportJobSummary(
                jobs=[
                    reviewer.ReportJobSummaryItem(
                        company="비모소프트",
                        position="[인턴] iOS 개발자",
                        url="https://www.wanted.co.kr/wd/355442",
                        field_count=4,
                    )
                ]
            )

    def fake_structured_model(model, schema, **_kwargs):
        assert model == "fake-summary-model"
        assert schema is reviewer.ReportJobSummary
        return FakeStructuredLLM()

    monkeypatch.setenv("VISION_WORKER_SUMMARY_MODE", "llm")
    monkeypatch.setenv("VISION_WORKER_SUMMARY_MODEL", "fake-summary-model")
    monkeypatch.setattr(
        "agent.application.model_clients.get_structured_google_model",
        fake_structured_model,
    )

    submission = build_worker_submission(
        {
            "goal": "collect iOS jobs",
            "current_url": "https://www.wanted.co.kr/wd/355442",
            "extracted_jd": {
                "공고목록": [
                    {
                        "회사명": "비모소프트",
                        "직무명": "[인턴] iOS 개발자",
                        "URL": "https://www.wanted.co.kr/wd/355442",
                        "주요업무": "Swift 기반 iOS App 개발",
                    }
                ]
            },
        },
        site="wanted",
        keyword="iOS 개발자",
        run_status="finished",
    )

    summary = submission["extracted_summary"]
    assert summary["summary_source"] == "llm"
    assert summary["jobs"] == [
        {
            "company": "비모소프트",
            "position": "[인턴] iOS 개발자",
            "url": "https://www.wanted.co.kr/wd/355442",
            "field_count": 4,
        }
    ]


def test_submission_store_commits_and_reads_recent(tmp_path):
    from agent.recipe.submission_store import SubmissionStore

    submission = {
        "run_id": "worker-run-1",
        "goal": "collect jobs",
        "site": "wanted",
        "keyword": "ai engineer",
        "run_status": "finished",
        "review_attempt": 0,
        "collected_count": 1,
        "reflex_state_key": "state-a",
        "screen_signature": {"phash": "0" * 16},
        "recorded_steps": [
            {"seq": 0, "state_key": "state-a", "screen_signature": {"phash": "1" * 16}},
        ],
    }
    review = {"decision": "accept", "confidence": 0.7, "feedback_to_worker": ""}
    store = SubmissionStore(tmp_path / "submissions.db")

    submission_id = store.commit_submission(submission, review=review, source="test")
    rows = store.list_recent(limit=5)

    assert submission_id == "worker-run-1:0"
    assert len(rows) == 1
    assert rows[0]["review_decision"] == "accept"
    assert rows[0]["payload"]["keyword"] == "ai engineer"
    assert "reflex_state_key" not in rows[0]["payload"]
    assert "screen_signature" not in rows[0]["payload"]
    assert "state_key" not in rows[0]["payload"]["recorded_steps"][0]
    assert "screen_signature" not in rows[0]["payload"]["recorded_steps"][0]
    assert rows[0]["review"]["confidence"] == 0.7


def test_recipe_candidate_store_commits_reviewed_candidate(tmp_path):
    from agent.recipe.candidate_store import RecipeCandidateStore

    submission = {
        "run_id": "worker-run-1",
        "goal": "collect jobs",
        "site": "wanted",
        "keyword": "ai engineer",
        "review_attempt": 0,
        "reflex_state_key": "state-a",
        "screen_signature": {"phash": "0" * 16},
        "recorded_steps": [
            {
                "seq": 0,
                "decision_capture_id": "worker-run-1:attempt:00:capture:0001",
                "state_key": "state-a",
                "screen_signature": {"phash": "1" * 16},
                "action": "click_marker",
                "target": {"text": "AI Engineer"},
            }
        ],
        "transition_observations": [
            {
                "action_seq": 0,
                "from_capture_id": "worker-run-1:attempt:00:capture:0001",
                "to_capture_id": "worker-run-1:attempt:00:capture:0002",
                "status": "unknown",
                "marker_texts": ["AI Engineer", "주요업무"],
            }
        ],
    }
    review = {"decision": "accept", "recipe_candidate": True, "confidence": 0.7}
    store = RecipeCandidateStore(tmp_path / "candidates.db")

    candidate_id = store.commit_candidate(submission, review=review, source="test", submission_id="worker-run-1:0")
    rows = store.list_recent(limit=5)

    assert candidate_id == "worker-run-1:0"
    assert len(rows) == 1
    assert rows[0]["status"] == "pending_replay"
    assert rows[0]["site"] == "wanted"
    assert "state_key" not in rows[0]["steps"][0]
    assert "screen_signature" not in rows[0]["steps"][0]
    assert (
        rows[0]["steps"][0]["decision_capture_id"]
        == "worker-run-1:attempt:00:capture:0001"
    )
    assert "reflex_state_key" not in rows[0]["payload"]
    assert "screen_signature" not in rows[0]["payload"]
    assert "state_key" not in rows[0]["payload"]["recorded_steps"][0]
    assert "screen_signature" not in rows[0]["payload"]["recorded_steps"][0]
    assert (
        rows[0]["payload"]["transition_observations"][0]["from_capture_id"]
        == "worker-run-1:attempt:00:capture:0001"
    )
    assert rows[0]["payload"]["keyword"] == "ai engineer"
    assert rows[0]["review"]["recipe_candidate"] is True


def test_recipe_candidate_store_skips_non_candidates(tmp_path):
    from agent.recipe.candidate_store import RecipeCandidateStore

    store = RecipeCandidateStore(tmp_path / "candidates.db")
    candidate_id = store.commit_candidate(
        {"run_id": "worker-run-1", "review_attempt": 0, "recorded_steps": [{"state_key": "state-a"}]},
        review={"decision": "accept", "recipe_candidate": False},
        source="test",
    )

    assert candidate_id == ""
    assert store.list_recent(limit=5) == []

def _sample_recipe_candidate_submission():
    return {
        "run_id": "worker-run-critic",
        "goal": "collect jobs",
        "site": "wanted",
        "task_category": "검색",
        "keyword": "ai engineer",
        "review_attempt": 0,
        "skill_metadata_evidence": {"site": "wanted", "task_category": "검색"},
        "recorded_steps": [
            {
                "seq": 0,
                "state_key": "state-a",
                "page_role": "home",
                "action": "click_marker",
                "target": {"text": "AI Engineer"},
            }
        ],
        "transition_observations": [
            {
                "action_seq": 0,
                "status": "unknown",
                "marker_texts": ["AI Engineer", "주요업무"],
            }
        ],
        "feedback_episodes": [
            {
                "seq": 0,
                "proposal": {"args": {"page_role": "home"}},
                "observation": {
                    "before": {
                        "marker_texts": ["채용", "검색", "AI Engineer"],
                    }
                },
            }
        ],
    }


def test_recipe_candidate_status_update_records_llm_validation(tmp_path):
    from agent.recipe.candidate_store import RecipeCandidateStore

    store = RecipeCandidateStore(tmp_path / "candidates.db")
    candidate_id = store.commit_candidate(
        _sample_recipe_candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    assert store.update_status(
        candidate_id,
        "revise",
        validation={"review": {"decision": "revise", "reasons": ["needs clearer evidence"]}},
    ) is True
    row = store.get_candidate(candidate_id)

    assert row["status"] == "revise"
    assert row["validation"]["review"]["decision"] == "revise"


def test_candidate_reviewer_records_review_without_active_promotion(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        _sample_recipe_candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    seen = {}

    def critic(payload):
        seen["payload"] = payload
        return {
            "decision": "accept",
            "reasons": ["critic chose to promote"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "when_to_use": "Use on a job search result page.",
                "goal_pattern": "collect jobs",
                "site": "wanted",
                "inputs": [{"name": "query", "required": True, "observed_value": "AI"}],
                "step_intents": [
                    {
                        "seq": 0,
                        "action": "click_marker",
                        "intent": "Open the stable search control.",
                        "expected_after": "Search overlay is visible.",
                        "replay_mode": "fixed",
                    }
                ],
                "verification": {"success_signals": ["job data collected"]},
            },
            "transition_contracts": [
                {
                    "seq": 0,
                    "contract": {
                        "common_ready_cues": [{"kind": "text_any", "values": ["채용 상세"]}],
                        "outcomes": [{"name": "detail_opened", "cues": [{"kind": "text_any", "values": ["주요업무"]}]}],
                    },
                }
            ],
            "confidence": 0.82,
        }

    review = review_and_apply_candidate(candidate_id, db_path=tmp_path / "critic.db", critic=critic)
    candidate = store.get_candidate(candidate_id)
    recipes = RecipeStore(tmp_path / "critic.db").get_by_site("wanted")

    assert seen["payload"]["candidate_id"] == candidate_id
    assert seen["payload"]["task_category"] == "검색"
    assert "state_key" not in seen["payload"]["steps"][0]
    assert "worker_submission" not in seen["payload"]
    assert seen["payload"]["worker_execution"] == {"extracted_summary": {}}
    assert seen["payload"]["feedback_evidence"][0]["before_marker_texts"] == ["채용", "검색", "AI Engineer"]
    assert "overall successful run does not prove" in seen["payload"]["review_task"]
    assert seen["payload"]["transition_observations"][0]["action_seq"] == 0
    assert review["decision"] == "accept"
    assert candidate["status"] == "accepted"
    assert recipes == []


def test_candidate_reviewer_promote_requires_roi_signature(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _sample_recipe_candidate_submission()
    submission["recorded_steps"].append(
        {
            "seq": 1,
            "state_key": "state-results",
            "action": "click_marker",
            "target": {"text": "Specific Job Title"},
            "component": "job_card_title",
        }
    )
    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        mode="promote",
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["stable control only"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "step_intents": [
                    {"seq": 0, "action": "click_marker", "replay_mode": "fixed"},
                    {"seq": 1, "action": "click_marker", "replay_mode": "reasoning"},
                ],
            },
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(tmp_path / "critic.db").get_by_site("wanted")
    assert review["decision"] == "accept"
    assert review["promotion"]["promoted"] is False
    assert review["promotion"]["skipped_steps"][0]["reason"] == "roi_signature_missing"
    assert recipes == []


def test_candidate_reviewer_promotes_roi_fixed_steps_only(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _sample_recipe_candidate_submission()
    submission["recorded_steps"][0].update(
        {
            "roi_signature": {
                "algorithm": "roi-phash-dct64-v1",
                "phash": "0" * 16,
                "crop_rect_ratio": [0.7, 0.0, 0.9, 0.2],
            },
            "target": {
                "text": "검색",
                "bbox_ratio": [0.75, 0.1, 0.85, 0.2],
                "center_ratio": [0.8, 0.15],
            },
        }
    )
    submission["recorded_steps"].append(
        {
            "seq": 1,
            "state_key": "state-results",
            "action": "click_marker",
            "roi_signature": {
                "algorithm": "roi-phash-dct64-v1",
                "phash": "f" * 16,
                "crop_rect_ratio": [0.1, 0.2, 0.6, 0.4],
            },
            "target": {"text": "Specific Job Title"},
            "component": "job_card_title",
        }
    )
    submission["recorded_steps"].append(
        {
            "seq": 2,
            "state_key": "state-a",
            "action": "press_key",
            "param": {"key": "enter"},
        }
    )
    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        mode="promote",
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["stable control only"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "step_intents": [
                    {"seq": 0, "action": "click_marker", "replay_mode": "fixed"},
                    {"seq": 1, "action": "click_marker", "replay_mode": "reasoning"},
                    {"seq": 2, "action": "press_key", "replay_mode": "fixed"},
                ],
            },
            "transition_contracts": [
                {
                    "seq": 0,
                    "contract": {
                        "common_ready_cues": [{"kind": "text_any", "values": ["검색어"]}],
                    },
                }
            ],
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(tmp_path / "critic.db").get_by_site("wanted")

    assert review["decision"] == "accept"
    assert review["promotion"]["promoted"] is True
    assert review["promotion"]["promoted_step_count"] == 1
    assert len(recipes) == 1
    assert "state_key" not in recipes[0]["steps"][0]
    assert recipes[0]["steps"][0]["replay_mode"] == "fixed"
    assert len(recipes[0]["steps"]) == 1
    assert recipes[0]["skill_metadata"]["task_category"] == "검색"
    assert recipes[0]["steps"][0]["transition_contract"]["common_ready_cues"][0]["values"] == ["검색어"]
    assert {"seq": 2, "action": "press_key", "reason": "non_target_action"} in review["promotion"]["skipped_steps"]


def test_candidate_promotion_blocks_page_policy_managed_target(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _sample_recipe_candidate_submission()
    submission["recorded_steps"][0].update(
        {
            "component": "expand_detail_button",
            "roi_signature": {
                "algorithm": "roi-phash-dct64-v2",
                "phash": "0" * 16,
                "crop_rect_ratio": [0.2, 0.2, 0.5, 0.4],
            },
            "target": {"text": "정보더보기", "center_ratio": [0.35, 0.3]},
        }
    )
    submission["transition_observations"][0].update(
        {"source": "page_policy", "visual_change_ratio": 0.2}
    )
    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-policy:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        mode="promote",
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["critic incorrectly accepted the deterministic action"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "task_category": "검색",
                "step_intents": [{"seq": 0, "action": "click_marker", "replay_mode": "fixed"}],
            },
            "confidence": 0.9,
        },
    )

    assert review["promotion"]["promoted"] is False
    assert review["promotion"]["skipped_steps"][0]["reason"] == "managed_by_page_policy"
    assert RecipeStore(tmp_path / "critic.db").get_by_site("wanted") == []


def test_candidate_promotion_blocks_explicit_no_effect_target(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _sample_recipe_candidate_submission()
    submission["recorded_steps"][0].update(
        {
            "component": "tab_button",
            "roi_signature": {
                "algorithm": "roi-phash-dct64-v2",
                "phash": "1" * 16,
                "crop_rect_ratio": [0.1, 0.1, 0.4, 0.3],
            },
            "target": {"text": "포지션", "center_ratio": [0.25, 0.2]},
        }
    )
    submission["feedback_episodes"][0]["feedback"] = {
        "label": "no_effect",
        "reason": "screen_changed_during_reasoning",
    }
    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-no-effect:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        mode="promote",
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["critic incorrectly accepted the no-op"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "task_category": "검색",
                "step_intents": [{"seq": 0, "action": "click_marker", "replay_mode": "fixed"}],
            },
            "confidence": 0.9,
        },
    )

    assert review["promotion"]["promoted"] is False
    assert review["promotion"]["skipped_steps"][0]["reason"] == "feedback_no_effect"
    assert RecipeStore(tmp_path / "critic.db").get_by_site("wanted") == []


def test_candidate_reviewer_promotes_observed_page_role_over_llm_args(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _sample_recipe_candidate_submission()
    submission["recorded_steps"][0].update(
        {
            "page_role": "home",
            "action": "type_in_marker",
            "roi_signature": {
                "algorithm": "roi-phash-dct64-v1",
                "phash": "0" * 16,
                "crop_rect_ratio": [0.2, 0.1, 0.5, 0.2],
            },
            "target": {
                "text": "검색어를 입력해주세요",
                "bbox_ratio": [0.25, 0.12, 0.5, 0.18],
                "center_ratio": [0.375, 0.15],
            },
            "param": {"text": "ios 개발자", "slot_name": "query"},
            "slot_refs": ["query"],
        }
    )
    submission["feedback_episodes"][0] = {
        "seq": 0,
        "proposal": {"args": {"page_role": "home"}},
        "observation": {
            "before": {
                "url": "https://www.wanted.co.kr/",
                "marker_texts": ["검색어를 입력해주세요", "인기검색어"],
            }
        },
    }
    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        mode="promote",
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["search overlay input is reusable"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "inputs": [{"name": "query", "required": True}],
                "step_intents": [
                    {"seq": 0, "action": "type_in_marker", "replay_mode": "parameterized"}
                ],
            },
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(tmp_path / "critic.db").get_by_site("wanted")

    assert len(recipes) == 1
    assert recipes[0]["steps"][0]["page_role"] == "search_overlay"


def test_candidate_reviewer_revise_does_not_promote(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        _sample_recipe_candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        critic=lambda payload: {
            "decision": "revise",
            "reasons": ["critic requested another worker pass"],
            "feedback_to_worker": "collect clearer action rationale",
            "promote_to_active_recipe": False,
            "confidence": 0.6,
        },
    )
    candidate = store.get_candidate(candidate_id)

    assert review["decision"] == "revise"
    assert candidate["status"] == "revise"
    assert RecipeStore(tmp_path / "critic.db").get_by_site("wanted") == []


def test_candidate_reviewer_invalid_llm_shape_falls_back_to_revise(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore

    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        _sample_recipe_candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        critic=lambda payload: {"not_the_schema": True},
    )
    candidate = store.get_candidate(candidate_id)

    assert review["decision"] == "revise"
    assert review["promote_to_active_recipe"] is False
    assert candidate["status"] == "revise"
    assert "critic_review_failed" in candidate["validation"]["review"]["reasons"][0]


def test_recipe_promotion_request_only_enqueues_and_worker_reviews(tmp_path, monkeypatch):
    from agent.application import recipe_promotion_service as service
    from agent.application.recipe_promotion_worker import RecipePromotionWorker
    from agent.recipe import candidate_reviewer
    from agent.recipe.candidate_store import RecipeCandidateStore

    db_path = tmp_path / "promotion-worker.db"
    store = RecipeCandidateStore(db_path)
    seen = []
    candidate_id = store.commit_candidate(
        _sample_recipe_candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="candidate-background:0",
    )
    monkeypatch.setenv("VISION_RECIPE_AUTO_PROMOTE", "1")

    assert service.schedule_recipe_candidate_promotion(candidate_id, db_path=db_path) is True
    assert seen == []
    queued = service.get_recipe_candidate_promotion_status(candidate_id, db_path=db_path)
    assert queued["status"] == "pending_review"

    def review(value, db_path=None, mode="review", raise_on_critic_error=False):
        seen.append((value, mode, raise_on_critic_error))
        result = {
            "decision": "accept",
            "promotion": {"promoted": True, "saved_count": 1},
        }
        RecipeCandidateStore(db_path).update_status(
            value,
            "accepted",
            validation={"review": {"decision": "accept"}, "promotion": result["promotion"]},
        )
        return result

    monkeypatch.setattr(
        candidate_reviewer,
        "review_and_apply_candidate",
        review,
    )

    result = RecipePromotionWorker(db_path).process_one()

    assert seen == [(candidate_id, "promote", True)]
    assert result["decision"] == "accept"
    assert result["promotion"]["saved_count"] == 1
    completed = service.get_recipe_candidate_promotion_status(candidate_id, db_path=db_path)
    assert completed["status"] == "accepted"


def test_recipe_promotion_worker_retries_transport_failure(tmp_path, monkeypatch):
    from agent.application import recipe_promotion_service as service
    from agent.application.recipe_promotion_worker import RecipePromotionWorker
    from agent.recipe import candidate_reviewer
    from agent.recipe.candidate_store import RecipeCandidateStore

    db_path = tmp_path / "promotion-retry.db"
    store = RecipeCandidateStore(db_path)
    candidate_id = store.commit_candidate(
        _sample_recipe_candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="candidate-retry:0",
    )
    monkeypatch.setenv("VISION_RECIPE_AUTO_PROMOTE", "1")
    assert service.schedule_recipe_candidate_promotion(candidate_id, db_path=db_path) is True
    monkeypatch.setattr(
        candidate_reviewer,
        "review_and_apply_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("critic timeout")),
    )
    worker = RecipePromotionWorker(
        db_path,
        retry_delay_sec=0,
        max_attempts=2,
    )

    first = worker.process_one()
    second = worker.process_one()

    assert first["status"] == "pending_review"
    assert second["status"] == "review_failed"
    failed = store.get_candidate(candidate_id)
    assert failed["status"] == "review_failed"
    assert failed["review_attempts"] == 2
    assert "critic timeout" in failed["review_error"]

def _sample_worker_result_for_learning_mode():
    return {
        "submission": _sample_recipe_candidate_submission(),
        "extracted_jd": {
            "jobs": [
                {"company_name": "Acme", "position": "AI Engineer", "url": "https://example.com/jobs/1"}
            ]
        },
        "keyword": "ai engineer",
    }


def test_realtime_persists_review_accepted_partial_data_without_recipe(monkeypatch):
    from agent.tools import realtime_scraping as rs

    monkeypatch.setenv("VISION_RECIPE_LEARNING_MODE", "record")
    monkeypatch.setattr(
        rs,
        "_persist_collected_data_with_report",
        lambda extracted, keyword, collection_intent=None: {
            "submitted_count": 1,
            "persisted_count": 1,
            "created_count": 1,
            "updated_count": 0,
            "persisted_items": [{"job_id": 1, "url": "https://example.com/jobs/1"}],
            "rejected_count": 0,
            "rejected_items": [],
        },
    )
    candidates = []
    monkeypatch.setattr(
        rs,
        "_commit_recipe_candidate",
        lambda *args, **kwargs: candidates.append(args) or "candidate-1",
    )
    monkeypatch.setattr(
        "agent.recipe.submission_store.SubmissionStore.commit_submission",
        lambda self, submission, review=None, source="": "worker-partial:0",
    )

    persisted_count, _submission, review, _submission_id = rs.persist_accepted_worker_result(
        _sample_worker_result_for_learning_mode(),
        {
            "decision": "revise",
            "accept_collected_data": True,
            "continue_collection": True,
            "recipe_candidate": False,
        },
    )

    assert persisted_count == 1
    assert review["decision"] == "revise"
    assert candidates == []


def test_realtime_recipe_learning_mode_off_skips_candidate(monkeypatch):
    from agent.tools import realtime_scraping as rs

    monkeypatch.setenv("VISION_RECIPE_LEARNING_MODE", "off")
    monkeypatch.setattr(
        rs,
        "_persist_collected_data_with_report",
        lambda extracted, keyword, collection_intent=None: {
            "submitted_count": 1,
            "persisted_count": 1,
            "rejected_count": 0,
            "rejected_items": [],
        },
    )
    called = []
    monkeypatch.setattr(rs, "_commit_recipe_candidate", lambda *args, **kwargs: called.append(args) or "candidate-1")
    monkeypatch.setattr("agent.recipe.submission_store.SubmissionStore.commit_submission", lambda self, submission, review=None, source="": "worker-run-critic:0")

    persisted_count, submission, _review, _submission_id = rs.persist_accepted_worker_result(
        _sample_worker_result_for_learning_mode(),
        {"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
    )

    assert persisted_count == 1
    assert called == []
    assert "recipe_candidate_id" not in submission


def test_realtime_recipe_learning_mode_record_saves_candidate_without_critic(monkeypatch):
    from agent.tools import realtime_scraping as rs

    monkeypatch.setenv("VISION_RECIPE_LEARNING_MODE", "record")
    monkeypatch.setattr(
        rs,
        "_persist_collected_data_with_report",
        lambda extracted, keyword, collection_intent=None: {
            "submitted_count": 1,
            "persisted_count": 1,
            "rejected_count": 0,
            "rejected_items": [],
        },
    )
    seen = {}
    def fake_commit_recipe_candidate(submission, review, source, submission_id, mode):
        seen["mode"] = mode
        return "candidate-1"

    monkeypatch.setattr(rs, "_commit_recipe_candidate", fake_commit_recipe_candidate)
    scheduled = []
    monkeypatch.setattr(rs, "_schedule_recipe_candidate_promotion", lambda candidate_id: scheduled.append(candidate_id) or True)
    monkeypatch.setattr("agent.recipe.submission_store.SubmissionStore.commit_submission", lambda self, submission, review=None, source="": "worker-run-critic:0")

    _count, submission, _review, _submission_id = rs.persist_accepted_worker_result(
        _sample_worker_result_for_learning_mode(),
        {"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
    )

    assert seen["mode"] == "record"
    assert submission["recipe_candidate_id"] == "candidate-1"
    assert submission["recipe_learning_mode"] == "record"
    assert "recipe_candidate_review" not in submission
    assert scheduled == ["candidate-1"]


def test_realtime_recipe_learning_mode_promote_is_record_only(monkeypatch):
    from agent.tools import realtime_scraping as rs

    monkeypatch.setenv("VISION_RECIPE_LEARNING_MODE", "promote")
    monkeypatch.setattr(
        rs,
        "_persist_collected_data_with_report",
        lambda extracted, keyword, collection_intent=None: {
            "submitted_count": 1,
            "persisted_count": 1,
            "rejected_count": 0,
            "rejected_items": [],
        },
    )
    seen = {}
    def fake_commit_recipe_candidate(submission, review, source, submission_id, mode):
        seen["mode"] = mode
        return "candidate-1"

    monkeypatch.setattr(
        rs,
        "_commit_recipe_candidate",
        fake_commit_recipe_candidate,
    )
    scheduled = []
    monkeypatch.setattr(rs, "_schedule_recipe_candidate_promotion", lambda candidate_id: scheduled.append(candidate_id) or True)
    monkeypatch.setattr("agent.recipe.submission_store.SubmissionStore.commit_submission", lambda self, submission, review=None, source="": "worker-run-critic:0")

    _count, submission, _review, _submission_id = rs.persist_accepted_worker_result(
        _sample_worker_result_for_learning_mode(),
        {"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
    )

    assert seen["mode"] == "record"
    assert submission["recipe_learning_mode"] == "record"
    assert submission["recipe_candidate_id"] == "candidate-1"
    assert "recipe_candidate_review" not in submission
    assert scheduled == ["candidate-1"]


def test_process_recipe_candidates_review_mode_does_not_promote(tmp_path):
    from agent.recipe.candidate_reviewer import process_recipe_candidates
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    db_path = tmp_path / "batch.db"
    store = RecipeCandidateStore(db_path)
    candidate_ids = []
    for idx in range(2):
        submission = _sample_recipe_candidate_submission()
        submission["run_id"] = f"worker-run-batch-{idx}"
        submission["recorded_steps"][0]["state_key"] = f"state-{idx}"
        candidate_ids.append(
            store.commit_candidate(
                submission,
                review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
                source="test",
                submission_id=f"worker-run-batch-{idx}:0",
            )
        )

    seen = []

    def critic(payload):
        seen.append(payload["candidate_id"])
        return {
            "decision": "accept",
            "reasons": ["critic accepted replay evidence"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "confidence": 0.9,
        }

    result = process_recipe_candidates(limit=10, mode="review", db_path=db_path, critic=critic)

    assert result["mode"] == "review"
    assert result["processed_count"] == 2
    assert set(seen) == set(candidate_ids)
    assert RecipeStore(db_path).get_by_site("wanted") == []
    for candidate_id in candidate_ids:
        candidate = store.get_candidate(candidate_id)
        assert candidate["status"] == "accepted"


def test_process_recipe_candidates_promote_mode_writes_active_recipe(tmp_path):
    from agent.recipe.candidate_reviewer import process_recipe_candidates
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    db_path = tmp_path / "batch.db"
    store = RecipeCandidateStore(db_path)
    submission = _sample_recipe_candidate_submission()
    submission["recorded_steps"][0].update(
        {
            "roi_signature": {
                "algorithm": "roi-phash-dct64-v1",
                "phash": "0" * 16,
                "crop_rect_ratio": [0.7, 0.0, 0.9, 0.2],
            },
            "target": {
                "text": "검색",
                "bbox_ratio": [0.75, 0.1, 0.85, 0.2],
                "center_ratio": [0.8, 0.15],
            },
        }
    )
    candidate_id = store.commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-batch:0",
    )

    result = process_recipe_candidates(
        limit=5,
        mode="promote",
        db_path=db_path,
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["critic promoted replay evidence"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "step_intents": [
                    {"seq": 0, "action": "click_marker", "replay_mode": "fixed"}
                ]
            },
            "confidence": 0.88,
        },
    )

    candidate = store.get_candidate(candidate_id)
    recipes = RecipeStore(db_path).get_by_site("wanted")

    assert result["mode"] == "promote"
    assert result["processed_count"] == 1
    assert candidate["status"] == "accepted"
    assert len(recipes) == 1
    assert recipes[0]["steps"][0]["replay_mode"] == "fixed"
    assert result["results"][0]["promotion"]["promoted"] is True


def test_review_recipe_candidates_tool_returns_batch_json(monkeypatch):
    import json

    import agent.recipe.candidate_reviewer as reviewer
    from agent.tools.recipe_learning import review_recipe_candidates

    seen = {}

    def fake_process_recipe_candidates(limit=5, mode="review", status="pending_replay"):
        seen.update({"limit": limit, "mode": mode, "status": status})
        return {"mode": mode, "requested_limit": limit, "status": status, "processed_count": 0}

    monkeypatch.setattr(reviewer, "process_recipe_candidates", fake_process_recipe_candidates)

    payload = json.loads(
        review_recipe_candidates.invoke({"mode": "promote", "limit": 2, "status": "accepted"})
    )

    assert seen == {"limit": 2, "mode": "promote", "status": "accepted"}
    assert payload["mode"] == "promote"
    assert payload["requested_limit"] == 2
def test_parameterized_reflex_input_requires_runtime_slot_value():
    from agent.runtime.reflex_runtime import reflex_action_args

    step = {
        "action": "type_in_marker",
        "replay_mode": "parameterized",
        "param": {"slot_name": "result_filter_query", "text": "iOS"},
        "value": "iOS",
    }

    assert reflex_action_args(step, 12, params={"query": "데이터 엔지니어"}) is None
    assert reflex_action_args(
        step,
        12,
        params={"result_filter_query": "데이터 엔지니어"},
    )["text"] == "데이터 엔지니어"


def test_skill_metadata_marks_parameter_candidate_as_required():
    from agent.recipe.skill_metadata import build_skill_metadata_evidence

    evidence = build_skill_metadata_evidence(
        goal="필터로 검색",
        site="wanted",
        keyword="iOS",
        target_count=2,
        recorded_steps=[],
        feedback_episodes=[
            {
                "proposal": {
                    "parameter_candidates": [
                        {
                            "slot_candidate": "result_filter_query",
                            "value": "iOS",
                            "reason": "현재 요청의 필터 검색어",
                        }
                    ]
                }
            }
        ],
        extracted_summary={},
    )

    slot = next(item for item in evidence["inputs"] if item["name"] == "result_filter_query")
    assert slot["required"] is True


def test_card_queue_replay_accepts_close_current_tab_return():
    from agent.runtime.result_card_queue import queue_replay_after_return

    state = {
        "result_card_queue": [
            {
                "queue_id": "card-2",
                "status": "pending",
                "title": "두 번째 iOS 개발자",
                "bbox_ratio": [0.2, 0.3, 0.5, 0.4],
                "center_ratio": [0.35, 0.35],
            }
        ],
        "result_page_memory": {
            "screen_signature": {
                "phash": "0" * 16,
                "anchors": ["두 번째 iOS 개발자"],
                "size": [800, 600],
            }
        },
        "active_result_card": {},
    }

    message, markers, trace = queue_replay_after_return(
        state,
        {"action": "close_current_tab"},
        "https://www.jobkorea.co.kr/Search/?stext=iOS",
        [],
        {"phash": "0" * 16, "anchors": [], "size": [800, 600]},
        require_anchors=False,
    )

    assert message is not None
    assert trace["hit"] is True
    assert message.tool_calls[0].metadata["queue_id"] == "card-2"
    assert markers[0]["bbox"] == [160, 180, 400, 240]


def test_autonomous_tab_action_no_effect_is_detected_by_phash():
    from agent.runtime.transition_runtime import transition_no_effect_by_phash

    no_effect, distance = transition_no_effect_by_phash(
        {
            "action": "close_current_tab",
            "source": "autonomous",
            "before_url": "https://www.jobkorea.co.kr/Recruit/GI_Read/1",
            "before_phash": "0" * 16,
        },
        "https://www.jobkorea.co.kr/Recruit/GI_Read/1",
        {"phash": "0" * 16},
    )

    assert no_effect is True
    assert distance == 0


def test_autonomous_no_effect_routes_directly_to_reasoning():
    from agent.graph.workflow import route_after_selection

    route = route_after_selection(
        {
            "transition_status": "unknown",
            "transition_source": "autonomous",
            "transition_observations": [
                {
                    "action": "close_current_tab",
                    "source": "autonomous",
                    "reason": "no_screen_change",
                }
            ],
        }
    )

    assert route == "reasoning"

import time

from agent.graph import (
    worker_observation,
    worker_transition,
)
from agent.runtime.worker_contracts import action_event_transitions
from agent.tests.worker_test_support import (
    apply_update,
    node_runtime,
    worker_state,
)


def test_roi_record_and_replay_uses_target_crop(tmp_path):
    from PIL import Image, ImageDraw

    from agent.recipe.phash_replay import match_step_by_screen_signature
    from agent.recipe.record import record_ui_step

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    for path in (saved, current):
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        if path == current:
            draw.rectangle([0, 120, 200, 200], fill="black")
        image.save(path)

    steps: list[dict] = []
    record_ui_step(
        steps,
        {
            "goal": "검색",
            "current_url": "https://www.wanted.co.kr",
            "current_page_role": "home",
            "screen_signature": {"phash": "f" * 16, "size": [200, 200]},
            "current_screenshot": str(saved),
            "current_markers": [
                {"id": 1, "bbox": [150, 20, 170, 40], "text": "검색"},
            ],
        },
        "click_marker",
        {
            "marker_id": 1,
            "target_role": "button",
            "target_component": "search_button",
        },
        0,
    )

    marker_id, trace = match_step_by_screen_signature(
        steps[0],
        {"phash": "0" * 16, "size": [200, 200]},
        [{"id": 7, "bbox": [150, 20, 170, 40], "text": "검색"}],
        current_image_path=str(current),
    )

    assert steps[0]["page_role"] == "home"
    assert steps[0]["roi_signature"]["algorithm"] == "roi-phash-dct64-v2"
    assert marker_id == 7
    assert trace["matched"] is True
    assert trace["mode"] == "roi_phash"


def test_roi_replay_rejects_step_without_roi_signature():
    from agent.recipe.phash_replay import match_step_by_screen_signature

    marker_id, trace = match_step_by_screen_signature(
        {
            "screen_signature": {"phash": "0" * 16, "size": [1000, 1000]},
            "target": {
                "text": "검색",
                "bbox_ratio": [0.79, 0.08, 0.83, 0.12],
                "center_ratio": [0.81, 0.1],
            },
        },
        {"phash": "0" * 16, "size": [1000, 1000]},
        [{"id": 3, "bbox": [790, 80, 830, 120], "text": "검색"}],
    )

    assert marker_id is None
    assert trace["reason"] == "roi_signature_missing"


def test_contextual_step_records_and_matches_screen_context():
    from agent.recipe.phash_replay import screen_context_signature_match
    from agent.recipe.record import record_ui_step

    steps: list[dict] = []
    record_ui_step(
        steps,
        {
            "current_url": "https://www.wanted.co.kr",
            "current_page_role": "search_overlay",
            "screen_signature": {
                "algorithm": "phash-dct64-v1",
                "phash": "a" * 16,
                "size": [1921, 2088],
            },
            "current_markers": [],
        },
        "press_key",
        {"key": "enter", "page_role": "search_overlay"},
        2,
    )

    saved = steps[0]["screen_context_signature"]
    matched = screen_context_signature_match(
        saved,
        {"phash": "a" * 16, "size": [1921, 2088]},
    )
    rejected = screen_context_signature_match(
        saved,
        {"phash": "5" * 16, "size": [1921, 2088]},
    )

    assert saved["phash"] == "a" * 16
    assert matched["matched"] is True
    assert rejected["matched"] is False
    assert rejected["reason"] == "screen_context_phash_distance"


def test_replay_mode_requires_autonomous_declaration():
    from agent.recipe.record import record_ui_step

    state = {
        "current_url": "https://www.wanted.co.kr/search",
        "current_page_role": "search_overlay",
        "screen_signature": {
            "phash": "a" * 16,
            "size": [1920, 1080],
        },
        "current_markers": [],
    }
    steps: list[dict] = []

    record_ui_step(
        steps,
        state,
        "press_key",
        {"key": "enter"},
        1,
    )
    record_ui_step(
        steps,
        state,
        "press_key",
        {"key": "enter", "replay_mode": "fixed"},
        2,
    )

    assert steps[0]["replay_mode"] == "reasoning"
    assert steps[1]["replay_mode"] == "fixed"


def test_no_effect_reuses_ocr_only_for_matching_capture(monkeypatch, tmp_path):
    from PIL import Image

    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        last_capture_quality = {}

        def wait_for_transition_change(self, _reference_image_path):
            return False

        def capture_usable_screen(self):
            return screenshot

        def get_current_url(self):
            return "https://example.com/jobs"

        def analyze_ui(self, _path):
            raise AssertionError("같은 화면에서는 OCR을 다시 실행하면 안 됩니다.")

    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime

    raw_signature = worker_observation.compute_screen_phash_signature(screenshot)
    runtime = node_runtime(VisionWorkerRuntime(perception_factory=FakePerception))

    from agent.graph.worker_selection import selection_node

    working = worker_state(
        request={"worker_run_id": "worker-no-effect"},
        observation={
            "current_capture_id": "worker-no-effect:capture:0004",
            "capture_sequence": 4,
            "current_screenshot": str(screenshot),
            "current_url": "https://example.com/jobs",
            "current_url_stale": False,
            "current_markers": [
                {"id": 1, "bbox": [10, 20, 200, 60], "text": "검색"},
            ],
            "ui_context": "검색",
            "marked_image": str(screenshot),
            "screen_signature": raw_signature,
            "current_page_role": "search",
            "analysis_mode": "full",
            "ocr_complete": True,
        },
        replay={"reflex_blocked_recipe_keys": []},
        transition={
            "transition_request": {
                "action": "click_marker",
                "replay_mode": "reasoning",
                "action_seq": 3,
                "from_capture_id": "worker-no-effect:capture:0004",
                "source": "reflex",
                "recipe_key": "roi#search",
                "before_url": "https://example.com/jobs",
                "before_screenshot": str(screenshot),
                "started_at": time.time(),
                "contract": {},
            }
        },
    )
    result = {}
    for node in (
        worker_observation.capture_node,
        worker_transition.transition_node,
        selection_node,
    ):
        update = node(working, runtime)
        working = apply_update(working, update)
        result.update(update)

    assert (
        working["transition"]["transition_result"]["reason"]
        == "reflex_no_screen_change"
    )
    assert working["observation"]["ocr_complete"] is True
    assert working["observation"]["current_markers"][0]["id"] == 1
    assert (
        working["observation"]["previous_screen_observation"]["capture_id"]
        == "worker-no-effect:capture:0005"
    )

    monkeypatch.setattr(
        worker_transition,
        "transition_has_visual_change",
        lambda _pending, _path: (False, 0.0),
    )
    stale = worker_transition.transition_node(
        worker_state(
            observation={
                "current_capture_id": "worker-test:capture:0003",
                "current_screenshot": str(screenshot),
                "current_url": "https://example.com/jobs",
                "raw_screen_signature": {
                    "phash": "0" * 16,
                    "size": [800, 600],
                },
                "ocr_complete": False,
                "current_markers": [],
                "previous_screen_observation": {
                    "capture_id": "worker-test:capture:0001",
                    "screenshot": str(screenshot),
                    "markers": [{"id": 4, "bbox": [10, 20, 30, 40]}],
                },
            },
            transition={
                "transition_request": {
                    "action_seq": 0,
                    "action": "click_marker",
                    "from_capture_id": "worker-test:capture:0002",
                    "source": "autonomous",
                    "before_url": "https://example.com/jobs",
                    "before_screenshot": str(screenshot),
                    "started_at": time.time(),
                },
                "action_events": [
                    {
                        "seq": 0,
                        "result": {
                            "action": "click_marker",
                            "status": "success",
                        },
                    }
                ],
            },
        ),
        node_runtime(),
    )

    assert stale.get("observation", {}).get("ocr_complete") is None
    assert (
        action_event_transitions(stale["transition"]["action_events"])[0][
            "marker_count"
        ]
        == 0
    )

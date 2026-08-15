import time

from agent.graph import (
    worker_observation,
    worker_transition,
)
from agent.recipe.record import build_physical_action, build_screen_checkpoint
from agent.runtime.worker_contracts import action_event_transitions
from agent.runtime.transition_runtime import latest_no_effect_transition
from agent.tests.worker_test_support import (
    apply_update,
    node_runtime,
    worker_state,
)
from shared.schema.execution_record_schema import ActionTarget, ObservedAction


def _append_recorded_step(steps, state, action_name, args, seq):
    step = build_physical_action(state, action_name, args, seq)
    if step is not None:
        steps.append(step)


def test_roi_record_and_replay_uses_target_crop(tmp_path):
    from PIL import Image, ImageDraw

    from agent.runtime.target_matching import roi_signature_match

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    for path in (saved, current):
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        if path == current:
            draw.rectangle([0, 120, 200, 200], fill="black")
        image.save(path)

    steps: list[ObservedAction] = []
    _append_recorded_step(
        steps,
        worker_state(
            request={"goal": "검색"},
            observation={
                "current_url": "https://www.wanted.co.kr",
                "current_page_role": "home",
                "screen_signature": {"phash": "f" * 16, "size": [200, 200]},
                "current_screenshot": str(saved),
                "current_markers": [
                    {
                        "id": 1,
                        "bbox": [150, 20, 170, 40],
                        "text": "검색",
                        "type": "text",
                    },
                ],
            },
        ),
        "click_marker",
        {
            "marker_id": 1,
            "target_role": "button",
            "target_component": "search_button",
        },
        0,
    )
    _append_recorded_step(
        steps,
        worker_state(
            request={"goal": "검색"},
            observation={
                "current_url": "https://www.wanted.co.kr",
                "current_page_role": "home",
                "screen_signature": {"phash": "f" * 16, "size": [200, 200]},
                "current_screenshot": str(saved),
                "current_markers": [
                    {
                        "id": 1,
                        "bbox": [150, 20, 170, 40],
                        "text": "검색",
                        "type": "text",
                    },
                ],
            },
        ),
        "type_in_marker",
        {
            "marker_id": 1,
            "text": "AI 엔지니어",
            "slot_name": "search_keyword",
        },
        1,
    )

    trace = roi_signature_match(
        dict(steps[0].roi_signature),
        str(current),
        current_signature={"size": [200, 200]},
    )
    assert (
        build_screen_checkpoint(
            worker_state(
                observation={
                    "current_url": "https://www.wanted.co.kr",
                    "current_page_role": "home",
                }
            )
        ).page_role
        == "home"
    )
    assert steps[0].roi_signature["algorithm"] == "roi-phash-dct64-v2"
    assert "replay_mode" not in steps[0].model_dump()
    assert "replay_mode" not in steps[1].model_dump()
    assert steps[1].slot_refs == ["search_keyword"]
    assert trace["matched"] is True
    assert trace["mode"] == "roi_phash"


def test_roi_replay_rejects_step_without_roi_signature():
    from agent.runtime.target_matching import roi_signature_match

    action = ObservedAction(
        source_seq=0,
        action="click_marker",
        target=ActionTarget(
            text="검색",
            bbox_ratio=[0.79, 0.08, 0.83, 0.12],
            center_ratio=[0.81, 0.1],
        ),
    )
    trace = roi_signature_match(
        dict(action.roi_signature),
        "unused.png",
        current_signature={"size": [1000, 1000]},
    )

    assert trace["reason"] == "roi_signature_missing"


def test_trajectory_records_context_actions_without_promoting_them():
    from agent.runtime.worker_actions import is_supported_recipe_action_group

    steps: list[ObservedAction] = []
    state = worker_state(
        observation={
            "current_url": "https://www.wanted.co.kr",
            "current_page_role": "search_overlay",
            "screen_signature": {
                "algorithm": "phash-dct64-v1",
                "phash": "a" * 16,
                "size": [1921, 2088],
            },
            "current_markers": [],
        },
    )
    _append_recorded_step(
        steps,
        state,
        "press_key",
        {"key": "enter"},
        2,
    )
    _append_recorded_step(steps, state, "scroll", {"direction": "down"}, 3)
    _append_recorded_step(steps, state, "go_back", {}, 4)

    assert [step.action for step in steps] == ["press_key", "scroll", "go_back"]
    assert all("replay_mode" not in step.model_dump() for step in steps)
    assert build_screen_checkpoint(state).screen_context_signature["phash"] == "a" * 16
    assert is_supported_recipe_action_group(steps) is False
    assert (
        is_supported_recipe_action_group(
            [
                ObservedAction(source_seq=1, action="type_in_marker"),
                ObservedAction(
                    source_seq=2,
                    action="press_key",
                    param={"key": "enter"},
                ),
            ]
        )
        is True
    )


def test_recording_keeps_only_explicit_input_slot():

    state = worker_state(
        observation={
            "current_url": "https://www.wanted.co.kr/search",
            "current_page_role": "search_overlay",
            "screen_signature": {
                "phash": "a" * 16,
                "size": [1920, 1080],
            },
            "current_markers": [],
        },
    )
    steps: list[ObservedAction] = []

    _append_recorded_step(
        steps,
        state,
        "type_in_marker",
        {"text": "AI 엔지니어"},
        1,
    )
    _append_recorded_step(
        steps,
        state,
        "type_in_marker",
        {"text": "AI 엔지니어", "slot_name": "search_keyword"},
        2,
    )
    _append_recorded_step(
        steps,
        state,
        "press_key",
        {"key": "enter"},
        3,
    )

    assert steps[0].slot_refs == []
    assert steps[1].slot_refs == ["search_keyword"]
    assert all("replay_mode" not in step.model_dump() for step in steps)


def test_no_effect_reuses_ocr_only_for_matching_capture(monkeypatch, tmp_path):
    from PIL import Image

    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        last_capture_quality = {}

        def capture_usable_screen(self, *, reference_image_path=None):
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
            "observation_id": "worker-no-effect:observation:0004",
            "observation_sequence": 4,
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
            "ocr_complete": True,
        },
        replay={"reflex_blocked_recipe_keys": []},
        transition={
            "transition_request": {
                "action": "click_marker",
                "action_seq": 3,
                "before_observation_id": "worker-no-effect:observation:0004",
                "source": "reflex",
                "recipe_key": "experience-rule10#search",
                "before_url": "https://example.com/jobs",
                "before_screenshot": str(screenshot),
                "started_at": time.time(),
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
        working["observation"]["previous_observation"]["observation_id"]
        == "worker-no-effect:observation:0005"
    )

    monkeypatch.setattr(
        worker_transition,
        "transition_has_visual_change",
        lambda _pending, _path: (False, 0.0),
    )
    stale = worker_transition.transition_node(
        worker_state(
            observation={
                "observation_id": "worker-test:observation:0003",
                "current_screenshot": str(screenshot),
                "current_url": "https://example.com/jobs",
                "raw_screen_signature": {
                    "phash": "0" * 16,
                    "size": [800, 600],
                },
                "ocr_complete": False,
                "current_markers": [],
                "previous_observation": {
                    "observation_id": "worker-test:observation:0001",
                    "screenshot": str(screenshot),
                    "markers": [{"id": 4, "bbox": [10, 20, 30, 40]}],
                },
            },
            transition={
                "transition_request": {
                    "action_seq": 0,
                    "action": "click_marker",
                    "before_observation_id": "worker-test:observation:0002",
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
                        "candidate_action": {
                            "source_seq": 0,
                            "action": "click_marker",
                        },
                        "before_checkpoint": {
                            "observation_id": "worker-test:observation:0002",
                            "url_template": "example.com/jobs",
                        },
                    }
                ],
            },
        ),
        node_runtime(),
    )

    assert stale.get("observation", {}).get("ocr_complete") is None
    assert (
        action_event_transitions(stale["transition"]["action_events"])[
            0
        ].evidence.after_marker_texts
        == []
    )
    stale_state = apply_update(
        worker_state(
            observation={"current_screenshot": str(screenshot)},
            transition={"action_events": []},
        ),
        stale,
    )
    assert latest_no_effect_transition(stale_state)["action"] == "click_marker"

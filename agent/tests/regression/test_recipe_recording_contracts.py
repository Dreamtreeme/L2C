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
from shared.schema.recipe_schema import ActionTarget, PhysicalAction


def _append_recorded_step(steps, state, action_name, args, seq):
    step = build_physical_action(state, action_name, args, seq)
    if step is not None:
        steps.append(step)


def test_roi_record_and_replay_uses_target_crop(tmp_path):
    from PIL import Image, ImageDraw

    from agent.runtime.target_matching import match_local_target, roi_signature_match

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    for path in (saved, current):
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        if path == current:
            draw.rectangle([0, 120, 200, 200], fill="black")
        image.save(path)

    steps: list[PhysicalAction] = []
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
            "replay_mode": "fixed",
        },
        1,
    )

    trace = roi_signature_match(
        dict(steps[0].roi_signature),
        str(current),
        current_signature={"size": [200, 200]},
    )
    marker_id = match_local_target(
        steps[0].target.model_dump(mode="json") if steps[0].target else None,
        [
            {
                "id": 7,
                "bbox": [150, 20, 170, 40],
                "text": "검색",
                "type": "text",
            }
        ],
        [200, 200],
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
    assert steps[0].replay_mode == "fixed"
    assert steps[1].replay_mode == "parameterized"
    assert steps[1].slot_refs == ["search_keyword"]
    assert marker_id == 7
    assert trace["matched"] is True
    assert trace["mode"] == "roi_phash"


def test_roi_replay_rejects_step_without_roi_signature():
    from agent.runtime.target_matching import roi_signature_match

    action = PhysicalAction(
        source_seq=0,
        action="click_marker",
        replay_mode="fixed",
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


def test_local_target_match_rejects_merged_neighbor_text():
    from agent.runtime.target_matching import match_local_target

    target = {
        "text": "JOB검색",
        "marker_type": "text",
        "bbox_ratio": [0.4, 0.1, 0.6, 0.2],
        "center_ratio": [0.5, 0.15],
    }
    merged = {
        "id": 1,
        "bbox": [350, 100, 650, 200],
        "text": "지역 전체 JOB검색",
        "type": "text",
    }
    exact = {
        "id": 2,
        "bbox": [400, 100, 600, 200],
        "text": "JOB 검색",
        "type": "text",
    }

    assert match_local_target(target, [merged], [1000, 1000]) is None
    assert match_local_target(target, [merged, exact], [1000, 1000]) == 2
    assert (
        match_local_target(
            target,
            [
                {
                    "id": 3,
                    "bbox": [400, 100, 600, 200],
                    "text": "icon",
                    "type": "icon",
                }
            ],
            [1000, 1000],
        )
        == 3
    )


def test_trajectory_records_context_actions_without_promoting_them():
    from agent.runtime.worker_actions import is_supported_recipe_action_group

    steps: list[PhysicalAction] = []
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
        {"key": "enter", "replay_mode": "fixed"},
        2,
    )
    _append_recorded_step(steps, state, "scroll", {"direction": "down"}, 3)
    _append_recorded_step(steps, state, "go_back", {}, 4)

    assert [step.action for step in steps] == ["press_key", "scroll", "go_back"]
    assert [step.replay_mode for step in steps] == [
        "fixed",
        "reasoning",
        "reasoning",
    ]
    assert build_screen_checkpoint(state).screen_context_signature["phash"] == "a" * 16
    assert is_supported_recipe_action_group(steps) is False
    assert (
        is_supported_recipe_action_group(
            [
                PhysicalAction(source_seq=1, action="type_in_marker"),
                PhysicalAction(
                    source_seq=2,
                    action="press_key",
                    param={"key": "enter"},
                ),
            ]
        )
        is True
    )


def test_replay_mode_is_derived_from_executed_tool_contract():

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
    steps: list[PhysicalAction] = []

    _append_recorded_step(
        steps,
        state,
        "press_key",
        {"key": "enter"},
        1,
    )
    _append_recorded_step(
        steps,
        state,
        "press_key",
        {"key": "enter", "replay_mode": "fixed"},
        2,
    )
    _append_recorded_step(
        steps,
        state,
        "press_key",
        {"key": "escape", "replay_mode": "fixed"},
        3,
    )

    assert steps[0].replay_mode == "fixed"
    assert steps[1].replay_mode == "fixed"
    assert steps[2].replay_mode == "reasoning"


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
                "replay_mode": "reasoning",
                "action_seq": 3,
                "before_observation_id": "worker-no-effect:observation:0004",
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

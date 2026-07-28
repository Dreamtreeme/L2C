import time

from agent.graph import (
    worker_execution_dispatch,
    worker_observation,
    worker_transition,
)
from agent.graph.worker_reflex import reflex_node
from agent.runtime.job_card_queue import replay_job_card_after_return


def _candidate_submission() -> dict:
    return {
        "run_id": "worker-contract",
        "goal": "채용공고 수집",
        "site": "wanted",
        "task_category": "검색",
        "keyword": "AI 엔지니어",
        "review_attempt": 0,
        "skill_metadata_evidence": {
            "site": "wanted",
            "task_category": "검색",
        },
        "recorded_steps": [
            {
                "seq": 0,
                "page_role": "home",
                "action": "click_marker",
                "target": {
                    "text": "검색",
                    "bbox_ratio": [0.75, 0.1, 0.85, 0.2],
                    "center_ratio": [0.8, 0.15],
                },
                "roi_signature": {
                    "algorithm": "roi-phash-dct64-v2",
                    "phash": "0" * 16,
                    "crop_rect_ratio": [0.7, 0.0, 0.9, 0.3],
                    "target_center_ratio": [0.8, 0.15],
                },
            }
        ],
        "transition_records": [
            {
                "action_seq": 0,
                "status": "ready",
                "marker_texts": ["검색어"],
            }
        ],
        "feedback_episodes": [
            {
                "seq": 0,
                "proposal": {
                    "action": "click_marker",
                    "args": {"page_role": "home"},
                },
                "feedback": {"label": "success"},
                "observation": {
                    "before": {"marker_texts": ["채용", "검색"]},
                },
            }
        ],
    }


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
            "recent_images": [saved],
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


def test_no_effect_reuses_ocr_only_for_matching_capture(monkeypatch, tmp_path):
    from PIL import Image

    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        def capture_screen(self):
            return screenshot

        def analyze_ui(self, _path):
            raise AssertionError("같은 화면에서는 OCR을 다시 실행하면 안 됩니다.")

    monkeypatch.setattr(
        worker_observation,
        "_perception_engine",
        lambda: FakePerception(),
    )
    monkeypatch.setattr(
        worker_observation,
        "raw_screen_phash_signature",
        lambda _path: {"phash": "0" * 16, "size": [800, 600]},
    )

    from agent.graph.worker_selection import selection_node

    working = {
        "worker_run_id": "worker-no-effect",
        "worker_attempt_index": 0,
        "current_capture_id": "worker-no-effect:attempt:00:capture:0004",
        "capture_sequence": 4,
        "current_screenshot": str(screenshot),
        "current_url": "https://example.com/jobs",
        "current_url_stale": False,
        "current_markers": [
            {"id": 1, "bbox": [10, 20, 200, 60], "text": "검색"},
        ],
        "ui_context": "검색",
        "marked_image": str(screenshot),
        "screen_signature": {"phash": "0" * 16, "size": [800, 600]},
        "current_page_role": "search",
        "analysis_mode": "full",
        "ocr_complete": True,
        "reflex_blocked_recipe_keys": [],
        "transition_request": {
            "action": "click_marker",
            "action_seq": 3,
            "from_capture_id": "worker-no-effect:attempt:00:capture:0004",
            "source": "reflex",
            "recipe_key": "roi#search",
            "before_url": "https://example.com/jobs",
            "before_phash": "0" * 16,
            "before_screenshot": str(screenshot),
            "started_at": time.time(),
            "contract": {},
        },
    }
    result = {}
    for node in (
        worker_observation.capture_node,
        worker_transition.transition_node,
        selection_node,
    ):
        update = node(working)
        working.update(update)
        result.update(update)

    assert (
        result["transition_result"]["reason"]
        == "reflex_no_screen_change"
    )
    assert result["ocr_complete"] is True
    assert result["current_markers"][0]["id"] == 1
    assert (
        result["previous_screen_observation"]["capture_id"]
        == "worker-no-effect:attempt:00:capture:0005"
    )

    monkeypatch.setattr(
        worker_transition,
        "transition_has_visual_change",
        lambda _pending, _path: (False, 0.0),
    )
    monkeypatch.setattr(
        worker_transition,
        "transition_no_effect_by_phash",
        lambda _pending, _url, _signature: (True, 0),
    )
    stale = worker_transition.transition_node(
        {
            "current_capture_id": "worker-test:capture:0003",
            "current_screenshot": str(screenshot),
            "current_url": "https://example.com/jobs",
            "raw_screen_signature": {"phash": "0" * 16, "size": [800, 600]},
            "ocr_complete": False,
            "current_markers": [],
            "previous_screen_observation": {
                "capture_id": "worker-test:capture:0001",
                "screenshot": str(screenshot),
                "markers": [{"id": 4, "bbox": [10, 20, 30, 40]}],
            },
            "transition_request": {
                "action": "click_marker",
                "from_capture_id": "worker-test:capture:0002",
                "source": "autonomous",
                "before_url": "https://example.com/jobs",
                "before_phash": "0" * 16,
                "before_screenshot": str(screenshot),
                "started_at": time.time(),
            },
        }
    )

    assert stale.get("ocr_complete") is None
    assert stale["transition_records"][0]["marker_count"] == 0


def test_result_queue_replays_cached_card_after_return():
    state = {
        "job_card_queue": [
            {
                "queue_id": "card-2",
                "status": "pending",
                "title": "두 번째 iOS 개발자",
                "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                "center_ratio": [0.4, 0.425],
                "target": {
                    "text": "두 번째 iOS 개발자",
                    "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                    "center_ratio": [0.4, 0.425],
                },
            }
        ],
        "job_results_memory": {
            "screen_signature": {
                "phash": "0" * 16,
                "size": [1000, 1000],
                "anchors": ["두 번째 iOS 개발자"],
            },
        },
    }

    request, markers, trace = replay_job_card_after_return(
        state,
        {"action": "go_back"},
        "https://www.wanted.co.kr/search?query=ios",
        [],
        {
            "phash": "0" * 16,
            "size": [1000, 1000],
            "anchors": ["두 번째 iOS 개발자"],
        },
    )

    assert request is not None
    assert trace["hit"] is True
    assert request.tool_calls[0].name == "click_marker"
    assert markers[0]["bbox"] == [300, 400, 500, 450]


def test_recipe_store_scopes_by_site_and_task_category(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "recipes.db")
    store.commit_recipe(
        "wanted",
        "검색",
        [
            {
                "seq": 0,
                "page_role": "home",
                "action": "click_marker",
                "target": {"text": "검색", "center_ratio": [0.8, 0.1]},
                "roi_signature": {
                    "phash": "0" * 16,
                    "target_center_ratio": [0.8, 0.1],
                },
            }
        ],
        metadata={"task_category": "검색"},
    )

    assert len(store.get_site_recipes("wanted", task_category="검색")) == 1
    assert store.get_site_recipes("wanted", task_category="로그인") == []
    assert store.get_site_recipes("saramin", task_category="검색") == []


def test_recipe_store_saves_input_and_submit_as_one_path(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "recipe-path.db")
    saved = store.commit_recipe(
        "saramin",
        "검색",
        [
            {
                "seq": 1,
                "url_template": "saramin.co.kr/zf_user/",
                "page_role": "home",
                "action": "type_in_marker",
                "replay_mode": "parameterized",
                "slot_refs": ["query"],
                "param": {"slot_name": "query"},
                "target": {"text": "검색어"},
                "roi_signature": {"phash": "0" * 16},
            },
            {
                "seq": 2,
                "url_template": "saramin.co.kr/zf_user/",
                "page_role": "home",
                "action": "click_marker",
                "replay_mode": "fixed",
                "target": {"text": "검색"},
                "roi_signature": {"phash": "1" * 16},
            },
        ],
        metadata={"task_category": "검색"},
    )

    recipes = store.get_by_site("saramin")

    assert saved == 1
    assert len(recipes) == 1
    assert [
        step["action"]
        for step in recipes[0]["steps"]
    ] == ["type_in_marker", "click_marker"]


def test_recipe_store_keeps_cross_page_steps_in_one_path(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "separate-actions.db")
    saved = store.commit_recipe(
        "saramin",
        "검색",
        [
            {
                "seq": 1,
                "url_template": "saramin.co.kr/zf_user/",
                "page_role": "home",
                "action": "type_in_marker",
                "replay_mode": "parameterized",
                "slot_refs": ["query"],
                "param": {"slot_name": "query"},
                "target": {"text": "검색어"},
                "roi_signature": {"phash": "0" * 16},
            },
            {
                "seq": 2,
                "url_template": "saramin.co.kr/zf_user/search",
                "page_role": "search",
                "action": "click_marker",
                "replay_mode": "fixed",
                "target": {"text": "검색"},
                "roi_signature": {"phash": "1" * 16},
            },
        ],
        metadata={"task_category": "검색"},
    )

    recipes = store.get_by_site("saramin")

    assert saved == 1
    assert len(recipes) == 1
    assert len(recipes[0]["steps"]) == 2


def test_recipe_store_preserves_two_paths_with_overlapping_steps(tmp_path):
    from agent.recipe.store import RecipeStore

    def click_step(seq: int, label: str) -> dict:
        return {
            "seq": seq,
            "url_template": "example.com/search",
            "page_role": "search_results",
            "action": "click_marker",
            "replay_mode": "fixed",
            "component": f"control_{label.casefold()}",
            "target": {"text": label},
            "roi_signature": {"phash": str(seq) * 16},
        }

    store = RecipeStore(tmp_path / "branch-paths.db")
    first_path = [
        click_step(1, "A"),
        click_step(2, "B"),
        click_step(3, "C"),
    ]
    second_path = [
        click_step(1, "A"),
        click_step(2, "D"),
        click_step(3, "C"),
    ]

    assert store.replace_recipe_paths(
        "example",
        "탐색",
        [first_path],
        metadata={"task_category": "사이트 탐색"},
        candidate_id="candidate-abc",
    ) == 1
    assert store.replace_recipe_paths(
        "example",
        "탐색",
        [second_path],
        metadata={"task_category": "사이트 탐색"},
        candidate_id="candidate-adc",
    ) == 1

    recipes = store.get_by_site("example")
    assert len(recipes) == 2
    assert {
        tuple(step["target"]["text"] for step in recipe["steps"])
        for recipe in recipes
    } == {("A", "B", "C"), ("A", "D", "C")}


def test_replacing_one_candidate_keeps_shared_path_evidence(tmp_path):
    from agent.recipe.store import RecipeStore

    def path(middle: str) -> list[dict]:
        return [
            {
                "seq": seq,
                "page_role": "search_results",
                "action": "click_marker",
                "replay_mode": "fixed",
                "component": f"control_{label.casefold()}",
                "target": {"text": label},
                "roi_signature": {"phash": str(seq) * 16},
            }
            for seq, label in enumerate(["A", middle, "C"], start=1)
        ]

    store = RecipeStore(tmp_path / "shared-evidence.db")
    for candidate_id in ("candidate-one", "candidate-two"):
        assert store.replace_recipe_paths(
            "example",
            "탐색",
            [path("B")],
            metadata={"task_category": "사이트 탐색"},
            candidate_id=candidate_id,
        ) == 1

    shared = store.get_by_site("example")
    assert len(shared) == 1
    assert shared[0]["success_count"] == 2
    assert shared[0]["source_count"] == 2

    assert store.replace_recipe_paths(
        "example",
        "탐색",
        [path("D")],
        metadata={"task_category": "사이트 탐색"},
        candidate_id="candidate-two",
    ) == 1

    recipes = store.get_by_site("example")
    assert len(recipes) == 2
    assert {
        tuple(step["target"]["text"] for step in recipe["steps"]): (
            recipe["success_count"],
            recipe["source_count"],
        )
        for recipe in recipes
    } == {
        ("A", "B", "C"): (1, 1),
        ("A", "D", "C"): (1, 1),
    }


def test_stable_recipe_paths_split_at_unapproved_sequence_gap():
    from agent.recipe.replay_actions import split_stable_replay_paths

    steps = [
        {"seq": 1, "action": "click_marker"},
        {"seq": 2, "action": "press_key"},
        {"seq": 4, "action": "click_marker"},
        {"seq": 5, "action": "go_back"},
    ]

    paths = split_stable_replay_paths(steps)

    assert [
        [step["seq"] for step in path]
        for path in paths
    ] == [[1, 2], [4, 5]]


def test_reflex_replays_one_parameterized_roi_step(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw

    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    screenshot = tmp_path / "screen.png"
    image = Image.new("RGB", (200, 120), "white")
    ImageDraw.Draw(image).rectangle([10, 10, 70, 40], fill="black")
    image.save(screenshot)
    roi_signature = compute_target_roi_signature(
        screenshot,
        [10, 10, 70, 40],
        [200, 120],
    )

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            assert site == "wanted"
            return [
                (
                    "recipe-search",
                    SiteRecipe(
                        site="wanted",
                        goal="검색",
                        steps=[
                            RecipeStep(
                                seq=0,
                                action="type_in_marker",
                                page_role="home",
                                replay_mode="parameterized",
                                roi_signature=roi_signature,
                                target={
                                    "text": "검색",
                                    "bbox_ratio": [0.05, 0.0833, 0.35, 0.3333],
                                    "center_ratio": [0.2, 0.2083],
                                },
                                param={"slot_name": "query"},
                                slot_refs=["query"],
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())
    result = reflex_node(
        {
            "goal": "AI 엔지니어 공고",
            "current_url": "https://www.wanted.co.kr",
            "current_page_role": "home",
            "screen_signature": {"size": [200, 120]},
            "recent_images": [screenshot],
            "current_markers": [
                {"id": 7, "bbox": [10, 10, 70, 40], "text": "검색"},
            ],
            "recipe_params": {
                "site": "wanted",
                "task_category": "검색",
                "query": "AI 엔지니어",
            },
        }
    )

    call = result["pending_action"].tool_calls[0]
    assert result["reflex_trace"]["hit"] is True
    assert call.name == "type_in_marker"
    assert call.args["marker_id"] == 7
    assert call.args["text"] == "AI 엔지니어"
    assert len(result["pending_action"].tool_calls) == 1


def test_reflex_replays_selected_recipe_path_in_order(
    monkeypatch,
    tmp_path,
):
    from PIL import Image, ImageDraw

    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    input_screen = tmp_path / "recipe-path-input.png"
    input_image = Image.new("RGB", (240, 120), "white")
    input_draw = ImageDraw.Draw(input_image)
    input_draw.rectangle([10, 10, 130, 40], fill="black")
    input_image.save(input_screen)
    submit_screen = tmp_path / "recipe-path-submit.png"
    submit_image = Image.new("RGB", (240, 120), "white")
    submit_draw = ImageDraw.Draw(submit_image)
    submit_draw.rectangle([160, 10, 220, 40], fill="gray")
    submit_image.save(submit_screen)
    input_signature = compute_target_roi_signature(
        input_screen,
        [10, 10, 130, 40],
        [240, 120],
    )
    submit_signature = compute_target_roi_signature(
        submit_screen,
        [160, 10, 220, 40],
        [240, 120],
    )

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            return [
                (
                    "recipe-search-set",
                    SiteRecipe(
                        site=site,
                        goal="검색",
                        steps=[
                            RecipeStep(
                                seq=1,
                                action="type_in_marker",
                                page_role="home",
                                url_template="saramin.co.kr/zf_user/",
                                replay_mode="parameterized",
                                roi_signature=input_signature,
                                target={
                                    "text": "검색어",
                                    "bbox_ratio": [
                                        0.0417,
                                        0.0833,
                                        0.5417,
                                        0.3333,
                                    ],
                                    "center_ratio": [0.2917, 0.2083],
                                },
                                param={"slot_name": "query"},
                                slot_refs=["query"],
                            ),
                            RecipeStep(
                                seq=2,
                                action="click_marker",
                                page_role="home",
                                url_template="saramin.co.kr/zf_user/",
                                replay_mode="fixed",
                                roi_signature=submit_signature,
                                target={
                                    "text": "검색",
                                    "bbox_ratio": [
                                        0.6667,
                                        0.0833,
                                        0.9167,
                                        0.3333,
                                    ],
                                    "center_ratio": [0.7917, 0.2083],
                                },
                            ),
                            RecipeStep(
                                seq=3,
                                action="press_key",
                                page_role="home",
                                url_template="saramin.co.kr/zf_user/",
                                replay_mode="fixed",
                                param={"key": "enter"},
                                expected_after="검색 결과가 표시된다.",
                                transition_contract={
                                    "common_ready_cues": [
                                        {
                                            "kind": "text_any",
                                            "values": ["검색 결과"],
                                        }
                                    ]
                                },
                            ),
                        ],
                    ),
                )
            ]

    monkeypatch.setattr(
        "agent.recipe.store.RecipeStore",
        lambda: FakeStore(),
    )
    first = reflex_node(
        {
            "goal": "AI 엔지니어 공고",
            "current_url": "https://www.saramin.co.kr/zf_user/",
            "current_page_role": "home",
            "screen_signature": {"size": [240, 120]},
            "recent_images": [input_screen],
            "current_markers": [
                {
                    "id": 7,
                    "bbox": [10, 10, 130, 40],
                    "text": "검색어",
                },
            ],
            "recipe_params": {
                "site": "saramin",
                "task_category": "검색",
                "query": "AI 엔지니어",
            },
        }
    )

    assert first["reflex_trace"]["hit"] is True
    assert first["pending_action"].summary == "cached recipe path step"
    assert len(first["pending_action"].tool_calls) == 1
    assert first["pending_action"].tool_calls[0].name == "type_in_marker"
    assert (
        first["pending_action"].tool_calls[0].args["text"]
        == "AI 엔지니어"
    )
    assert first["active_reflex_recipe"]["next_step_index"] == 1

    second = reflex_node(
        {
            "goal": "AI 엔지니어 공고",
            "current_url": "https://www.saramin.co.kr/zf_user/",
            "current_page_role": "home",
            "screen_signature": {"size": [240, 120]},
            "recent_images": [submit_screen],
            "current_markers": [
                {
                    "id": 8,
                    "bbox": [160, 10, 220, 40],
                    "text": "검색",
                },
            ],
            "recipe_params": {
                "site": "saramin",
                "task_category": "검색",
                "query": "AI 엔지니어",
            },
            "active_reflex_recipe": first["active_reflex_recipe"],
        }
    )

    assert second["reflex_trace"]["hit"] is True
    assert len(second["pending_action"].tool_calls) == 1
    assert second["pending_action"].tool_calls[0].name == "click_marker"
    assert second["pending_action"].tool_calls[0].args["marker_id"] == 8
    assert second["active_reflex_recipe"]["next_step_index"] == 2

    third = reflex_node(
        {
            "goal": "AI 엔지니어 공고",
            "current_url": "https://www.saramin.co.kr/zf_user/",
            "current_page_role": "home",
            "screen_signature": {"size": [240, 120]},
            "recent_images": [submit_screen],
            "current_markers": [],
            "recipe_params": {
                "site": "saramin",
                "task_category": "검색",
                "query": "AI 엔지니어",
            },
            "active_reflex_recipe": second["active_reflex_recipe"],
        }
    )

    assert third["reflex_trace"]["hit"] is True
    assert third["pending_action"].tool_calls[0].name == "press_key"
    assert third["pending_action"].tool_calls[0].args["key"] == "enter"
    assert third["active_reflex_recipe"]["next_step_index"] == 3


def test_active_reflex_recipe_is_cleared_only_after_final_transition():
    base_state = {
        "ocr_complete": True,
        "current_url": "https://www.saramin.co.kr/zf_user/search",
        "current_screenshot": "",
        "current_capture_id": "capture:0002",
        "current_markers": [
            {"id": 1, "bbox": [0, 0, 20, 20], "text": "검색 결과"},
        ],
        "screen_signature": {},
        "transition_request": {
            "action": "click_marker",
            "source": "reflex",
            "recipe_key": "recipe-search-set",
            "before_url": "https://www.saramin.co.kr/zf_user/",
            "started_at": time.time(),
            "contract": {
                "common_ready_cues": [
                    {
                        "kind": "text_any",
                        "values": ["검색 결과"],
                    }
                ],
                "timeout_sec": 8.0,
            },
        },
    }

    intermediate = worker_transition.transition_node(
        {
            **base_state,
            "active_reflex_recipe": {
                "recipe_key": "recipe-search-set",
                "next_step_index": 2,
                "step_count": 3,
            },
        }
    )
    completed = worker_transition.transition_node(
        {
            **base_state,
            "active_reflex_recipe": {
                "recipe_key": "recipe-search-set",
                "next_step_index": 3,
                "step_count": 3,
            },
        }
    )

    assert intermediate["transition_result"]["status"] == "ready"
    assert intermediate["active_reflex_recipe"]["next_step_index"] == 2
    assert completed["transition_result"]["status"] == "ready"
    assert completed["active_reflex_recipe"] == {}


def test_detail_finish_extracts_once_and_clears_buffer(monkeypatch):
    monkeypatch.setattr(
        worker_execution_dispatch,
        "extract_job_from_job_detail_buffer",
        lambda _state, current_url: {
            "company_name": "보이저엑스",
            "position": "iOS 개발자",
            "url": current_url,
            "requirements": ["Swift"],
        },
    )

    result, extracted = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {
            "page_role": "job_detail",
            "observed_fields": {
                "company_name": "보이저엑스",
                "position": "iOS 개발자",
                "requirements": "자격요건 Swift",
            },
        },
        {},
        current_url="https://www.wanted.co.kr/wd/1",
        state={
            "job_collection_contract": {
                "required_fields": [
                    "company_name",
                    "position",
                    "url",
                    "requirements",
                ]
            },
            "job_detail_buffer": {
                "url": "https://www.wanted.co.kr/wd/1",
                "lines": [{"text": "자격요건 Swift"}],
            }
        },
    )

    assert result["status"] == "success"
    assert result["_job_detail_buffer"] == {}
    assert extracted["공고목록"][0]["position"] == "iOS 개발자"


def test_detail_action_args_compaction_is_idempotent():
    from agent.graph.worker_execution_policy import compact_action_args

    args = {
        "page_role": "job_detail",
        "observed_fields": {
            "requirements": "Python",
            "main_tasks": "API 개발",
        },
        "page_exhausted": True,
    }

    compacted = compact_action_args("finish_detail_reading", args)

    assert compact_action_args(
        "finish_detail_reading",
        compacted,
    ) == compacted


def test_detail_observation_accepts_multiple_evidence_lines():
    from agent.graph.action_request import build_action_request

    request = build_action_request(
        "llm",
        "",
        [
            {
                "id": "finish",
                "name": "finish_detail_reading",
                "args": {
                    "observed_fields": {
                        "main_tasks": [
                            "API 개발",
                            "성능 최적화",
                        ]
                    }
                },
            }
        ],
    )

    assert request.tool_calls[0].args["observed_fields"] == {
        "main_tasks": "API 개발; 성능 최적화"
    }


def test_detail_finish_skips_extraction_until_required_evidence_is_complete(
    monkeypatch,
):
    calls = []

    def fail_if_called(_state, _current_url):
        calls.append(True)
        return {}

    monkeypatch.setattr(
        worker_execution_dispatch,
        "extract_job_from_job_detail_buffer",
        fail_if_called,
    )

    result, _extracted = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {
            "page_role": "job_detail",
            "observed_fields": {
                "company_name": "예시회사",
                "position": "백엔드 개발자",
            },
        },
        {},
        current_url="https://www.wanted.co.kr/wd/2",
        state={
            "job_collection_contract": {
                "required_fields": [
                    "company_name",
                    "position",
                    "url",
                    "main_tasks",
                    "requirements",
                ]
            },
            "job_detail_buffer": {
                "url": "https://www.wanted.co.kr/wd/2",
                "lines": [{"text": "백엔드 개발자"}],
            },
        },
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "required_field_evidence_incomplete"
    assert result["_job_detail_followup"]["missing_fields"] == [
        "main_tasks",
        "requirements",
    ]
    assert calls == []


def test_detail_finish_allows_explicit_unavailable_field_at_page_end(
    monkeypatch,
):
    calls = []

    def extract_once(_state, current_url):
        calls.append(True)
        return {
            "company_name": "예시회사",
            "position": "백엔드 개발자",
            "url": current_url,
            "main_tasks": ["API 개발"],
            "requirements": ["Python"],
        }

    monkeypatch.setattr(
        worker_execution_dispatch,
        "extract_job_from_job_detail_buffer",
        extract_once,
    )
    required_fields = [
        "company_name",
        "position",
        "url",
        "main_tasks",
        "requirements",
        "benefits",
    ]

    result, extracted = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {
            "page_role": "job_detail",
            "observed_fields": {
                "company_name": "예시회사",
                "position": "백엔드 개발자",
                "main_tasks": "API 개발",
                "requirements": "Python",
            },
            "unavailable_fields": ["benefits"],
            "page_exhausted": True,
        },
        {},
        current_url="https://www.wanted.co.kr/wd/3",
        state={
            "job_collection_contract": {
                "required_fields": required_fields,
            },
            "job_detail_buffer": {
                "url": "https://www.wanted.co.kr/wd/3",
                "lines": [{"text": "API 개발"}, {"text": "Python"}],
            },
        },
    )

    assert result["status"] == "success"
    assert calls == [True]
    job = extracted["공고목록"][0]
    assert job["_collection_required_fields"] == required_fields
    assert job["_collection_unavailable_fields"] == ["benefits"]


def test_candidate_promotion_keeps_only_safe_roi_target(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _candidate_submission()
    submission["recorded_steps"].append(
        {
            "seq": 1,
            "page_role": "search",
            "action": "click_marker",
            "component": "job_card_title",
            "target": {"text": "실행마다 달라지는 공고"},
            "roi_signature": {
                "phash": "f" * 16,
                "crop_rect_ratio": [0.1, 0.2, 0.6, 0.4],
            },
        }
    )
    submission["recorded_steps"].append(
        {
            "seq": 2,
            "action": "press_key",
            "param": {"key": "enter"},
        }
    )
    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        submission,
        review={
            "decision": "accept",
            "recipe_candidate": True,
            "confidence": 0.8,
        },
        source="test",
        submission_id="worker-contract:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        mode="promote",
        critic=lambda _payload: {
            "decision": "accept",
            "reasons": ["검색 버튼만 재사용 가능"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "task_category": "검색",
                "step_intents": [
                    {"seq": 0, "action": "click_marker", "replay_mode": "fixed"},
                    {"seq": 1, "action": "click_marker", "replay_mode": "reasoning"},
                    {"seq": 2, "action": "press_key", "replay_mode": "fixed"},
                ],
            },
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(tmp_path / "critic.db").get_by_site("wanted")
    assert review["promotion"]["promoted"] is True
    assert review["promotion"]["promoted_step_count"] == 1
    assert len(recipes) == 1
    assert recipes[0]["steps"][0]["action"] == "click_marker"


def test_candidate_promotion_saves_consecutive_steps_as_one_path(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _candidate_submission()
    submission["recorded_steps"].append(
        {
            "seq": 1,
            "page_role": "search_results",
            "action": "click_marker",
            "component": "filter_button",
            "target": {"text": "직무 필터"},
            "roi_signature": {
                "phash": "1" * 16,
                "crop_rect_ratio": [0.1, 0.1, 0.4, 0.3],
            },
        }
    )
    submission["transition_records"].append(
        {
            "action_seq": 1,
            "status": "ready",
            "marker_texts": ["직무 선택"],
        }
    )
    submission["feedback_episodes"].append(
        {
            "seq": 1,
            "proposal": {
                "action": "click_marker",
                "args": {"page_role": "search_results"},
                "component_candidate": "filter_button",
            },
            "feedback": {"label": "success"},
            "observation": {
                "before": {"marker_texts": ["직무 필터"]},
            },
        }
    )
    db_path = tmp_path / "path-promotion.db"
    candidate_id = RecipeCandidateStore(db_path).commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True},
        source="test",
        submission_id="worker-path:0",
    )

    result = review_and_apply_candidate(
        candidate_id,
        db_path=db_path,
        mode="promote",
        critic=lambda _payload: {
            "decision": "accept",
            "reasons": ["두 단계가 모두 검증됨"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "task_category": "검색",
                "step_intents": [
                    {
                        "seq": 0,
                        "action": "click_marker",
                        "replay_mode": "fixed",
                    },
                    {
                        "seq": 1,
                        "action": "click_marker",
                        "replay_mode": "fixed",
                    },
                ],
            },
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(db_path).get_by_site("wanted")
    assert result["promotion"]["promoted_path_count"] == 1
    assert len(recipes) == 1
    assert [step["seq"] for step in recipes[0]["steps"]] == [0, 1]


def test_contextual_followup_is_promoted_and_selected(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore
    from agent.runtime.followup_runtime import (
        select_followup_after_transition,
    )

    submission = _candidate_submission()
    submission["recorded_steps"].extend(
        [
            {
                "seq": 1,
                "url_template": "wanted.co.kr/search",
                "page_role": "search_overlay",
                "declared_page_role": "search_overlay",
                "action": "type_in_marker",
                "component": "search_input",
                "target": {"text": "검색어"},
                "roi_signature": {
                    "phash": "1" * 16,
                    "crop_rect_ratio": [0.1, 0.1, 0.7, 0.2],
                },
            },
            {
                "seq": 2,
                "url_template": "wanted.co.kr/search",
                "page_role": "search_overlay",
                "declared_page_role": "search_overlay",
                "action": "press_key",
                "param": {"key": "enter"},
                "expected_after": "검색 결과가 표시된다.",
            },
        ]
    )
    submission["feedback_episodes"].extend(
        [
            {
                "seq": 1,
                "proposal": {
                    "action": "type_in_marker",
                    "args": {
                        "page_role": "search_overlay",
                        "target_component": "search_input",
                    },
                    "component_candidate": "search_input",
                },
                "feedback": {"label": "partial"},
                "observation": {
                    "before": {
                        "url": "https://www.wanted.co.kr/search",
                        "marker_texts": ["검색어"],
                    },
                    "result": {"status": "success"},
                },
            },
            {
                "seq": 2,
                "proposal": {
                    "action": "press_key",
                    "args": {
                        "key": "enter",
                        "page_role": "search_overlay",
                    },
                },
                "feedback": {"label": "partial"},
                "observation": {
                    "before": {
                        "url": "https://www.wanted.co.kr/search",
                        "marker_texts": ["AI 엔지니어"],
                    },
                    "result": {"status": "success"},
                },
            },
        ]
    )
    submission["transition_records"].extend(
        [
            {
                "action_seq": 1,
                "action": "type_in_marker",
                "status": "unknown",
                "reason": "transition_contract_missing",
                "marker_count": 3,
                "visual_change_ratio": 0.08,
                "marker_texts": ["AI 엔지니어"],
            },
            {
                "action_seq": 2,
                "action": "press_key",
                "status": "unknown",
                "reason": "transition_contract_missing",
                "marker_count": 5,
                "visual_change_ratio": 0.12,
                "marker_texts": ["검색 결과", "AI 엔지니어"],
            },
        ]
    )
    db_path = tmp_path / "followup.db"
    store = RecipeCandidateStore(db_path)
    candidate_id = store.commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True},
        source="test",
        submission_id="worker-followup:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=db_path,
        mode="promote",
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["검색 입력 직후 Enter 전환이 검증됨"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "task_category": "검색",
                "step_intents": [
                    {
                        "seq": item["seq"],
                        "action": item["action"],
                        "replay_mode": (
                            "fixed"
                            if item["seq"] in {0, 2}
                            else "reasoning"
                        ),
                    }
                    for item in payload["required_step_intents"]
                ],
            },
            "transition_contracts": [
                {
                    "seq": 2,
                    "contract": {
                        "common_ready_cues": [
                            {
                                "kind": "text_any",
                                "values": ["검색 결과"],
                            }
                        ],
                        "timeout_sec": 8.0,
                    },
                }
            ],
            "confidence": 0.9,
        },
    )

    active = RecipeStore(db_path).active_counts("wanted")
    assert review["promotion"]["promoted_followup_count"] == 1
    assert active["followup_strategies"] == 1

    request, trace = select_followup_after_transition(
        {
            "current_url": "https://www.wanted.co.kr/search",
            "current_page_role": "home",
            "recipe_params": {"task_category": "검색"},
        },
        {
            "status": "ready",
            "action": "type_in_marker",
            "step": {
                "component": "search_input",
                "page_role": "search_overlay",
            },
        },
        db_path=db_path,
    )

    assert trace["hit"] is True
    assert request is not None
    assert request.source == "followup_strategy"
    assert request.tool_calls[0].name == "press_key"
    assert request.tool_calls[0].args["key"] == "enter"


def test_queue_phash_match_records_return_transition(monkeypatch):
    from agent.graph import worker_selection

    monkeypatch.setattr(
        worker_selection,
        "select_followup_after_transition",
        lambda *_args, **_kwargs: (
            None,
            {"hit": False, "reason": "test"},
        ),
    )
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

    monkeypatch.setattr(
        worker_selection,
        "select_followup_after_transition",
        lambda *_args, **_kwargs: (
            None,
            {"hit": False, "reason": "test"},
        ),
    )
    transition_request = {
        "status": "needs_ocr",
        "action": "go_back",
        "action_seq": 9,
        "source": "followup_strategy",
        "started_at": time.time(),
        "contract": {"timeout_sec": 8.0},
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


def test_detail_completion_promotes_go_back_followup(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.runtime.followup_runtime import select_followup_action

    submission = _candidate_submission()
    submission["recorded_steps"].append(
        {
            "seq": 2,
            "url_template": "wanted.co.kr/wd/{id}",
            "page_role": "job_detail",
            "declared_page_role": "job_detail",
            "action": "go_back",
            "param": {},
            "expected_after": "검색 결과 목록이 표시된다.",
        }
    )
    submission["feedback_episodes"].extend(
        [
            {
                "seq": 1,
                "proposal": {
                    "action": "finish_detail_reading",
                    "args": {
                        "page_role": "job_detail",
                    },
                },
                "feedback": {"label": "success"},
                "observation": {
                    "before": {
                        "url": "https://www.wanted.co.kr/wd/123",
                        "marker_texts": ["주요업무", "자격요건"],
                    },
                    "result": {"status": "success"},
                },
            },
            {
                "seq": 2,
                "proposal": {
                    "action": "go_back",
                    "args": {"page_role": "job_detail"},
                },
                "feedback": {"label": "partial"},
                "observation": {
                    "before": {
                        "url": "https://www.wanted.co.kr/wd/123",
                        "marker_texts": ["주요업무", "자격요건"],
                    },
                    "result": {"status": "success"},
                },
            },
        ]
    )
    submission["transition_records"].append(
        {
            "action_seq": 2,
            "action": "go_back",
            "source": "autonomous",
            "status": "ready",
            "reason": "queue_return_phash_match",
            "marker_count": 2,
            "marker_texts": ["검색 결과", "iOS 개발자"],
            "ocr_skipped": True,
        }
    )
    db_path = tmp_path / "go-back.db"
    candidate_id = RecipeCandidateStore(db_path).commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True},
        source="test",
        submission_id="worker-go-back:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=db_path,
        mode="promote",
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["상세 완료 뒤 목록 복귀가 검증됨"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "task_category": "검색",
                "step_intents": [
                    {
                        "seq": item["seq"],
                        "action": item["action"],
                        "replay_mode": "fixed",
                    }
                    for item in payload["required_step_intents"]
                ],
            },
            "transition_contracts": [
                {
                    "seq": 2,
                    "contract": {
                        "common_ready_cues": [
                            {
                                "kind": "text_any",
                                "values": ["검색 결과"],
                            }
                        ],
                        "timeout_sec": 8.0,
                    },
                }
            ],
            "confidence": 0.9,
        },
    )

    assert review["promotion"]["promoted_followup_count"] == 1
    request, trace = select_followup_action(
        {
            "current_url": "https://www.wanted.co.kr/wd/999",
            "current_page_role": "job_detail",
            "recipe_params": {"task_category": "검색"},
        },
        trigger_action="finish_detail_reading",
        trigger_page_role="job_detail",
        page_role="job_detail",
        current_url="https://www.wanted.co.kr/wd/999",
        db_path=db_path,
    )
    assert trace["hit"] is True
    assert request is not None
    assert request.tool_calls[0].name == "go_back"


def test_pending_transition_skips_ocr_until_phash_changes(
    tmp_path,
    monkeypatch,
):
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(b"not-read")

    class FakePerception:
        def wait_for_transition_phash_change(
            self,
            _reference_phash,
            *,
            max_wait_sec=None,
        ):
            return False

        def capture_usable_screen(self, **_kwargs):
            raise AssertionError("같은 pHash에서는 파일 캡처를 하면 안 됩니다.")

    monkeypatch.setattr(
        worker_observation,
        "_perception_engine",
        lambda: FakePerception(),
    )
    request = {
        "action": "press_key",
        "action_seq": 2,
        "source": "followup_strategy",
        "pending_screen_phash": "0" * 16,
        "started_at": time.time(),
        "attempts": 1,
        "contract": {"timeout_sec": 8.0},
    }
    capture = worker_observation.capture_node(
        {
            "transition_request": request,
            "ocr_complete": True,
        }
    )
    assert capture == {
        "ocr_complete": False,
        "transition_probe_unchanged": True,
    }

    transition = worker_transition.transition_node(
        {
            "transition_request": request,
            "transition_probe_unchanged": True,
            "ocr_complete": False,
        }
    )
    assert transition["transition_result"]["status"] == "pending"
    assert transition["transition_result"]["needs_ocr"] is False
    assert transition["transition_request"]["attempts"] == 2


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
                "contract": {"timeout_sec": 8.0},
            },
            "ocr_complete": False,
        }
    )

    assert capture == {
        "ocr_complete": False,
        "transition_probe_unchanged": True,
    }


def test_candidate_promotion_blocks_no_effect_step(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _candidate_submission()
    submission["feedback_episodes"][0]["feedback"] = {
        "label": "no_effect",
        "reason": "screen_unchanged",
    }
    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True},
        source="test",
        submission_id="worker-no-effect:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        mode="promote",
        critic=lambda _payload: {
            "decision": "accept",
            "reasons": ["모델은 승인했지만 실행 증거가 실패임"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "task_category": "검색",
                "step_intents": [
                    {"seq": 0, "action": "click_marker", "replay_mode": "fixed"},
                ],
            },
            "confidence": 0.9,
        },
    )

    assert review["promotion"]["promoted"] is False
    assert review["promotion"]["skipped_steps"][0]["reason"] == "feedback_no_effect"
    assert RecipeStore(tmp_path / "critic.db").get_by_site("wanted") == []


def test_promotion_worker_stops_after_bounded_critic_failures(tmp_path, monkeypatch):
    from agent.application import recipe_promotion_service
    from agent.application.recipe_promotion_worker import RecipePromotionWorker
    from agent.recipe import candidate_reviewer
    from agent.recipe.candidate_store import RecipeCandidateStore

    db_path = tmp_path / "promotion.db"
    store = RecipeCandidateStore(db_path)
    candidate_id = store.commit_candidate(
        _candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True},
        source="test",
        submission_id="worker-retry:0",
    )
    monkeypatch.setenv("VISION_RECIPE_AUTO_PROMOTE", "1")
    assert recipe_promotion_service.schedule_recipe_candidate_promotion(
        candidate_id,
        db_path=db_path,
    )
    monkeypatch.setattr(
        candidate_reviewer,
        "review_and_apply_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            TimeoutError("critic timeout")
        ),
    )
    worker = RecipePromotionWorker(
        db_path,
        retry_delay_sec=0,
        max_attempts=2,
    )

    assert worker.process_one()["status"] == "pending_review"
    assert worker.process_one()["status"] == "review_failed"
    failed = store.get_candidate(candidate_id)
    assert failed["review_attempts"] == 2
    assert "critic timeout" in failed["review_error"]

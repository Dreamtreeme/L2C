import time

from agent.graph import (
    worker_execution_dispatch,
    worker_observation,
    worker_selection,
    worker_transition,
)
from agent.graph.worker_reflex import reflex_node
from agent.runtime.job_card_queue import replay_job_card_after_return


def test_reflex_replays_one_parameterized_roi_step(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw

    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import (
        RecipeAction,
        RecipeCheckpoint,
        RecipeTransition,
        SiteRecipe,
    )

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
                        start_state=RecipeCheckpoint(
                            url_template="wanted.co.kr/",
                            page_role="home",
                        ),
                        transitions=[
                            RecipeTransition(
                                seq=0,
                                before=RecipeCheckpoint(
                                    url_template="wanted.co.kr/",
                                    page_role="home",
                                ),
                                actions=[
                                    RecipeAction(
                                        source_seq=0,
                                        action="type_in_marker",
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
                                after=RecipeCheckpoint(
                                    url_template="wanted.co.kr/search",
                                    page_role="search_results",
                                    screen_context_signature={
                                        "phash": "f" * 16,
                                        "size": [200, 120],
                                    },
                                ),
                            )
                        ],
                        completion_state=RecipeCheckpoint(
                            url_template="wanted.co.kr/search",
                            page_role="search_results",
                            screen_context_signature={
                                "phash": "f" * 16,
                                "size": [200, 120],
                            },
                        ),
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


def test_reflex_replays_action_group_then_advances_after_verification(
    monkeypatch,
    tmp_path,
):
    from PIL import Image, ImageDraw

    from agent.vision.screen_signature import (
        compute_screen_signature,
        compute_target_roi_signature,
    )
    from shared.schema.recipe_schema import (
        RecipeAction,
        RecipeCheckpoint,
        RecipeTransition,
        SiteRecipe,
    )

    input_screen = tmp_path / "recipe-input.png"
    input_image = Image.new("RGB", (240, 120), "white")
    ImageDraw.Draw(input_image).rectangle(
        [10, 10, 130, 40],
        fill="black",
    )
    input_image.save(input_screen)
    result_screen = tmp_path / "recipe-result.png"
    result_image = Image.new("RGB", (240, 120), "white")
    ImageDraw.Draw(result_image).rectangle(
        [160, 10, 220, 40],
        fill="gray",
    )
    result_image.save(result_screen)

    input_roi = compute_target_roi_signature(
        input_screen,
        [10, 10, 130, 40],
        [240, 120],
    )
    result_roi = compute_target_roi_signature(
        result_screen,
        [160, 10, 220, 40],
        [240, 120],
    )
    input_context = compute_screen_signature(input_screen, [])
    result_context = compute_screen_signature(result_screen, [])
    input_target = {
        "text": "검색어",
        "bbox_ratio": [0.0417, 0.0833, 0.5417, 0.3333],
        "center_ratio": [0.2917, 0.2083],
    }
    result_target = {
        "text": "검색",
        "bbox_ratio": [0.6667, 0.0833, 0.9167, 0.3333],
        "center_ratio": [0.7917, 0.2083],
    }
    before = RecipeCheckpoint(
        url_template="saramin.co.kr/zf_user/",
        page_role="home",
        screen_context_signature=input_context,
        anchor_target=input_target,
        anchor_roi_signature=input_roi,
    )
    result_state = RecipeCheckpoint(
        url_template="saramin.co.kr/zf_user/",
        page_role="search_results",
        screen_context_signature=result_context,
        anchor_target=result_target,
        anchor_roi_signature=result_roi,
    )
    completion = RecipeCheckpoint(
        url_template="saramin.co.kr/zf_user/search",
        page_role="search_results",
        screen_context_signature=result_context,
    )
    recipe = SiteRecipe(
        site="saramin",
        goal="검색",
        start_state=before,
        transitions=[
            RecipeTransition(
                seq=0,
                before=before,
                actions=[
                    RecipeAction(
                        source_seq=1,
                        action="type_in_marker",
                        replay_mode="parameterized",
                        roi_signature=input_roi,
                        target=input_target,
                        param={"slot_name": "query"},
                        slot_refs=["query"],
                    ),
                    RecipeAction(
                        source_seq=2,
                        action="press_key",
                        replay_mode="fixed",
                        param={"key": "enter"},
                    ),
                ],
                after=result_state,
            ),
            RecipeTransition(
                seq=1,
                before=result_state,
                actions=[
                    RecipeAction(
                        source_seq=3,
                        action="click_marker",
                        replay_mode="fixed",
                        roi_signature=result_roi,
                        target=result_target,
                    )
                ],
                after=completion,
            ),
        ],
        completion_state=completion,
    )

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            return [("recipe-search-set", recipe)]

    monkeypatch.setattr(
        "agent.recipe.store.RecipeStore",
        lambda: FakeStore(),
    )
    first = reflex_node(
        {
            "goal": "AI 엔지니어 공고",
            "current_url": "https://www.saramin.co.kr/zf_user/",
            "current_page_role": "home",
            "screen_signature": input_context,
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
    assert first["pending_action"].summary == "cached recipe transition"
    assert [
        call.name for call in first["pending_action"].tool_calls
    ] == ["type_in_marker", "press_key"]
    assert (
        first["pending_action"].tool_calls[0].args["text"]
        == "AI 엔지니어"
    )
    assert (
        first["active_reflex_recipe"]["current_transition_index"]
        == 0
    )
    assert first["active_reflex_recipe"]["pending_transition_index"] == 0

    verified = worker_transition.transition_node(
        {
            "ocr_complete": True,
            "current_url": "https://www.saramin.co.kr/zf_user/",
            "current_screenshot": str(result_screen),
            "current_capture_id": "capture:0002",
            "current_markers": [
                {
                    "id": 8,
                    "bbox": [160, 10, 220, 40],
                    "text": "검색",
                },
            ],
            "screen_signature": result_context,
            "transition_request": {
                "action": "press_key",
                "source": "reflex",
                "recipe_key": "recipe-search-set",
                "before_url": "https://www.saramin.co.kr/zf_user/",
                "before_screenshot": str(input_screen),
                "expected_after_state": result_state.model_dump(),
                "recipe_transition_index": 0,
                "recipe_transition_count": 2,
                "started_at": time.time(),
            },
            "active_reflex_recipe": first["active_reflex_recipe"],
        }
    )

    assert verified["transition_result"]["status"] == "ready"
    assert (
        verified["active_reflex_recipe"]["current_transition_index"]
        == 1
    )
    assert (
        "pending_transition_index"
        not in verified["active_reflex_recipe"]
    )

    second = reflex_node(
        {
            "goal": "AI 엔지니어 공고",
            "current_url": "https://www.saramin.co.kr/zf_user/",
            "current_page_role": "search_results",
            "screen_signature": result_context,
            "recent_images": [result_screen],
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
            "active_reflex_recipe": verified[
                "active_reflex_recipe"
            ],
        }
    )

    assert second["reflex_trace"]["hit"] is True
    assert len(second["pending_action"].tool_calls) == 1
    assert second["pending_action"].tool_calls[0].name == "click_marker"
    assert second["pending_action"].tool_calls[0].args["marker_id"] == 8
    assert (
        second["active_reflex_recipe"]["current_transition_index"]
        == 1
    )


def test_active_reflex_recipe_is_cleared_only_after_final_transition():
    base_state = {
        "ocr_complete": True,
        "current_url": "https://www.saramin.co.kr/zf_user/search",
        "current_screenshot": "",
        "current_capture_id": "capture:0002",
        "current_markers": [
            {"id": 1, "bbox": [0, 0, 20, 20], "text": "검색 결과"},
        ],
        "screen_signature": {
            "phash": "a" * 16,
            "size": [1920, 1080],
        },
        "transition_request": {
            "action": "click_marker",
            "source": "reflex",
            "recipe_key": "recipe-search-set",
            "before_url": "https://www.saramin.co.kr/zf_user/",
            "expected_after_state": {
                "url_template": "saramin.co.kr/zf_user/search",
                "page_role": "search_results",
                "screen_context_signature": {
                    "phash": "a" * 16,
                    "size": [1920, 1080],
                },
            },
            "started_at": time.time(),
        },
    }

    intermediate = worker_transition.transition_node(
        {
            **base_state,
            "active_reflex_recipe": {
                "recipe_key": "recipe-search-set",
                "current_transition_index": 1,
                "pending_transition_index": 1,
                "transition_count": 3,
            },
        }
    )
    completed = worker_transition.transition_node(
        {
            **base_state,
            "active_reflex_recipe": {
                "recipe_key": "recipe-search-set",
                "current_transition_index": 2,
                "pending_transition_index": 2,
                "transition_count": 3,
            },
        }
    )

    assert intermediate["transition_result"]["status"] == "ready"
    assert (
        intermediate["active_reflex_recipe"][
            "current_transition_index"
        ]
        == 2
    )
    assert completed["transition_result"]["status"] == "ready"
    assert completed["active_reflex_recipe"] == {}

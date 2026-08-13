import time

from agent.graph import (
    worker_transition,
)
from agent.recipe.replay_runtime import attempt_reflex_replay as reflex_node
from agent.tests.worker_test_support import (
    node_runtime,
    worker_data_services,
    worker_state,
)
from shared.schema.collection_intent import CollectionIntent
from shared.schema.recipe_schema import ScreenCheckpoint


class _TargetPerception:
    def __init__(self, markers_by_screenshot):
        self.markers_by_screenshot = markers_by_screenshot

    def detect_target_roi(self, image_path, _crop_rect_ratio, marker_type):
        markers = self.markers_by_screenshot[image_path.name]
        if isinstance(markers, dict):
            return markers.get(marker_type, [])
        return markers


class _TargetVision:
    def __init__(self, markers_by_screenshot):
        self.perception = _TargetPerception(markers_by_screenshot)

    def get_perception(self):
        return self.perception


def test_reflex_replays_one_parameterized_roi_step(tmp_path):
    from PIL import Image, ImageDraw

    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import (
        ExperienceTransition,
        PhysicalAction,
        ScreenCheckpoint,
        SiteExperience,
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

    def load_site_recipes(site, *, task_category=None):
        assert site == "wanted"
        return [
            (
                "recipe-search",
                SiteExperience(
                    site="wanted",
                    goal="검색",
                    transitions=[
                        ExperienceTransition(
                            seq=0,
                            before=ScreenCheckpoint(
                                url_template="wanted.co.kr/",
                                page_role="home",
                            ),
                            actions=[
                                PhysicalAction(
                                    source_seq=0,
                                    action="type_in_marker",
                                    replay_mode="parameterized",
                                    roi_signature=roi_signature,
                                    target={
                                        "text": "검색",
                                        "marker_type": "text",
                                        "bbox_ratio": [0.05, 0.0833, 0.35, 0.3333],
                                        "center_ratio": [0.2, 0.2083],
                                    },
                                    param={"slot_name": "search_keyword"},
                                    slot_refs=["search_keyword"],
                                )
                            ],
                            after=ScreenCheckpoint(
                                url_template="wanted.co.kr/search",
                                page_role="search_results",
                                screen_context_signature={
                                    "phash": "f" * 16,
                                    "size": [200, 120],
                                },
                            ),
                        )
                    ],
                ),
            )
        ]

    result = reflex_node(
        worker_state(
            goal="AI 엔지니어 공고",
            observation={
                "current_url": "https://www.wanted.co.kr",
                "current_page_role": "home",
                "screen_signature": {"size": [200, 120]},
                "current_screenshot": str(screenshot),
                "current_markers": [
                    {
                        "id": 7,
                        "bbox": [10, 10, 70, 40],
                        "text": "지역 전체 JOB검색",
                        "type": "text",
                    },
                ],
            },
            request={
                "collection_intent": CollectionIntent(
                    site="wanted",
                    task_category="검색",
                    search_keyword="AI 엔지니어",
                ),
            },
        ),
        node_runtime(
            vision=_TargetVision(
                {
                    screenshot.name: [
                        {
                            "id": 0,
                            "bbox": [10, 10, 70, 40],
                            "text": "검색",
                            "type": "text",
                        }
                    ]
                }
            ),
            data=worker_data_services(
                load_site_recipes=load_site_recipes,
            ),
        ),
    )

    call = result["decision"]["pending_action"].tool_calls[0]
    assert result["replay"]["reflex_trace"]["hit"] is True
    assert call.name == "type_in_marker"
    assert call.args["marker_id"] == 8
    assert call.args["text"] == "AI 엔지니어"
    assert result["observation"]["current_markers"][-1]["text"] == "검색"
    assert len(result["decision"]["pending_action"].tool_calls) == 1


def test_reflex_uses_current_icon_marker_before_roi_redetection(tmp_path):
    from PIL import Image, ImageDraw

    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import (
        ExperienceTransition,
        PhysicalAction,
        SiteExperience,
    )

    screenshot = tmp_path / "icon-screen.png"
    image = Image.new("RGB", (200, 120), "white")
    ImageDraw.Draw(image).rectangle([130, 10, 150, 30], fill="black")
    image.save(screenshot)
    roi_signature = compute_target_roi_signature(
        screenshot,
        [130, 10, 150, 30],
        [200, 120],
    )
    target = {
        "text": "Q",
        "marker_type": "icon",
        "bbox_ratio": [0.65, 0.0833, 0.75, 0.25],
        "center_ratio": [0.7, 0.1667],
    }
    recipe = SiteExperience(
        site="saramin",
        transitions=[
            ExperienceTransition(
                seq=0,
                before={
                    "url_template": "saramin.co.kr/zf_user/",
                    "page_role": "home",
                },
                actions=[
                    PhysicalAction(
                        source_seq=0,
                        action="click_marker",
                        replay_mode="fixed",
                        roi_signature=roi_signature,
                        target=target,
                    )
                ],
                after={
                    "url_template": "saramin.co.kr/zf_user/",
                    "page_role": "home",
                    "screen_context_signature": {
                        "phash": "f" * 16,
                        "size": [200, 120],
                    },
                },
            )
        ],
    )

    result = reflex_node(
        worker_state(
            observation={
                "current_url": "https://www.saramin.co.kr/zf_user/",
                "current_page_role": "home",
                "screen_signature": {"size": [200, 120]},
                "current_screenshot": str(screenshot),
                "current_markers": [
                    {
                        "id": 17,
                        "bbox": [130, 10, 150, 30],
                        "text": "Q",
                        "type": "icon",
                    }
                ],
            },
            request={
                "collection_intent": CollectionIntent(
                    site="saramin",
                    task_category="검색",
                )
            },
        ),
        node_runtime(
            vision=_TargetVision({screenshot.name: {"icon": [], "text": []}}),
            data=worker_data_services(
                load_site_recipes=lambda _site, **_kwargs: [
                    ("recipe-icon", recipe)
                ]
            ),
        ),
    )

    request = result["decision"]["pending_action"]
    assert result["replay"]["reflex_trace"]["hit"] is True
    assert request.tool_calls[0].args["marker_id"] == 18
    assert result["replay"]["reflex_trace"]["tool_calls"][
        request.tool_calls[0].id
    ]["match_mode"] == "current_marker_ratio"


def test_reflex_replays_action_group_then_advances_after_verification(
    tmp_path,
):
    from PIL import Image, ImageDraw

    from agent.vision.screen_signature import (
        compute_screen_signature,
        compute_target_roi_signature,
    )
    from shared.schema.recipe_schema import (
        ExperienceTransition,
        PhysicalAction,
        ScreenCheckpoint,
        SiteExperience,
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
        "marker_type": "text",
        "bbox_ratio": [0.0417, 0.0833, 0.5417, 0.3333],
        "center_ratio": [0.2917, 0.2083],
    }
    result_target = {
        "text": "검색",
        "marker_type": "text",
        "bbox_ratio": [0.6667, 0.0833, 0.9167, 0.3333],
        "center_ratio": [0.7917, 0.2083],
    }
    before = ScreenCheckpoint(
        url_template="saramin.co.kr/zf_user/",
        page_role="home",
        screen_context_signature=input_context,
        anchor_target=input_target,
        anchor_roi_signature=input_roi,
    )
    result_state = ScreenCheckpoint(
        url_template="saramin.co.kr/zf_user/",
        page_role="search_results",
        screen_context_signature=result_context,
        anchor_target=result_target,
        anchor_roi_signature=result_roi,
    )
    completion = ScreenCheckpoint(
        url_template="saramin.co.kr/zf_user/search",
        page_role="search_results",
        screen_context_signature=result_context,
    )
    recipe = SiteExperience(
        site="saramin",
        goal="검색",
        transitions=[
            ExperienceTransition(
                seq=0,
                before=before,
                actions=[
                    PhysicalAction(
                        source_seq=1,
                        action="type_in_marker",
                        replay_mode="parameterized",
                        roi_signature=input_roi,
                        target=input_target,
                        param={"slot_name": "search_keyword"},
                        slot_refs=["search_keyword"],
                    ),
                    PhysicalAction(
                        source_seq=2,
                        action="press_key",
                        replay_mode="fixed",
                        param={"key": "enter"},
                    ),
                ],
                after=result_state,
            ),
            ExperienceTransition(
                seq=1,
                before=result_state,
                actions=[
                    PhysicalAction(
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
    )

    def load_site_recipes(_site, *, task_category=None):
        return [("recipe-search-set", recipe)]

    replay_runtime = node_runtime(
        vision=_TargetVision(
            {
                input_screen.name: [
                    {
                        "id": 0,
                        "bbox": [10, 10, 130, 40],
                        "text": "검색어",
                        "type": "text",
                    }
                ],
                result_screen.name: {
                    "text": [],
                    "icon": [
                        {
                            "id": 0,
                            "bbox": [160, 10, 220, 40],
                            "text": "icon",
                            "type": "icon",
                        }
                    ],
                },
            }
        ),
        data=worker_data_services(load_site_recipes=load_site_recipes),
    )
    first = reflex_node(
        worker_state(
            goal="AI 엔지니어 공고",
            observation={
                "current_url": "https://www.saramin.co.kr/zf_user/",
                "current_page_role": "home",
                "screen_signature": input_context,
                "current_screenshot": str(input_screen),
                "current_markers": [
                    {
                        "id": 7,
                        "bbox": [10, 10, 130, 40],
                        "text": "검색어",
                        "type": "text",
                    },
                ],
            },
            request={
                "collection_intent": CollectionIntent(
                    site="saramin",
                    task_category="검색",
                    search_keyword="AI 엔지니어",
                ),
            },
        ),
        replay_runtime,
    )

    assert first["replay"]["reflex_trace"]["hit"] is True
    assert first["decision"]["pending_action"].summary == "cached recipe transition"
    assert [call.name for call in first["decision"]["pending_action"].tool_calls] == [
        "type_in_marker",
        "press_key",
    ]
    assert (
        first["decision"]["pending_action"].tool_calls[0].args["text"] == "AI 엔지니어"
    )
    assert first["replay"]["replay_session"].current_transition_index == 0
    assert first["replay"]["replay_session"].pending_transition_index == 0

    verified = worker_transition.transition_node(
        worker_state(
            observation={
                "ocr_complete": True,
                "current_url": "https://www.saramin.co.kr/zf_user/",
                "current_screenshot": str(result_screen),
                "observation_id": "observation:0002",
                "current_markers": [
                    {
                        "id": 8,
                        "bbox": [160, 10, 220, 40],
                        "text": "검색",
                        "type": "text",
                    },
                ],
                "screen_signature": result_context,
            },
            transition={
                "transition_request": {
                    "action": "press_key",
                    "source": "reflex",
                    "recipe_key": "recipe-search-set",
                    "before_url": "https://www.saramin.co.kr/zf_user/",
                    "before_screenshot": str(input_screen),
                    "expected_after_state": result_state,
                    "recipe_transition_index": 0,
                    "recipe_transition_count": 2,
                    "started_at": time.time(),
                }
            },
            replay={"replay_session": first["replay"]["replay_session"]},
        ),
        replay_runtime,
    )

    assert verified["transition"]["transition_result"]["status"] == "ready"
    assert verified["replay"]["replay_session"].current_transition_index == 1
    assert verified["replay"]["replay_session"].pending_transition_index is None

    second = reflex_node(
        worker_state(
            goal="AI 엔지니어 공고",
            observation={
                "current_url": "https://www.saramin.co.kr/zf_user/",
                "current_page_role": "search_results",
                "screen_signature": result_context,
                "current_screenshot": str(result_screen),
                "current_markers": [
                    {
                        "id": 8,
                        "bbox": [160, 10, 220, 40],
                        "text": "검색",
                        "type": "text",
                    },
                ],
            },
            request={
                "collection_intent": CollectionIntent(
                    site="saramin",
                    task_category="검색",
                    search_keyword="AI 엔지니어",
                ),
            },
            replay={"replay_session": verified["replay"]["replay_session"]},
        ),
        replay_runtime,
    )

    assert second["replay"]["reflex_trace"]["hit"] is True
    assert len(second["decision"]["pending_action"].tool_calls) == 1
    assert second["decision"]["pending_action"].tool_calls[0].name == "click_marker"
    assert second["decision"]["pending_action"].tool_calls[0].args["marker_id"] == 9
    assert second["replay"]["replay_session"].current_transition_index == 1


def test_replay_success_is_recorded_only_after_final_transition():
    replay_results = []
    runtime = node_runtime(
        data=worker_data_services(
            record_recipe_replay=lambda recipe_key, succeeded: (
                replay_results.append((recipe_key, succeeded)) or True
            ),
        )
    )
    observation = {
        "ocr_complete": True,
        "current_url": "https://www.saramin.co.kr/zf_user/search",
        "current_screenshot": "",
        "observation_id": "observation:0002",
        "current_markers": [
            {"id": 1, "bbox": [0, 0, 20, 20], "text": "검색 결과"},
        ],
        "screen_signature": {
            "phash": "a" * 16,
            "size": [1920, 1080],
        },
    }
    transition_request = {
        "action": "click_marker",
        "source": "reflex",
        "recipe_key": "recipe-search-set",
        "before_url": "https://www.saramin.co.kr/zf_user/",
        "expected_after_state": ScreenCheckpoint(
            url_template="saramin.co.kr/zf_user/search",
            page_role="search_results",
            screen_context_signature={
                "phash": "a" * 16,
                "size": [1920, 1080],
            },
        ),
        "started_at": time.time(),
    }

    intermediate = worker_transition.transition_node(
        worker_state(
            observation=observation,
            transition={"transition_request": transition_request},
            replay={
                "replay_session": {
                    "recipe_key": "recipe-search-set",
                    "current_transition_index": 1,
                    "pending_transition_index": 1,
                    "transition_count": 3,
                }
            },
        ),
        runtime,
    )
    completed = worker_transition.transition_node(
        worker_state(
            observation=observation,
            transition={"transition_request": transition_request},
            replay={
                "replay_session": {
                    "recipe_key": "recipe-search-set",
                    "current_transition_index": 2,
                    "pending_transition_index": 2,
                    "transition_count": 3,
                }
            },
        ),
        runtime,
    )

    assert intermediate["transition"]["transition_result"]["status"] == "ready"
    assert intermediate["replay"]["replay_session"].current_transition_index == 2
    assert completed["transition"]["transition_result"]["status"] == "ready"
    assert completed["replay"]["replay_session"] is None
    assert replay_results == [("recipe-search-set", True)]

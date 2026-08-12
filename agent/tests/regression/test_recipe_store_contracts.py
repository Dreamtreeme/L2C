from agent.runtime.job_card_queue import replay_job_card_on_results
from agent.tests.worker_test_support import worker_state
from shared.schema.feedback_schema import RecipeCandidate, WorkerSubmission
from shared.schema.skill_schema import RecipeSkillMetadata


def _recipe_metadata(task_category: str) -> RecipeSkillMetadata:
    return RecipeSkillMetadata(task_category=task_category)


def _recipe_candidate(steps: list[dict], transitions: list[dict]) -> RecipeCandidate:
    transitions_by_seq = {
        transition.get("action_seq"): transition for transition in transitions
    }
    submission = WorkerSubmission(
        run_id="recipe-path-test",
        collection_intent={"site": "example"},
        action_events=[
            {
                "seq": int(step.get("seq") or 0),
                "recipe_step": step,
                "transition": transitions_by_seq.get(step.get("seq")),
            }
            for step in steps
        ],
    )
    return RecipeCandidate(
        run_id="recipe-path-test",
        status="pending_replay",
        submission=submission,
    )


def _active_recipe_path(actions: list[dict]) -> dict:
    """저장소 계약 테스트용 상태 전이 경로를 만든다."""

    converted = []
    for index, raw in enumerate(actions):
        action = {
            "source_seq": int(raw.get("seq", index)),
            "action": raw["action"],
            "target": raw.get("target"),
            "roi_signature": raw.get("roi_signature", {}),
            "value": raw.get("value"),
            "param": raw.get("param", {}),
            "is_param": bool(raw.get("is_param")),
            "intent": raw.get("intent", ""),
            "target_role": raw.get("target_role", ""),
            "component": raw.get("component", ""),
            "slot_refs": raw.get("slot_refs", []),
            "replay_mode": raw.get("replay_mode", "fixed"),
        }
        before = {
            "url_template": raw.get(
                "url_template",
                "example.com/search",
            ),
            "page_role": raw.get("page_role", "search_results"),
            "screen_context_signature": {
                "phash": f"{index + 1:x}" * 16,
                "size": [1920, 1080],
            },
        }
        if action["target"] and action["roi_signature"]:
            before["anchor_target"] = action["target"]
            before["anchor_roi_signature"] = action["roi_signature"]
        converted.append((action, before))

    transitions = []
    for index, (action, before) in enumerate(converted):
        if index + 1 < len(converted):
            after = dict(converted[index + 1][1])
        else:
            after = {
                "url_template": before["url_template"],
                "page_role": before["page_role"],
                "screen_context_signature": {
                    "phash": "f" * 16,
                    "size": [1920, 1080],
                },
            }
        transitions.append(
            {
                "seq": index,
                "before": before,
                "actions": [action],
                "after": after,
            }
        )
    return {
        "start_state": transitions[0]["before"],
        "transitions": transitions,
        "completion_state": transitions[-1]["after"],
    }


def test_result_queue_replays_cached_card_on_results_screen(tmp_path):
    from PIL import Image, ImageDraw

    from agent.vision.screen_signature import compute_target_roi_signature

    original_path = tmp_path / "original.png"
    changed_path = tmp_path / "changed.png"
    original = Image.new("RGB", (1000, 1000), "white")
    original_draw = ImageDraw.Draw(original)
    original_draw.rectangle((300, 400, 500, 450), fill="black")
    original.save(original_path)
    changed = Image.new("RGB", (1000, 1000), "white")
    changed_draw = ImageDraw.Draw(changed)
    for x in range(280, 521, 20):
        changed_draw.line((x, 370, x, 480), fill="black", width=5)
    changed.save(changed_path)
    roi_signature = compute_target_roi_signature(
        original_path,
        [300, 400, 500, 450],
        [1000, 1000],
    )

    state = worker_state(
        collection={
            "job_card_queue": [
                {
                    "queue_id": "card-2",
                    "status": "pending",
                    "title": "두 번째 iOS 개발자",
                    "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                    "center_ratio": [0.4, 0.425],
                    "roi_signature": roi_signature,
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
    )

    request, markers, trace = replay_job_card_on_results(
        state,
        {"action": "go_back"},
        "https://www.wanted.co.kr/search?query=ios",
        [],
        {
            "phash": "0" * 16,
            "size": [1000, 1000],
            "anchors": ["두 번째 iOS 개발자"],
        },
        require_anchors=False,
        current_image_path=str(original_path),
    )

    assert request is not None
    assert trace["hit"] is True
    assert request.tool_calls[0].name == "click_marker"
    assert markers[0]["bbox"] == [300, 400, 500, 450]

    missed_request, _, missed_trace = replay_job_card_on_results(
        state,
        {"action": "go_back"},
        "https://www.wanted.co.kr/search?query=ios",
        [],
        {
            "phash": "0" * 16,
            "size": [1000, 1000],
            "anchors": ["두 번째 iOS 개발자"],
        },
        require_anchors=False,
        current_image_path=str(changed_path),
    )

    assert missed_request is None
    assert missed_trace["reason"] == "roi_phash_distance"


def test_recipe_store_scopes_by_site_and_task_category(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "recipes.db")
    store.commit_recipe_path(
        "wanted",
        "검색",
        _active_recipe_path(
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
            ]
        ),
        metadata=_recipe_metadata("검색"),
    )

    assert len(store.get_site_recipes("wanted", task_category="검색")) == 1
    assert store.get_site_recipes("wanted", task_category="로그인") == []
    assert store.get_site_recipes("saramin", task_category="검색") == []


def test_recipe_store_keeps_actions_in_one_path_across_page_changes(tmp_path):
    from agent.recipe.store import RecipeStore

    cases = [
        ("same-page", "saramin.co.kr/zf_user/", "home"),
        ("cross-page", "saramin.co.kr/zf_user/search", "search"),
    ]
    for case_name, second_url, second_role in cases:
        store = RecipeStore(tmp_path / f"{case_name}.db")
        saved = store.commit_recipe_path(
            "saramin",
            "검색",
            _active_recipe_path(
                [
                    {
                        "seq": 1,
                        "url_template": "saramin.co.kr/zf_user/",
                        "page_role": "home",
                        "action": "type_in_marker",
                        "replay_mode": "parameterized",
                        "slot_refs": ["search_keyword"],
                        "param": {"slot_name": "search_keyword"},
                        "target": {"text": "검색어"},
                        "roi_signature": {"phash": "0" * 16},
                    },
                    {
                        "seq": 2,
                        "url_template": second_url,
                        "page_role": second_role,
                        "action": "click_marker",
                        "replay_mode": "fixed",
                        "target": {"text": "검색"},
                        "roi_signature": {"phash": "1" * 16},
                    },
                ]
            ),
            metadata=_recipe_metadata("검색"),
        )

        recipes = store.get_by_site("saramin")
        actions = [
            transition["actions"][0]["action"]
            for transition in recipes[0]["transitions"]
        ]
        assert saved == 1, case_name
        assert len(recipes) == 1, case_name
        assert actions == ["type_in_marker", "click_marker"], case_name


def test_recipe_path_accepts_page_role_change_without_full_screen_hash(
    tmp_path,
):
    from agent.recipe.path_builder import build_recipe_path
    from agent.recipe.store import RecipeStore

    before = {
        "observation_id": "observation:0001",
        "url_template": "example.com/search",
        "page_role": "search_overlay",
        "screen_context_signature": {
            "phash": "1" * 16,
            "size": [1920, 1080],
        },
    }
    after = {
        "observation_id": "observation:0002",
        "url_template": "example.com/search",
        "page_role": "search",
    }
    step = {
        "seq": 1,
        "action": "click_marker",
        "replay_mode": "fixed",
        "target": {"text": "검색", "center_ratio": [0.5, 0.5]},
        "roi_signature": {
            "phash": "2" * 16,
            "crop_rect_ratio": [0.4, 0.4, 0.6, 0.6],
        },
        "before_state": before,
    }
    candidate = _recipe_candidate(
        [step],
        [{"action_seq": 1, "after_state": after}],
    )

    path, issues = build_recipe_path(candidate, [step])

    assert path is not None
    assert issues == []
    assert path["completion_state"]["screen_context_signature"] == {}
    assert (
        RecipeStore(tmp_path / "role-change.db").commit_recipe_path(
            "example",
            "검색",
            path,
            metadata=_recipe_metadata("검색"),
        )
        == 1
    )


def test_recipe_path_validates_screen_changing_targets_on_separate_screens():
    from agent.recipe.path_builder import build_recipe_path

    def state(observation_id: str, role: str, phash: str) -> dict:
        return {
            "observation_id": observation_id,
            "url_template": "example.com/",
            "page_role": role,
            "screen_context_signature": {
                "phash": phash,
                "size": [1920, 1080],
            },
        }

    type_step = {
        "seq": 1,
        "action": "type_in_marker",
        "replay_mode": "parameterized",
        "param": {"slot_name": "search_keyword"},
        "slot_refs": ["search_keyword"],
        "target": {"text": "검색어", "center_ratio": [0.5, 0.1]},
        "roi_signature": {"phash": "1" * 16},
        "before_state": state("observation:0001", "home", "1" * 16),
    }
    click_step = {
        "seq": 2,
        "action": "click_marker",
        "replay_mode": "fixed",
        "target": {"text": "검색", "center_ratio": [0.9, 0.1]},
        "roi_signature": {"phash": "2" * 16},
        "before_state": state("observation:0002", "search_overlay", "2" * 16),
    }
    candidate = _recipe_candidate(
        [type_step, click_step],
        [
            {
                "action_seq": 1,
                "after_state": state(
                    "observation:0002", "search_overlay", "2" * 16
                ),
            },
            {
                "action_seq": 2,
                "after_state": state("observation:0003", "search", "3" * 16),
            },
        ],
    )

    path, issues = build_recipe_path(candidate, [type_step, click_step])

    assert issues == []
    assert [
        [action["action"] for action in transition["actions"]]
        for transition in path["transitions"]
    ] == [["type_in_marker"], ["click_marker"]]
    assert path["transitions"][0]["after"]["anchor_target"] == click_step["target"]


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

    assert (
        store.replace_recipe_paths(
            "example",
            "탐색",
            [_active_recipe_path(first_path)],
            metadata=_recipe_metadata("사이트 탐색"),
            source_run_id="run-abc",
        )
        == 1
    )
    assert (
        store.replace_recipe_paths(
            "example",
            "탐색",
            [_active_recipe_path(second_path)],
            metadata=_recipe_metadata("사이트 탐색"),
            source_run_id="run-adc",
        )
        == 1
    )

    recipes = store.get_by_site("example")
    assert len(recipes) == 2
    assert {
        tuple(
            transition["actions"][0]["target"]["text"]
            for transition in recipe["transitions"]
        )
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
    for run_id in ("run-one", "run-two"):
        assert (
            store.replace_recipe_paths(
                "example",
                "탐색",
                [_active_recipe_path(path("B"))],
                metadata=_recipe_metadata("사이트 탐색"),
                source_run_id=run_id,
            )
            == 1
        )

    shared = store.get_by_site("example")
    assert len(shared) == 1
    assert shared[0]["support_count"] == 2
    assert shared[0]["source_count"] == 2

    assert (
        store.replace_recipe_paths(
            "example",
            "탐색",
            [_active_recipe_path(path("D"))],
            metadata=_recipe_metadata("사이트 탐색"),
            source_run_id="run-two",
        )
        == 1
    )

    recipes = store.get_by_site("example")
    assert len(recipes) == 2
    assert {
        tuple(
            transition["actions"][0]["target"]["text"]
            for transition in recipe["transitions"]
        ): (
            recipe["support_count"],
            recipe["source_count"],
        )
        for recipe in recipes
    } == {
        ("A", "B", "C"): (1, 1),
        ("A", "D", "C"): (1, 1),
    }


def test_recipe_store_separates_candidate_support_from_replay_results(tmp_path):
    from agent.recipe.store import RecipeStore

    path = _active_recipe_path(
        [
            {
                "seq": 1,
                "action": "click_marker",
                "replay_mode": "fixed",
                "target": {"text": "검색"},
                "roi_signature": {"phash": "1" * 16},
            }
        ]
    )
    store = RecipeStore(tmp_path / "replay-results.db")
    assert store.replace_recipe_paths(
        "example",
        "검색",
        [path],
        metadata=_recipe_metadata("검색"),
        source_run_id="run-one",
    ) == 1
    recipe_key = store.get_by_site("example")[0]["recipe_key"]

    assert store.record_replay_result(recipe_key, True) is True
    assert store.record_replay_result(recipe_key, False) is True

    recipe = store.get_by_site("example")[0]
    assert recipe["support_count"] == 1
    assert recipe["replay_success_count"] == 1
    assert recipe["replay_failure_count"] == 1
    assert recipe["last_replayed_at"]


def test_critic_does_not_join_different_recorded_observations():
    from agent.recipe.path_builder import build_recipe_path

    def state(observation_id: str, phash: str) -> dict:
        return {
            "observation_id": observation_id,
            "url_template": "example.com/search",
            "page_role": "search_results",
            "screen_context_signature": {
                "phash": phash,
                "size": [1920, 1080],
            },
        }

    first = {
        "seq": 1,
        "action": "click_marker",
        "page_role": "search_results",
        "replay_mode": "fixed",
        "target": {"text": "A"},
        "roi_signature": {"phash": "1" * 16},
        "before_state": state("observation:0001", "1" * 16),
    }
    removed = {
        "seq": 2,
        "action": "press_key",
        "page_role": "search_results",
        "replay_mode": "reasoning",
        "param": {"key": "tab"},
        "before_state": state("observation:0002", "2" * 16),
    }
    last = {
        "seq": 3,
        "action": "click_marker",
        "page_role": "search_results",
        "replay_mode": "fixed",
        "target": {"text": "B"},
        "roi_signature": {"phash": "3" * 16},
        "before_state": state("observation:0003", "2" * 16),
    }
    candidate = _recipe_candidate(
        [first, removed, last],
        [
            {
                "action_seq": 1,
                "after_state": state("observation:0002", "2" * 16),
            },
            {
                "action_seq": 3,
                "after_state": state("observation:0004", "4" * 16),
            },
        ],
    )

    path, issues = build_recipe_path(candidate, [first, last])

    assert path is not None
    assert [
        transition["actions"][0]["source_seq"] for transition in path["transitions"]
    ] == [1]
    assert issues == [
        {
            "seq": 3,
            "action": "click_marker",
            "reason": "state_continuity_unproven",
        }
    ]

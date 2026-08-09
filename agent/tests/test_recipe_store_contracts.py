from agent.runtime.job_card_queue import replay_job_card_on_results
from agent.tests.worker_test_support import worker_state
from shared.schema.feedback_schema import RecipeCandidate, WorkerSubmission
from shared.schema.skill_schema import RecipeSkillMetadata


def _recipe_metadata(task_category: str) -> RecipeSkillMetadata:
    return RecipeSkillMetadata(task_category=task_category)


def _recipe_candidate(steps: list[dict], transitions: list[dict]) -> RecipeCandidate:
    submission = WorkerSubmission(
        run_id="recipe-path-test",
        collection_intent={"site": "example"},
        recorded_steps=steps,
        transition_records=transitions,
    )
    return RecipeCandidate(
        candidate_id="recipe-path-test",
        submission_id="recipe-path-test",
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


def test_result_queue_replays_cached_card_on_results_screen():
    state = worker_state(
        collection={
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
    )

    assert request is not None
    assert trace["hit"] is True
    assert request.tool_calls[0].name == "click_marker"
    assert markers[0]["bbox"] == [300, 400, 500, 450]


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


def test_recipe_store_saves_input_and_submit_as_one_path(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "recipe-path.db")
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
            ]
        ),
        metadata=_recipe_metadata("검색"),
    )

    recipes = store.get_by_site("saramin")

    assert saved == 1
    assert len(recipes) == 1
    assert [
        transition["actions"][0]["action"] for transition in recipes[0]["transitions"]
    ] == ["type_in_marker", "click_marker"]


def test_recipe_store_keeps_cross_page_steps_in_one_path(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "separate-actions.db")
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
            ]
        ),
        metadata=_recipe_metadata("검색"),
    )

    recipes = store.get_by_site("saramin")

    assert saved == 1
    assert len(recipes) == 1
    assert len(recipes[0]["transitions"]) == 2


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
            candidate_id="candidate-abc",
        )
        == 1
    )
    assert (
        store.replace_recipe_paths(
            "example",
            "탐색",
            [_active_recipe_path(second_path)],
            metadata=_recipe_metadata("사이트 탐색"),
            candidate_id="candidate-adc",
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
    for candidate_id in ("candidate-one", "candidate-two"):
        assert (
            store.replace_recipe_paths(
                "example",
                "탐색",
                [_active_recipe_path(path("B"))],
                metadata=_recipe_metadata("사이트 탐색"),
                candidate_id=candidate_id,
            )
            == 1
        )

    shared = store.get_by_site("example")
    assert len(shared) == 1
    assert shared[0]["success_count"] == 2
    assert shared[0]["source_count"] == 2

    assert (
        store.replace_recipe_paths(
            "example",
            "탐색",
            [_active_recipe_path(path("D"))],
            metadata=_recipe_metadata("사이트 탐색"),
            candidate_id="candidate-two",
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
            recipe["success_count"],
            recipe["source_count"],
        )
        for recipe in recipes
    } == {
        ("A", "B", "C"): (1, 1),
        ("A", "D", "C"): (1, 1),
    }


def test_critic_gap_does_not_create_a_standalone_suffix_path():
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

    steps = [
        {
            "seq": 1,
            "action": "click_marker",
            "page_role": "search_results",
            "replay_mode": "fixed",
            "target": {"text": "A"},
            "roi_signature": {"phash": "1" * 16},
            "before_state": state("observation:0001", "1" * 16),
        },
        {
            "seq": 2,
            "action": "press_key",
            "page_role": "search_results",
            "replay_mode": "fixed",
            "param": {"key": "tab"},
            "screen_context_signature": {
                "phash": "2" * 16,
                "size": [1920, 1080],
            },
            "before_state": state("observation:0002", "2" * 16),
        },
        {
            "seq": 4,
            "action": "click_marker",
            "page_role": "search_results",
            "replay_mode": "fixed",
            "target": {"text": "B"},
            "roi_signature": {"phash": "4" * 16},
            "before_state": state("observation:0004", "4" * 16),
        },
    ]
    candidate = _recipe_candidate(
        steps,
        [
            {
                "action_seq": 1,
                "after_state": state("observation:0002", "2" * 16),
            },
            {
                "action_seq": 2,
                "after_state": state("observation:0003", "3" * 16),
            },
            {
                "action_seq": 4,
                "after_state": state("observation:0005", "5" * 16),
            },
        ],
    )

    path, issues = build_recipe_path(candidate, steps)

    assert path is not None
    assert [
        transition["actions"][0]["source_seq"] for transition in path["transitions"]
    ] == [1, 2]
    assert any(
        issue["seq"] == 4 and issue["reason"] == "state_continuity_unproven"
        for issue in issues
    )


def test_critic_can_remove_action_when_state_continuity_is_proven():
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
    ] == [1, 3]
    assert not any(issue["reason"] == "state_continuity_unproven" for issue in issues)

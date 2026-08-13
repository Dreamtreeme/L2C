import sqlite3

from shared.schema.recipe_schema import ExperiencePath
from shared.schema.skill_schema import RecipeSkillMetadata


def _recipe_metadata(task_category: str) -> RecipeSkillMetadata:
    return RecipeSkillMetadata(task_category=task_category)


def _active_recipe_path(actions: list[dict]) -> ExperiencePath:
    """저장소 계약 테스트용 상태 전이 경로를 만든다."""

    converted = []
    for index, raw in enumerate(actions):
        action = {
            "source_seq": int(raw.get("seq", index)),
            "action": raw["action"],
            "target": raw.get("target"),
            "roi_signature": raw.get("roi_signature", {}),
            "param": raw.get("param", {}),
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
    return ExperiencePath.model_validate({"transitions": transitions})


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


def test_recipe_store_ignores_legacy_paths_without_deleting_them(tmp_path):
    from agent.recipe.store import RecipeStore

    db_path = tmp_path / "legacy-recipes.db"
    store = RecipeStore(db_path)
    path = _active_recipe_path(
        [
            {
                "seq": 0,
                "action": "click_marker",
                "target": {"text": "검색", "center_ratio": [0.8, 0.1]},
                "roi_signature": {"phash": "0" * 16},
            }
        ]
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO recipes "
            "(recipe_key, site, goal, path_json, metadata_json, support_count, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                "experience7#legacy",
                "wanted",
                "검색",
                path.model_dump_json(),
                "{}",
                1,
                "2026-08-13T00:00:00",
                "2026-08-13T00:00:00",
            ),
        )

    assert store.get_by_site("wanted") == []
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM recipes WHERE recipe_key='experience7#legacy'"
        ).fetchone()[0] == 1


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
    assert (
        store.replace_recipe_paths(
            "example",
            "검색",
            [path],
            metadata=_recipe_metadata("검색"),
            source_run_id="run-one",
        )
        == 1
    )
    recipe_key = store.get_by_site("example")[0]["recipe_key"]

    assert store.record_replay_result(recipe_key, True) is True
    assert store.record_replay_result(recipe_key, False) is True

    recipe = store.get_by_site("example")[0]
    assert recipe["support_count"] == 1
    assert recipe["replay_success_count"] == 1
    assert recipe["replay_failure_count"] == 1
    assert recipe["last_replayed_at"]

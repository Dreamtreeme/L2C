import sqlite3

from langchain_core.messages import AIMessage


def test_record_ui_step_stays_in_marker_text_space():
    from agent.recipe.record import record_ui_step

    steps = []
    state = {
        "goal": "지원하기",
        "current_url": "https://www.wanted.co.kr/wd/12345",
        "current_markers": [
            {"id": 1, "bbox": [20, 20, 120, 60], "text": "지원하기"},
            {"id": 2, "bbox": [20, 80, 120, 120], "text": "공유하기"},
        ],
    }

    record_ui_step(steps, state, "click_marker", {"marker_id": 1}, 0)

    assert steps[0]["state_key"].startswith("wanted.co.kr/wd/{id}#")
    assert steps[0]["target"] == {
        "text": "지원하기",
        "region": "top-left",
        "ordinal": 0,
        "evidence_texts": ["공유하기"],
    }
    assert "bbox" not in steps[0]["target"]


def test_state_key_and_ocr_delta_ignore_dynamic_numeric_changes():
    from agent.recipe.ocr_delta import diff_marker_texts, marker_text_set
    from agent.recipe.state_key import compute_state_key

    before = [
        {"id": 1, "bbox": [0, 0, 10, 10], "text": "추천 0"},
        {"id": 2, "bbox": [0, 20, 10, 30], "text": "지원하기"},
        {"id": 3, "bbox": [0, 40, 10, 50], "text": "회사 소개"},
    ]
    after_count_change = [
        {"id": 1, "bbox": [0, 0, 10, 10], "text": "추천 1"},
        {"id": 2, "bbox": [0, 20, 10, 30], "text": "지원하기"},
        {"id": 3, "bbox": [0, 40, 10, 50], "text": "회사 소개"},
    ]
    after_new_element = after_count_change + [
        {"id": 4, "bbox": [0, 60, 10, 70], "text": "지원 완료"},
    ]

    url = "https://www.wanted.co.kr/wd/12345"
    assert compute_state_key(url, before) == compute_state_key(url, after_count_change)

    count_delta = diff_marker_texts(marker_text_set(before), after_count_change)
    assert count_delta["added"] == []
    assert count_delta["removed"] == []

    new_delta = diff_marker_texts(marker_text_set(before), after_new_element)
    assert new_delta["added"] == ["지원 완료"]
    assert new_delta["removed"] == []


def test_match_marker_uses_region_and_ordinal_tiebreak():
    from agent.recipe.matcher import match_marker

    markers = [
        {"id": 1, "bbox": [10, 10, 60, 40], "text": "검색"},
        {"id": 2, "bbox": [10, 80, 60, 110], "text": "검색"},
    ]
    step = {
        "target": {
            "text": "검색",
            "region": "bottom-left",
            "ordinal": 0,
        }
    }

    assert match_marker(step, markers) == 2


def test_recipe_store_commits_and_reads_by_state_key(tmp_path):
    from agent.recipe.store import RecipeStore

    db_path = tmp_path / "recipes.db"
    store = RecipeStore(db_path)
    saved = store.commit_recipe(
        "wanted.co.kr",
        "goal",
        [
            {
                "seq": 0,
                "state_key": "state-a",
                "url_template": "wanted.co.kr",
                "action": "click_marker",
                "target": {"text": "검색", "region": "top-left", "ordinal": 0},
                "param": {},
                "expected_next_state": "state-b",
            }
        ],
    )

    assert saved == 1
    recipe = store.get_recipe("state-a")
    assert recipe is not None
    assert recipe.steps[0].action == "click_marker"
    assert recipe.steps[0].expected_next_state == "state-b"

    conn = sqlite3.connect(db_path)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(recipes)").fetchall()]
    conn.close()
    assert "state_key" in columns
    assert "recipe_key" not in columns


def test_recipe_store_groups_same_state_action_chain(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "recipes.db")
    store.commit_recipe(
        "wanted.co.kr",
        "goal",
        [
            {
                "seq": 0,
                "state_key": "search-state",
                "action": "type_in_marker",
                "target": {"text": "검색", "region": "top-left", "ordinal": 0},
                "param": {"text": "ai 엔지니어"},
            },
            {
                "seq": 1,
                "state_key": "search-state",
                "action": "press_key",
                "param": {"key": "enter"},
            },
            {
                "seq": 2,
                "state_key": "results-state",
                "action": "click_marker",
                "target": {"text": "공고", "region": "middle-left", "ordinal": 0},
                "param": {},
            },
        ],
    )

    recipe = store.get_recipe("search-state")

    assert recipe is not None
    assert [step.action for step in recipe.steps] == ["type_in_marker", "press_key"]
    assert recipe.steps[0].expected_next_state == "results-state"
    assert recipe.steps[1].expected_next_state == "results-state"


def test_reflex_node_builds_action_tool_call(monkeypatch, tmp_path):
    from agent.graph import nodes
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    class FakeStore:
        def get_recipe(self, state_key):
            assert state_key == "state-a"
            return SiteRecipe(
                site="wanted.co.kr",
                goal="goal",
                steps=[
                    RecipeStep(
                        seq=0,
                        state_key="state-a",
                        action="type_in_marker",
                        target={"text": "검색", "region": "top-left", "ordinal": 0},
                        param={"text": "ai 엔지니어"},
                        expected_next_state="state-b",
                    ),
                    RecipeStep(
                        seq=1,
                        state_key="state-a",
                        action="press_key",
                        param={"key": "enter"},
                        expected_next_state="state-b",
                    ),
                ],
            )

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = nodes.reflex_node(
        {
            "goal": "ai 엔지니어 공고 찾아줘",
            "current_url": "https://www.wanted.co.kr",
            "reflex_state_key": "state-a",
            "current_markers": [{"id": 7, "bbox": [10, 10, 70, 40], "text": "검색"}],
        }
    )

    msg = result["last_action_result"]
    assert result["reflex_hit"] is True
    assert result["reflex_expected_next_state"] == "state-b"
    assert msg.tool_calls[0]["name"] == "type_in_marker"
    assert msg.tool_calls[0]["args"] == {"marker_id": 7, "text": "ai 엔지니어"}
    assert msg.tool_calls[1]["name"] == "press_key"
    assert msg.tool_calls[1]["args"] == {"key": "enter"}


def test_action_node_commits_accumulated_recorded_steps(monkeypatch):
    from agent.graph import nodes
    from agent.recipe import record

    seen = {}

    class FakeTools:
        def finish_task(self, result):
            return {"status": "success", "action": "finish_task", "result": result}

    monkeypatch.setattr(nodes, "_get_action_tools", lambda: FakeTools())
    monkeypatch.setattr(
        record,
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
            "expected_next_state": None,
        }
    ]

    result = nodes.action_node(
        {
            "current_markers": [],
            "current_url": "https://www.wanted.co.kr",
            "current_url_stale": False,
            "extracted_jd": {},
            "is_finished": False,
            "collected_data": [],
            "error_count": 0,
            "current_plan_step": 0,
            "plan": [],
            "recorded_steps": prior_steps,
            "last_action_result": AIMessage(
                content="",
                tool_calls=[{"name": "finish_task", "args": {"result": "done"}, "id": "1"}],
            ),
        }
    )

    assert result["is_finished"] is True
    assert seen["steps"] == prior_steps
    assert seen["current_url"] == "https://www.wanted.co.kr"


def test_reflex_routing_respects_flag_and_validation(monkeypatch):
    from agent.graph.workflow import route_after_perception, route_after_reflex

    monkeypatch.delenv("REFLEX_ENABLED", raising=False)
    assert route_after_perception({}) == "reasoning"

    monkeypatch.setenv("REFLEX_ENABLED", "1")
    assert route_after_perception({"reflex_pending_validation": False}) == "reflex"
    assert route_after_perception(
        {
            "reflex_pending_validation": True,
            "reflex_expected_next_state": "expected",
            "reflex_state_key": "current",
        }
    ) == "reasoning"

    assert route_after_reflex({"reflex_hit": True}) == "action"
    assert route_after_reflex({"reflex_hit": False}) == "reasoning"

def test_match_marker_uses_evidence_texts_to_disambiguate_repeated_targets():
    from agent.recipe.matcher import match_marker

    markers = [
        {"id": 1, "bbox": [100, 100, 180, 130], "text": "Open"},
        {"id": 2, "bbox": [100, 140, 240, 170], "text": "Alpha Item"},
        {"id": 3, "bbox": [100, 300, 180, 330], "text": "Open"},
        {"id": 4, "bbox": [100, 340, 240, 370], "text": "Beta Item"},
    ]
    step = {
        "target": {
            "text": "Open",
            "region": "top-left",
            "ordinal": 0,
            "evidence_texts": ["Beta Item"],
        }
    }

    assert match_marker(step, markers) == 3


def test_match_marker_returns_none_when_evidence_is_ambiguous():
    from agent.recipe.matcher import match_marker

    markers = [
        {"id": 1, "bbox": [100, 100, 180, 130], "text": "Open"},
        {"id": 2, "bbox": [100, 140, 240, 170], "text": "Shared Label"},
        {"id": 3, "bbox": [100, 300, 180, 330], "text": "Open"},
        {"id": 4, "bbox": [100, 340, 240, 370], "text": "Shared Label"},
    ]
    step = {
        "target": {
            "text": "Open",
            "region": "top-left",
            "ordinal": 0,
            "evidence_texts": ["Shared Label"],
        }
    }

    assert match_marker(step, markers) is None


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


def test_match_marker_prefers_llm_selected_card_title():
    from agent.recipe.matcher import match_marker

    markers = [
        {"id": 1, "bbox": [100, 100, 180, 130], "text": "Reward 100"},
        {"id": 2, "bbox": [100, 140, 280, 170], "text": "Alpha iOS Developer"},
        {"id": 3, "bbox": [100, 300, 180, 330], "text": "Reward 100"},
        {"id": 4, "bbox": [100, 340, 280, 370], "text": "Beta iOS Developer"},
    ]
    step = {
        "target": {
            "text": "Reward 100",
            "semantic_label": "Beta iOS Developer",
        }
    }

    assert match_marker(step, markers) == 4


def test_reflex_routes_to_reasoning_after_validated_hit_by_default(monkeypatch):
    from agent.graph.workflow import route_after_perception

    monkeypatch.setenv("REFLEX_ENABLED", "1")
    monkeypatch.delenv("REFLEX_REASON_AFTER_HIT", raising=False)
    state = {
        "reflex_pending_validation": True,
        "reflex_expected_next_state": "state-detail",
        "reflex_state_key": "state-detail",
    }

    assert route_after_perception(state) == "reasoning"

    monkeypatch.setenv("REFLEX_REASON_AFTER_HIT", "0")
    assert route_after_perception(state) == "reflex"
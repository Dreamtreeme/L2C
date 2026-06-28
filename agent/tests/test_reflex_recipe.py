import sqlite3

from langchain_core.messages import AIMessage


def test_record_ui_step_stays_in_marker_text_space():
    from agent.recipe.record import record_ui_step

    steps = []
    state = {
        "goal": "지원하기",
        "current_url": "https://www.wanted.co.kr/wd/12345",
        "screen_signature": {
            "phash": "0" * 16,
            "size": [200, 200],
            "anchors": ["지원하기", "공유하기"],
        },
        "current_markers": [
            {"id": 1, "bbox": [20, 20, 120, 60], "text": "지원하기"},
            {"id": 2, "bbox": [20, 80, 120, 120], "text": "공유하기"},
        ],
    }

    record_ui_step(
        steps,
        state,
        "click_marker",
        {
            "marker_id": 1,
            "reason": "open apply flow",
            "target_role": "apply_button",
            "target_component": "job_detail_header",
            "expected_after": "application modal opens",
        },
        0,
    )

    assert steps[0]["state_key"].startswith("ocr#")
    assert steps[0]["screen_signature"]["phash"] == "0" * 16
    assert steps[0]["target"] == {
        "text": "지원하기",
        "region": "top-left",
        "ordinal": 0,
        "evidence_texts": ["공유하기"],
        "bbox_ratio": [0.1, 0.1, 0.6, 0.3],
        "center_ratio": [0.35, 0.2],
    }
    assert steps[0]["intent"] == "open apply flow"
    assert steps[0]["target_role"] == "apply_button"
    assert steps[0]["component"] == "job_detail_header"
    assert steps[0]["expected_after"] == "application modal opens"
    assert "bbox" not in steps[0]["target"]


def test_phash_replay_matches_current_marker_by_ratio():
    from agent.recipe.phash_replay import match_step_by_screen_signature

    current_signature = {
        "phash": "f0f0f0f0f0f0f0f0",
        "size": [1000, 1000],
        "anchors": ["검색", "채용"],
    }
    step = {
        "screen_signature": {
            "phash": "f0f0f0f0f0f0f0f0",
            "size": [1000, 1000],
            "anchors": ["검색", "채용"],
        },
        "target": {
            "text": "검색",
            "bbox_ratio": [0.79, 0.08, 0.83, 0.12],
            "center_ratio": [0.81, 0.10],
        },
    }
    markers = [
        {"id": 99, "bbox": [790, 80, 830, 120], "text": "검색"},
        {"id": 3, "bbox": [100, 100, 140, 140], "text": "검색"},
    ]

    marker_id, result = match_step_by_screen_signature(step, current_signature, markers)

    assert result["matched"] is True
    assert marker_id == 99


def test_phash_replay_rejects_different_screen_before_text_match():
    from agent.recipe.phash_replay import match_step_by_screen_signature

    current_signature = {
        "phash": "0000000000000000",
        "size": [1000, 1000],
        "anchors": ["검색", "채용"],
    }
    step = {
        "screen_signature": {
            "phash": "ffffffffffffffff",
            "size": [1000, 1000],
            "anchors": ["검색", "채용"],
        },
        "target": {"text": "검색", "center_ratio": [0.81, 0.10]},
    }
    markers = [{"id": 99, "bbox": [790, 80, 830, 120], "text": "검색"}]

    marker_id, result = match_step_by_screen_signature(step, current_signature, markers)

    assert marker_id is None
    assert result["reason"] == "phash_distance"


def test_state_key_ignores_dynamic_numeric_changes():
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
    url = "https://www.wanted.co.kr/wd/12345"
    assert compute_state_key(url, before) == compute_state_key(url, after_count_change)
    assert compute_state_key(url, before) == compute_state_key("https://example.com/other", before)


def test_anchor_similarity_tolerates_partial_screen_changes():
    from agent.recipe.state_key import anchor_similarity

    saved = ["채용", "검색", "포지션 12", "회사 40", "Android 개발자"]
    current = [
        {"text": "채용"},
        {"text": "검색"},
        {"text": "포지션 13"},
        {"text": "회사 41"},
        {"text": "Android 개발자"},
        {"text": "새 광고 문구"},
    ]

    assert anchor_similarity(saved, current) >= 0.4


def test_transition_contract_waits_for_known_result_outcomes():
    from agent.recipe.transition import evaluate_transition

    contract = {
        "common_ready_cues": [
            {"kind": "slot_text", "slot": "query"},
            {"kind": "text_any", "values": ["포지션", "회사"]},
        ],
        "outcomes": [
            {"name": "results_found", "cues": [{"kind": "text_all", "values": ["Android 개발자", "마크노바"]}]},
            {"name": "results_empty", "cues": [{"kind": "text_any", "values": ["검색 결과 없음", "0건"]}]},
        ],
        "loading_cues": [{"kind": "text_any", "values": ["포지션(0)"]}],
        "timeout_sec": 5,
    }
    skeleton = [
        {"text": "android 개발자"},
        {"text": "포지션(0)"},
        {"text": "회사(0)"},
    ]
    found = skeleton + [{"text": "Android 개발자"}, {"text": "마크노바"}]
    empty = skeleton + [{"text": "검색 결과 없음"}]

    assert evaluate_transition(contract, skeleton, {"query": "android 개발자"}, 1)["status"] == "pending"
    assert evaluate_transition(contract, found, {"query": "android 개발자"}, 2)["outcome"] == "results_found"
    assert evaluate_transition(contract, empty, {"query": "android 개발자"}, 2)["outcome"] == "results_empty"
    assert evaluate_transition(contract, skeleton, {"query": "android 개발자"}, 6)["status"] == "unknown"
    assert evaluate_transition({}, skeleton, {"query": "android 개발자"}, 0)["reason"] == "transition_contract_missing"
    assert evaluate_transition({"timeout_sec": 12}, skeleton, {"query": "android 개발자"}, 0)["reason"] == "transition_contract_empty"


def test_perception_node_records_and_resolves_pending_transition(monkeypatch, tmp_path):
    import time
    from PIL import Image
    from agent.graph import nodes

    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        def capture_screen(self):
            return screenshot

        def analyze_ui(self, _path):
            return {
                "markers": [
                    {"id": 1, "bbox": [10, 150, 200, 180], "text": "android 개발자"},
                    {"id": 2, "bbox": [10, 200, 200, 230], "text": "포지션"},
                    {"id": 3, "bbox": [10, 250, 300, 280], "text": "Android App 개발자"},
                ],
                "marked_image": str(screenshot),
            }

    monkeypatch.setattr(nodes, "_get_perception", lambda: FakePerception())
    result = nodes.perception_node(
        {
            "current_url": "https://www.wanted.co.kr",
            "current_url_stale": False,
            "pending_transition": {
                "action_seq": 3,
                "action": "press_key",
                "expected_after": "검색 결과가 나타남",
                "source": "reflex",
                "started_at": time.time(),
                "attempts": 0,
                "params": {"query": "android 개발자"},
                "contract": {
                    "common_ready_cues": [
                        {"kind": "slot_text", "slot": "query"},
                        {"kind": "text_any", "values": ["포지션"]},
                    ],
                    "outcomes": [
                        {"name": "results_found", "cues": [{"kind": "text_any", "values": ["Android App 개발자"]}]}
                    ],
                },
            },
        }
    )

    assert result["transition_status"] == "ready"
    assert result["transition_outcome"] == "results_found"
    assert result["pending_transition"] == {}
    assert result["transition_observations"][0]["action_seq"] == 3
    assert "Android App 개발자" in result["transition_observations"][0]["marker_texts"]


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
                "transition_contract": {
                    "common_ready_cues": [{"kind": "text_any", "values": ["검색 결과"]}],
                    "outcomes": [],
                },
            }
        ],
    )

    assert saved == 1
    recipe = store.get_recipe("state-a")
    assert recipe is not None
    assert recipe.steps[0].action == "click_marker"
    assert recipe.steps[0].transition_contract.common_ready_cues[0].values == ["검색 결과"]

    conn = sqlite3.connect(db_path)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(recipes)").fetchall()]
    conn.close()
    assert "state_key" in columns
    assert "metadata_json" in columns
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
    assert recipe.steps[0].transition_contract is None
    assert recipe.steps[1].transition_contract is None


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
                        replay_mode="parameterized",
                        target={"text": "검색", "region": "top-left", "ordinal": 0},
                        param={"text": "ai 엔지니어"},
                    ),
                    RecipeStep(
                        seq=1,
                        state_key="state-a",
                        action="press_key",
                        replay_mode="fixed",
                        param={"key": "enter"},
                        transition_contract={
                            "common_ready_cues": [{"kind": "text_any", "values": ["포지션"]}],
                            "outcomes": [{"name": "results_found", "cues": [{"kind": "text_any", "values": ["회사명"]}]}],
                        },
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
    assert msg.tool_calls[0]["name"] == "type_in_marker"
    assert msg.tool_calls[0]["args"] == {"marker_id": 7, "text": "ai 엔지니어"}
    assert msg.tool_calls[1]["name"] == "press_key"
    assert msg.tool_calls[1]["args"] == {"key": "enter"}
    assert result["reflex_transition_contracts"][msg.tool_calls[1]["id"]]["outcomes"][0]["name"] == "results_found"


def test_reflex_node_uses_phash_ratio_when_step_has_screen_signature(monkeypatch):
    from agent.graph import nodes
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    class FakeStore:
        def get_recipe(self, state_key):
            return SiteRecipe(
                site="wanted.co.kr",
                goal="goal",
                steps=[
                    RecipeStep(
                        seq=0,
                        state_key="state-a",
                        action="click_marker",
                        replay_mode="fixed",
                        screen_signature={
                            "phash": "f0f0f0f0f0f0f0f0",
                            "size": [1000, 1000],
                            "anchors": ["검색"],
                        },
                        target={
                            "text": "검색",
                            "bbox_ratio": [0.79, 0.08, 0.83, 0.12],
                            "center_ratio": [0.81, 0.10],
                        },
                    )
                ],
            )

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = nodes.reflex_node(
        {
            "goal": "ai 엔지니어 공고 찾아줘",
            "current_url": "https://www.wanted.co.kr",
            "reflex_state_key": "state-a",
            "screen_signature": {
                "phash": "f0f0f0f0f0f0f0f0",
                "size": [1000, 1000],
                "anchors": ["검색"],
            },
            "current_markers": [{"id": 77, "bbox": [790, 80, 830, 120], "text": "검색"}],
        }
    )

    assert result["reflex_hit"] is True
    assert result["last_action_result"].tool_calls[0]["args"] == {"marker_id": 77}


def test_reflex_node_rejects_signed_step_when_phash_misses(monkeypatch):
    from agent.graph import nodes
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    class FakeStore:
        def get_recipe(self, state_key):
            return SiteRecipe(
                site="wanted.co.kr",
                goal="goal",
                steps=[
                    RecipeStep(
                        seq=0,
                        state_key="state-a",
                        action="click_marker",
                        replay_mode="fixed",
                        screen_signature={
                            "phash": "ffffffffffffffff",
                            "size": [1000, 1000],
                            "anchors": ["검색"],
                        },
                        target={"text": "검색", "center_ratio": [0.81, 0.10]},
                    )
                ],
            )

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = nodes.reflex_node(
        {
            "goal": "ai 엔지니어 공고 찾아줘",
            "current_url": "https://www.wanted.co.kr",
            "reflex_state_key": "state-a",
            "screen_signature": {
                "phash": "0000000000000000",
                "size": [1000, 1000],
                "anchors": ["검색"],
            },
            "current_markers": [{"id": 77, "bbox": [790, 80, 830, 120], "text": "검색"}],
        }
    )

    assert result["reflex_hit"] is False


def test_reflex_node_replaces_type_input_slot(monkeypatch):
    from agent.graph import nodes
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe
    from shared.schema.skill_schema import RecipeSkillMetadata, SkillInputSlot

    class FakeStore:
        def get_recipe(self, state_key):
            return SiteRecipe(
                site="wanted",
                goal="old goal",
                skill_metadata=RecipeSkillMetadata(
                    inputs=[SkillInputSlot(name="query", required=True)]
                ),
                steps=[
                    RecipeStep(
                        seq=0,
                        state_key="state-a",
                        action="type_in_marker",
                        replay_mode="parameterized",
                        target={"text": "Search", "region": "top-left", "ordinal": 0},
                        param={"text": "old query", "slot_name": "query"},
                        slot_refs=["query"],
                    )
                ],
            )

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = nodes.reflex_node(
        {
            "goal": "find android jobs",
            "current_url": "https://www.wanted.co.kr",
            "reflex_state_key": "state-a",
            "current_markers": [{"id": 7, "bbox": [10, 10, 70, 40], "text": "Search"}],
            "recipe_params": {"query": "android developer"},
        }
    )

    assert result["reflex_hit"] is True
    assert result["last_action_result"].tool_calls[0]["args"] == {
        "marker_id": 7,
        "text": "android developer",
        "slot_name": "query",
    }


def test_reflex_node_uses_ocr_similarity_when_exact_state_misses(monkeypatch):
    from agent.graph import nodes
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    class FakeStore:
        def get_recipe(self, state_key):
            assert state_key == "new-state"
            return None

        def get_similar_recipes(self, site, markers, min_similarity=0.0):
            assert site == "wanted"
            assert min_similarity == 0.25
            return [
                (
                    "recorded-state",
                    SiteRecipe(
                        site="wanted",
                        goal="collect jobs",
                        steps=[
                            RecipeStep(
                                seq=0,
                                state_key="recorded-state",
                                state_anchors=["검색", "채용"],
                                action="click_marker",
                                replay_mode="fixed",
                                target={"text": "검색"},
                            )
                        ],
                    ),
                    0.62,
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = nodes.reflex_node(
        {
            "goal": "android 개발자 공고 찾아줘",
            "reflex_state_key": "new-state",
            "current_markers": [{"id": 7, "bbox": [10, 10, 70, 40], "text": "검색"}],
            "recipe_params": {"site": "wanted", "query": "android 개발자"},
        }
    )

    assert result["reflex_hit"] is True
    assert result["reflex_state_key"] == "new-state"
    assert result["last_action_result"].tool_calls[0]["args"] == {"marker_id": 7}


def test_reflex_node_rejects_similar_recipe_when_target_does_not_match(monkeypatch):
    from agent.graph import nodes
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    class FakeStore:
        def get_recipe(self, state_key):
            return None

        def get_similar_recipes(self, site, markers, min_similarity=0.0):
            return [
                (
                    "recorded-state",
                    SiteRecipe(
                        site="wanted",
                        steps=[
                            RecipeStep(
                                seq=0,
                                state_key="recorded-state",
                                state_anchors=["검색", "채용"],
                                action="click_marker",
                                replay_mode="fixed",
                                target={"text": "상세 정보 더 보기"},
                            )
                        ],
                    ),
                    0.7,
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = nodes.reflex_node(
        {
            "goal": "android 개발자 공고 찾아줘",
            "reflex_state_key": "new-state",
            "current_markers": [{"id": 7, "bbox": [10, 10, 70, 40], "text": "검색"}],
            "recipe_params": {"site": "wanted", "query": "android 개발자"},
        }
    )

    assert result["reflex_hit"] is False


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
            "transition_contract": None,
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
    assert route_after_perception({}) == "reflex"
    assert route_after_perception({"transition_status": "pending"}) == "perception"
    assert route_after_perception({"transition_status": "unknown", "transition_source": "reflex"}) == "reasoning"
    assert route_after_perception({"transition_status": "unknown", "transition_source": "page_policy"}) == "reasoning"
    assert route_after_perception({"transition_status": "unknown", "transition_source": "autonomous"}) == "reflex"
    assert route_after_perception({"transition_status": "ready"}) == "reflex"

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


def test_match_marker_filters_generic_components_by_region_before_evidence():
    from agent.recipe.matcher import match_marker

    markers = [
        {"id": 1, "bbox": [900, 100, 960, 160], "text": "상호작용 가능한 요소 (icon)"},
        {"id": 2, "bbox": [100, 500, 300, 560], "text": "상호작용 가능한 요소 (icon)"},
        {"id": 3, "bbox": [700, 180, 820, 220], "text": "기업"},
        {"id": 4, "bbox": [760, 230, 900, 270], "text": "서비스"},
        {"id": 5, "bbox": [120, 570, 260, 610], "text": "기업"},
        {"id": 6, "bbox": [180, 620, 320, 660], "text": "서비스"},
    ]
    step = {
        "target": {
            "text": "상호작용 가능한 요소 (icon)",
            "region": "top-right",
            "evidence_texts": ["기업", "서비스"],
        }
    }

    assert match_marker(step, markers) == 1


def test_marker_ordinal_ignores_browser_chrome_and_uses_region():
    from agent.recipe.matcher import marker_ordinal

    markers = [
        {"id": 1, "bbox": [900, 140, 960, 170], "text": "상호작용 가능한 요소 (icon)"},
        {"id": 2, "bbox": [1200, 140, 1260, 170], "text": "상호작용 가능한 요소 (icon)"},
        {"id": 3, "bbox": [1500, 190, 1560, 250], "text": "상호작용 가능한 요소 (icon)"},
        {"id": 4, "bbox": [1600, 190, 1660, 250], "text": "상호작용 가능한 요소 (icon)"},
        {"id": 5, "bbox": [100, 500, 160, 560], "text": "상호작용 가능한 요소 (icon)"},
    ]

    assert marker_ordinal(markers[2], markers) == 0
    assert marker_ordinal(markers[3], markers) == 1


def test_match_marker_combines_adjacent_split_ocr_text():
    from agent.recipe.matcher import match_marker

    markers = [
        {"id": 1, "bbox": [100, 100, 160, 140], "text": "상세"},
        {"id": 2, "bbox": [165, 100, 230, 140], "text": "정보"},
        {"id": 3, "bbox": [100, 300, 220, 340], "text": "지원하기"},
    ]
    step = {
        "target": {
            "text": "상세정보",
            "semantic_label": "상세 정보 더 보기",
            "region": "top-left",
        }
    }

    assert match_marker(step, markers) == 1


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


def test_feedback_episode_records_parameter_candidate_and_observation():
    from langchain_core.messages import AIMessage
    from agent.recipe.feedback import record_action_episode

    episodes = []
    state = {
        "goal": "AI 엔지니어 채용공고 찾아줘",
        "current_url": "https://www.wanted.co.kr",
        "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "검색"}],
    }
    enriched = {
        "action": "type_in_marker",
        "status": "success",
        "result": "ok",
        "target": {"marker_id": 1, "text": "검색"},
    }

    record_action_episode(
        episodes,
        state,
        AIMessage(content="검색어를 입력한다"),
        "type_in_marker",
        {
            "marker_id": 1,
            "text": "AI 엔지니어",
            "reason": "enter search keyword",
            "target_role": "search_input",
            "target_component": "site_search",
            "expected_after": "search keyword is entered",
        },
        enriched,
        {"state_key": "state-home", "url": "https://www.wanted.co.kr", "screenshot": "s.png", "marked_image": "m.png"},
        {"current_url": "https://www.wanted.co.kr", "current_url_stale": True, "screen_changed": True, "extracted_jd": {}, "is_finished": False},
        0,
    )

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["proposal"]["action"] == "type_in_marker"
    assert episode["proposal"]["llm_thought"] == "검색어를 입력한다"
    assert episode["proposal"]["expected_after"] == "search keyword is entered"
    assert episode["proposal"]["parameter_candidates"][0]["slot_candidate"] == "query"
    assert episode["observation"]["before"]["state_key"] == "state-home"
    assert episode["observation"]["after"]["screen_changed"] is True
    assert episode["feedback"]["label"] == "partial"


def test_feedback_episode_does_not_infer_site_slot_from_open_url():
    from langchain_core.messages import AIMessage
    from agent.recipe.feedback import record_action_episode

    episodes = []
    state = {
        "goal": "collect jobs",
        "current_url": "",
        "current_markers": [],
    }

    record_action_episode(
        episodes,
        state,
        AIMessage(content="open the site home page"),
        "open_browser",
        {
            "url": "https://www.wanted.co.kr",
            "reason": "start from home page",
            "expected_after": "site home page is visible",
        },
        {"action": "open_browser", "status": "success", "result": {"opened": True}},
        {"state_key": "state-empty", "url": "", "screenshot": "s.png", "marked_image": "m.png"},
        {"current_url": "https://www.wanted.co.kr", "current_url_stale": True, "screen_changed": True, "extracted_jd": {}, "is_finished": False},
        0,
    )

    assert episodes[0]["proposal"]["parameter_candidates"] == []
    assert episodes[0]["proposal"]["expected_after"] == "site home page is visible"


def _sample_feedback_episode(seq=0):
    return {
        "seq": seq,
        "goal": "AI 엔지니어 채용공고 찾아줘",
        "site": "wanted.co.kr",
        "page_state_key": "state-home",
        "proposal": {
            "action": "type_in_marker",
            "args": {"marker_id": 1, "text": "AI 엔지니어"},
            "expected_after": "search results are visible",
            "parameter_candidates": [{"slot_candidate": "query", "value": "AI 엔지니어", "confidence": 0.45}],
        },
        "observation": {
            "before": {"state_key": "state-home", "url": "https://www.wanted.co.kr"},
            "after": {"screen_changed": True},
            "result": {"status": "success", "action": "type_in_marker"},
        },
        "feedback": {"label": "partial", "reason": "screen-changing action executed", "confidence": 0.45},
    }


def test_feedback_store_commits_and_reads_recent(tmp_path):
    from agent.recipe.feedback_store import FeedbackStore

    store = FeedbackStore(tmp_path / "feedback.db")
    saved = store.commit_episodes([_sample_feedback_episode()], run_id="run-1", run_status="finished", source="test")

    assert saved == 1
    rows = store.list_recent(limit=5)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["run_status"] == "finished"
    assert rows[0]["site"] == "wanted.co.kr"
    assert rows[0]["action"] == "type_in_marker"
    assert rows[0]["feedback_label"] == "partial"
    assert rows[0]["payload"]["proposal"]["parameter_candidates"][0]["slot_candidate"] == "query"


def test_database_initializes_feedback_episode_table(tmp_path):
    from shared.db.database import Database

    db_path = tmp_path / "jobs.db"
    Database(db_path)

    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    columns = [row[1] for row in conn.execute("PRAGMA table_info(feedback_episodes)").fetchall()]
    submission_columns = [row[1] for row in conn.execute("PRAGMA table_info(worker_submissions)").fetchall()]
    candidate_columns = [row[1] for row in conn.execute("PRAGMA table_info(recipe_candidates)").fetchall()]
    recipe_columns = [row[1] for row in conn.execute("PRAGMA table_info(recipes)").fetchall()]
    conn.close()

    assert "feedback_episodes" in tables
    assert "worker_submissions" in tables
    assert "recipe_candidates" in tables
    assert "episode_id" in columns
    assert "feedback_label" in columns
    assert "submission_id" in submission_columns
    assert "review_decision" in submission_columns
    assert "candidate_id" in candidate_columns
    assert "steps_json" in candidate_columns
    assert "validation_json" in candidate_columns
    assert "metadata_json" in recipe_columns


def test_realtime_scraping_commits_feedback_episodes_with_run_status(monkeypatch):
    from agent.tools.realtime_scraping import _commit_feedback_episodes

    seen = {}

    class FakeStore:
        def commit_episodes(self, episodes, run_id=None, run_status="", source=""):
            seen["episodes"] = episodes
            seen["run_id"] = run_id
            seen["run_status"] = run_status
            seen["source"] = source
            return len(episodes)

    monkeypatch.setattr("agent.recipe.feedback_store.FeedbackStore", lambda: FakeStore())

    saved = _commit_feedback_episodes({"feedback_episodes": [_sample_feedback_episode()]}, True, False, run_id="worker-run-1")

    assert saved == 1
    assert seen["run_id"] == "worker-run-1"
    assert seen["run_status"] == "recursion_limit"
    assert seen["source"] == "realtime_scraping"


def test_worker_submission_shape_review_requests_revision():
    from agent.recipe.reviewer import build_worker_submission, review_worker_submission

    submission = build_worker_submission(
        {
            "goal": "collect AI engineer jobs",
            "current_url": "https://www.wanted.co.kr/search?query=ai",
            "extracted_jd": {},
            "recorded_steps": [],
            "feedback_episodes": [],
        },
        site="wanted",
        keyword="ai engineer",
        run_status="stopped",
    )

    review = review_worker_submission(submission)

    assert review["decision"] == "revise"
    assert review["recipe_candidate"] is False
    assert "extracted_summary" in review["feedback_to_worker"]


def test_worker_submission_review_accepts_structured_data(monkeypatch):
    from agent.recipe.reviewer import build_worker_submission, review_worker_submission

    monkeypatch.setenv("VISION_WORKER_SUMMARY_MODE", "off")
    monkeypatch.setenv("VISION_WORKER_REVIEW_MODE", "shape")
    submission = build_worker_submission(
        {
            "goal": "collect AI engineer jobs",
            "current_url": "https://www.wanted.co.kr/wd/1",
            "extracted_jd": {
                "jobs": [
                    {
                        "company_name": "Acme",
                        "position": "AI Engineer",
                        "url": "https://www.wanted.co.kr/wd/1",
                    }
                ]
            },
            "recorded_steps": [
                {
                    "seq": 0,
                    "state_key": "state-a",
                    "action": "click_marker",
                    "target": {"text": "AI Engineer"},
                    "intent": "open the selected job card",
                    "target_role": "job_card_title",
                    "component": "search_result_card",
                    "expected_after": "job detail page is visible",
                }
            ],
            "feedback_episodes": [_sample_feedback_episode()],
            "transition_observations": [
                {"action_seq": 0, "status": "unknown", "marker_texts": ["AI Engineer", "주요업무"]}
            ],
        },
        site="wanted",
        keyword="ai engineer",
        run_status="finished",
    )

    review = review_worker_submission(submission)

    assert review["decision"] == "accept"
    assert review["recipe_candidate"] is True
    assert submission["skill_metadata_evidence"]["site"] == "wanted"
    assert submission["skill_metadata_evidence"]["actions"] == ["click_marker"]
    assert submission["skill_metadata_evidence"]["step_intents"][0]["expected_after"] == "job detail page is visible"
    assert submission["transition_observations"][0]["action_seq"] == 0


def test_worker_submission_report_summary_uses_llm(monkeypatch):
    import agent.recipe.reviewer as reviewer
    from agent.recipe.reviewer import build_worker_submission

    class FakeStructuredLLM:
        def invoke(self, messages):
            assert "raw_job" in messages[-1].content
            return reviewer.ReportJobSummary(
                jobs=[
                    reviewer.ReportJobSummaryItem(
                        company="비모소프트",
                        position="[인턴] iOS 개발자",
                        url="https://www.wanted.co.kr/wd/355442",
                        field_count=4,
                    )
                ]
            )

    class FakeLLM:
        def __init__(self, model="", temperature=0.0):
            self.model = model
            self.temperature = temperature

        def with_structured_output(self, schema):
            assert schema is reviewer.ReportJobSummary
            return FakeStructuredLLM()

    monkeypatch.setenv("VISION_WORKER_SUMMARY_MODE", "llm")
    monkeypatch.setenv("VISION_WORKER_SUMMARY_MODEL", "fake-summary-model")
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", FakeLLM)

    submission = build_worker_submission(
        {
            "goal": "collect iOS jobs",
            "current_url": "https://www.wanted.co.kr/wd/355442",
            "extracted_jd": {
                "공고목록": [
                    {
                        "회사명": "비모소프트",
                        "직무명": "[인턴] iOS 개발자",
                        "URL": "https://www.wanted.co.kr/wd/355442",
                        "주요업무": "Swift 기반 iOS App 개발",
                    }
                ]
            },
        },
        site="wanted",
        keyword="iOS 개발자",
        run_status="finished",
    )

    summary = submission["extracted_summary"]
    assert summary["summary_source"] == "llm"
    assert summary["jobs"] == [
        {
            "company": "비모소프트",
            "position": "[인턴] iOS 개발자",
            "url": "https://www.wanted.co.kr/wd/355442",
            "field_count": 4,
        }
    ]


def test_submission_store_commits_and_reads_recent(tmp_path):
    from agent.recipe.submission_store import SubmissionStore

    submission = {
        "run_id": "worker-run-1",
        "goal": "collect jobs",
        "site": "wanted",
        "keyword": "ai engineer",
        "run_status": "finished",
        "review_attempt": 0,
        "collected_count": 1,
    }
    review = {"decision": "accept", "confidence": 0.7, "feedback_to_worker": ""}
    store = SubmissionStore(tmp_path / "submissions.db")

    submission_id = store.commit_submission(submission, review=review, source="test")
    rows = store.list_recent(limit=5)

    assert submission_id == "worker-run-1:0"
    assert len(rows) == 1
    assert rows[0]["review_decision"] == "accept"
    assert rows[0]["payload"]["keyword"] == "ai engineer"
    assert rows[0]["review"]["confidence"] == 0.7


def test_recipe_candidate_store_commits_reviewed_candidate(tmp_path):
    from agent.recipe.candidate_store import RecipeCandidateStore

    submission = {
        "run_id": "worker-run-1",
        "goal": "collect jobs",
        "site": "wanted",
        "keyword": "ai engineer",
        "review_attempt": 0,
        "recorded_steps": [
            {"seq": 0, "state_key": "state-a", "action": "click_marker", "target": {"text": "AI Engineer"}}
        ],
        "transition_observations": [
            {
                "action_seq": 0,
                "status": "unknown",
                "marker_texts": ["AI Engineer", "주요업무"],
            }
        ],
    }
    review = {"decision": "accept", "recipe_candidate": True, "confidence": 0.7}
    store = RecipeCandidateStore(tmp_path / "candidates.db")

    candidate_id = store.commit_candidate(submission, review=review, source="test", submission_id="worker-run-1:0")
    rows = store.list_recent(limit=5)

    assert candidate_id == "worker-run-1:0"
    assert len(rows) == 1
    assert rows[0]["status"] == "pending_replay"
    assert rows[0]["site"] == "wanted"
    assert rows[0]["steps"][0]["state_key"] == "state-a"
    assert rows[0]["payload"]["keyword"] == "ai engineer"
    assert rows[0]["review"]["recipe_candidate"] is True


def test_recipe_candidate_store_skips_non_candidates(tmp_path):
    from agent.recipe.candidate_store import RecipeCandidateStore

    store = RecipeCandidateStore(tmp_path / "candidates.db")
    candidate_id = store.commit_candidate(
        {"run_id": "worker-run-1", "review_attempt": 0, "recorded_steps": [{"state_key": "state-a"}]},
        review={"decision": "accept", "recipe_candidate": False},
        source="test",
    )

    assert candidate_id == ""
    assert store.list_recent(limit=5) == []

def _sample_recipe_candidate_submission():
    return {
        "run_id": "worker-run-critic",
        "goal": "collect jobs",
        "site": "wanted",
        "keyword": "ai engineer",
        "review_attempt": 0,
        "recorded_steps": [
            {"seq": 0, "state_key": "state-a", "action": "click_marker", "target": {"text": "AI Engineer"}}
        ],
        "transition_observations": [
            {
                "action_seq": 0,
                "status": "unknown",
                "marker_texts": ["AI Engineer", "주요업무"],
            }
        ],
        "feedback_episodes": [
            {
                "seq": 0,
                "observation": {
                    "before": {
                        "marker_texts": ["채용", "검색", "AI Engineer"],
                    }
                },
            }
        ],
    }


def test_recipe_candidate_status_update_records_llm_validation(tmp_path):
    from agent.recipe.candidate_store import RecipeCandidateStore

    store = RecipeCandidateStore(tmp_path / "candidates.db")
    candidate_id = store.commit_candidate(
        _sample_recipe_candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    assert store.update_status(
        candidate_id,
        "revise",
        validation={"review": {"decision": "revise", "reasons": ["needs clearer evidence"]}},
    ) is True
    row = store.get_candidate(candidate_id)

    assert row["status"] == "revise"
    assert row["validation"]["review"]["decision"] == "revise"


def test_candidate_reviewer_promotes_only_when_llm_accepts(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        _sample_recipe_candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    seen = {}

    def critic(payload):
        seen["payload"] = payload
        return {
            "decision": "accept",
            "reasons": ["critic chose to promote"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "when_to_use": "Use on a job search result page.",
                "goal_pattern": "collect jobs",
                "site": "wanted",
                "inputs": [{"name": "query", "required": True, "observed_value": "AI"}],
                "step_intents": [
                    {
                        "seq": 0,
                        "action": "click_marker",
                        "intent": "Open the stable search control.",
                        "expected_after": "Search overlay is visible.",
                        "replay_mode": "fixed",
                    }
                ],
                "verification": {"success_signals": ["job data collected"]},
            },
            "transition_contracts": [
                {
                    "seq": 0,
                    "contract": {
                        "common_ready_cues": [{"kind": "text_any", "values": ["채용 상세"]}],
                        "outcomes": [{"name": "detail_opened", "cues": [{"kind": "text_any", "values": ["주요업무"]}]}],
                    },
                }
            ],
            "confidence": 0.82,
        }

    review = review_and_apply_candidate(candidate_id, db_path=tmp_path / "critic.db", critic=critic)
    candidate = store.get_candidate(candidate_id)
    recipes = RecipeStore(tmp_path / "critic.db").get_by_site("wanted")

    assert seen["payload"]["candidate_id"] == candidate_id
    assert seen["payload"]["steps"][0]["state_key"] == "state-a"
    assert seen["payload"]["transition_observations"][0]["action_seq"] == 0
    assert review["decision"] == "accept"
    assert review["promoted_count"] == 1
    assert candidate["status"] == "accepted"
    assert candidate["validation"]["promoted_count"] == 1
    assert recipes[0]["steps"][0]["state_key"] == "state-a"
    assert recipes[0]["steps"][0]["replay_mode"] == "fixed"
    assert recipes[0]["steps"][0]["intent"] == "Open the stable search control."
    assert recipes[0]["steps"][0]["expected_after"] == "Search overlay is visible."
    assert recipes[0]["steps"][0]["state_anchors"] == ["aiengineer", "검색", "채용"]
    assert recipes[0]["steps"][0]["transition_contract"]["outcomes"][0]["name"] == "detail_opened"
    assert recipes[0]["skill_metadata"]["when_to_use"] == "Use on a job search result page."


def test_candidate_reviewer_excludes_reasoning_steps_from_active_recipe(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _sample_recipe_candidate_submission()
    submission["recorded_steps"].append(
        {
            "seq": 1,
            "state_key": "state-results",
            "action": "click_marker",
            "target": {"text": "Specific Job Title"},
            "component": "job_card_title",
        }
    )
    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["stable control only"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "step_intents": [
                    {"seq": 0, "action": "click_marker", "replay_mode": "fixed"},
                    {"seq": 1, "action": "click_marker", "replay_mode": "reasoning"},
                ],
            },
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(tmp_path / "critic.db").get_by_site("wanted")
    assert review["promoted_count"] == 1
    assert [step["state_key"] for recipe in recipes for step in recipe["steps"]] == ["state-a"]


def test_candidate_reviewer_revise_does_not_promote(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        _sample_recipe_candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        critic=lambda payload: {
            "decision": "revise",
            "reasons": ["critic requested another worker pass"],
            "feedback_to_worker": "collect clearer action rationale",
            "promote_to_active_recipe": False,
            "confidence": 0.6,
        },
    )
    candidate = store.get_candidate(candidate_id)

    assert review["decision"] == "revise"
    assert review["promoted_count"] == 0
    assert candidate["status"] == "revise"
    assert RecipeStore(tmp_path / "critic.db").get_by_site("wanted") == []


def test_candidate_reviewer_invalid_llm_shape_falls_back_to_revise(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore

    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        _sample_recipe_candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-critic:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        critic=lambda payload: {"not_the_schema": True},
    )
    candidate = store.get_candidate(candidate_id)

    assert review["decision"] == "revise"
    assert review["promote_to_active_recipe"] is False
    assert candidate["status"] == "revise"
    assert "critic_review_failed" in candidate["validation"]["review"]["reasons"][0]

def _sample_worker_result_for_learning_mode():
    return {
        "submission": _sample_recipe_candidate_submission(),
        "extracted_jd": {
            "jobs": [
                {"company_name": "Acme", "position": "AI Engineer", "url": "https://example.com/jobs/1"}
            ]
        },
        "keyword": "ai engineer",
    }


def test_realtime_recipe_learning_mode_off_skips_candidate(monkeypatch):
    from agent.tools import realtime_scraping as rs

    monkeypatch.setenv("VISION_RECIPE_LEARNING_MODE", "off")
    monkeypatch.setattr(rs, "_persist_collected_data", lambda extracted, keyword: 1)
    called = []
    monkeypatch.setattr(rs, "_commit_recipe_candidate", lambda *args, **kwargs: called.append(args) or ("candidate-1", {}))
    monkeypatch.setattr("agent.recipe.submission_store.SubmissionStore.commit_submission", lambda self, submission, review=None, source="": "worker-run-critic:0")

    persisted_count, submission, _review, _submission_id = rs.persist_accepted_worker_result(
        _sample_worker_result_for_learning_mode(),
        {"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
    )

    assert persisted_count == 1
    assert called == []
    assert "recipe_candidate_id" not in submission


def test_realtime_recipe_learning_mode_record_saves_candidate_without_critic(monkeypatch):
    from agent.tools import realtime_scraping as rs

    monkeypatch.setenv("VISION_RECIPE_LEARNING_MODE", "record")
    monkeypatch.setattr(rs, "_persist_collected_data", lambda extracted, keyword: 1)
    seen = {}
    def fake_commit_recipe_candidate(submission, review, source, submission_id, mode):
        seen["mode"] = mode
        return "candidate-1", {}

    monkeypatch.setattr(rs, "_commit_recipe_candidate", fake_commit_recipe_candidate)
    monkeypatch.setattr("agent.recipe.submission_store.SubmissionStore.commit_submission", lambda self, submission, review=None, source="": "worker-run-critic:0")

    _count, submission, _review, _submission_id = rs.persist_accepted_worker_result(
        _sample_worker_result_for_learning_mode(),
        {"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
    )

    assert seen["mode"] == "record"
    assert submission["recipe_candidate_id"] == "candidate-1"
    assert submission["recipe_learning_mode"] == "record"
    assert "recipe_candidate_review" not in submission


def test_realtime_recipe_learning_mode_promote_attaches_critic_review(monkeypatch):
    from agent.tools import realtime_scraping as rs

    monkeypatch.setenv("VISION_RECIPE_LEARNING_MODE", "promote")
    monkeypatch.setattr(rs, "_persist_collected_data", lambda extracted, keyword: 1)
    monkeypatch.setattr(
        rs,
        "_commit_recipe_candidate",
        lambda submission, review, source, submission_id, mode: (
            "candidate-1",
            {"decision": "accept", "promote_to_active_recipe": True, "promoted_count": 1},
        ),
    )
    monkeypatch.setattr("agent.recipe.submission_store.SubmissionStore.commit_submission", lambda self, submission, review=None, source="": "worker-run-critic:0")

    _count, submission, _review, _submission_id = rs.persist_accepted_worker_result(
        _sample_worker_result_for_learning_mode(),
        {"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
    )

    assert submission["recipe_learning_mode"] == "promote"
    assert submission["recipe_candidate_review"]["promoted_count"] == 1


def test_process_recipe_candidates_review_mode_does_not_promote(tmp_path):
    from agent.recipe.candidate_reviewer import process_recipe_candidates
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    db_path = tmp_path / "batch.db"
    store = RecipeCandidateStore(db_path)
    candidate_ids = []
    for idx in range(2):
        submission = _sample_recipe_candidate_submission()
        submission["run_id"] = f"worker-run-batch-{idx}"
        submission["recorded_steps"][0]["state_key"] = f"state-{idx}"
        candidate_ids.append(
            store.commit_candidate(
                submission,
                review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
                source="test",
                submission_id=f"worker-run-batch-{idx}:0",
            )
        )

    seen = []

    def critic(payload):
        seen.append(payload["candidate_id"])
        return {
            "decision": "accept",
            "reasons": ["critic accepted replay evidence"],
            "feedback_to_worker": "",
            "promote_to_active_recipe": True,
            "confidence": 0.9,
        }

    result = process_recipe_candidates(limit=10, mode="review", db_path=db_path, critic=critic)

    assert result["mode"] == "review"
    assert result["processed_count"] == 2
    assert result["promoted_count"] == 0
    assert set(seen) == set(candidate_ids)
    assert RecipeStore(db_path).get_by_site("wanted") == []
    for candidate_id in candidate_ids:
        candidate = store.get_candidate(candidate_id)
        assert candidate["status"] == "accepted"
        assert candidate["validation"]["allow_promote"] is False
        assert candidate["validation"]["promoted_count"] == 0


def test_process_recipe_candidates_promote_mode_writes_active_recipe(tmp_path):
    from agent.recipe.candidate_reviewer import process_recipe_candidates
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    db_path = tmp_path / "batch.db"
    store = RecipeCandidateStore(db_path)
    candidate_id = store.commit_candidate(
        _sample_recipe_candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
        source="test",
        submission_id="worker-run-batch:0",
    )

    result = process_recipe_candidates(
        limit=5,
        mode="promote",
        db_path=db_path,
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["critic promoted replay evidence"],
                "feedback_to_worker": "",
                "promote_to_active_recipe": True,
                "skill_metadata": {
                    "step_intents": [
                        {"seq": 0, "action": "click_marker", "replay_mode": "fixed"}
                    ]
                },
                "confidence": 0.88,
            },
    )

    candidate = store.get_candidate(candidate_id)
    recipes = RecipeStore(db_path).get_by_site("wanted")

    assert result["mode"] == "promote"
    assert result["processed_count"] == 1
    assert result["promoted_count"] == 1
    assert candidate["status"] == "accepted"
    assert candidate["validation"]["allow_promote"] is True
    assert recipes[0]["steps"][0]["state_key"] == "state-a"


def test_review_recipe_candidates_tool_returns_batch_json(monkeypatch):
    import json

    import agent.recipe.candidate_reviewer as reviewer
    from agent.tools.recipe_learning import review_recipe_candidates

    seen = {}

    def fake_process_recipe_candidates(limit=5, mode="review", status="pending_replay"):
        seen.update({"limit": limit, "mode": mode, "status": status})
        return {"mode": mode, "requested_limit": limit, "status": status, "processed_count": 0}

    monkeypatch.setattr(reviewer, "process_recipe_candidates", fake_process_recipe_candidates)

    payload = json.loads(
        review_recipe_candidates.invoke({"mode": "promote", "limit": 2, "status": "accepted"})
    )

    assert seen == {"limit": 2, "mode": "promote", "status": "accepted"}
    assert payload["mode"] == "promote"
    assert payload["requested_limit"] == 2

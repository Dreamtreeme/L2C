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


def test_phash_replay_rejects_step_without_roi_signature():
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

    assert marker_id is None
    assert result["matched"] is False
    assert result["reason"] == "roi_signature_missing"
    assert result["mode"] == "roi_phash"


def test_roi_phash_replay_rejects_different_roi_before_text_match(tmp_path):
    from PIL import Image, ImageDraw
    from agent.recipe.phash_replay import match_step_by_screen_signature
    from agent.vision.screen_signature import compute_target_roi_signature

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([150, 20, 170, 40], fill="black")
    image.save(saved)
    Image.new("RGB", (200, 200), "white").save(current)

    roi_signature = compute_target_roi_signature(saved, [150, 20, 170, 40], [200, 200])
    current_signature = {
        "phash": "0000000000000000",
        "size": [200, 200],
        "anchors": ["검색", "채용"],
    }
    step = {
        "screen_signature": {
            "phash": "ffffffffffffffff",
            "size": [200, 200],
            "anchors": ["검색", "채용"],
        },
        "roi_signature": roi_signature,
        "target": {"text": "검색", "center_ratio": [0.8, 0.15]},
    }
    markers = [{"id": 99, "bbox": [150, 20, 170, 40], "text": "검색"}]

    marker_id, result = match_step_by_screen_signature(
        step,
        current_signature,
        markers,
        current_image_path=str(current),
    )

    assert marker_id is None
    assert result["reason"] == "roi_phash_distance"
    assert result["mode"] == "roi_phash"


def test_record_ui_step_stores_target_roi_signature(tmp_path):
    from PIL import Image, ImageDraw
    from agent.recipe.record import record_ui_step

    screenshot = tmp_path / "screen.png"
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([150, 20, 170, 40], fill="black")
    image.save(screenshot)

    steps = []
    state = {
        "goal": "검색",
        "current_url": "https://www.wanted.co.kr",
        "screen_signature": {
            "phash": "0" * 16,
            "size": [200, 200],
            "anchors": ["검색", "채용"],
        },
        "recent_images": [screenshot],
        "current_markers": [
            {"id": 1, "bbox": [150, 20, 170, 40], "text": "검색"},
            {"id": 2, "bbox": [20, 20, 60, 40], "text": "채용"},
        ],
    }

    record_ui_step(
        steps,
        state,
        "click_marker",
        {
            "marker_id": 1,
            "reason": "검색 아이콘 클릭",
            "target_role": "button",
            "target_component": "search_button",
        },
        0,
    )

    roi_signature = steps[0]["roi_signature"]
    crop_rect = roi_signature["crop_rect_ratio"]

    assert roi_signature["algorithm"] == "roi-phash-dct64-v1"
    assert len(roi_signature["phash"]) == 16
    assert crop_rect[0] <= steps[0]["target"]["bbox_ratio"][0]
    assert crop_rect[2] >= steps[0]["target"]["bbox_ratio"][2]
    assert roi_signature["target_center_ratio"] == [0.8, 0.15]


def test_marker_geometry_ratio_helpers_are_consistent():
    from agent.vision.marker_geometry import (
        bbox_from_ratio,
        bbox_to_ratio,
        center_ratio_from_bbox,
        marker_bbox,
        marker_center,
        marker_center_ratio,
        screen_size_from_signature,
    )

    marker = {"bbox": ["10", "20", "30", "60"]}
    size = [100, 200]

    assert marker_bbox(marker) == [10, 20, 30, 60]
    assert marker_center(marker) == (20, 40)
    assert bbox_to_ratio(marker_bbox(marker), size) == [0.1, 0.1, 0.3, 0.3]
    assert center_ratio_from_bbox(marker_bbox(marker), size) == [0.2, 0.2]
    assert marker_center_ratio(marker, size) == [0.2, 0.2]
    assert bbox_from_ratio([0.1, 0.1, 0.3, 0.3], size) == [10, 20, 30, 60]
    assert screen_size_from_signature({"size": ["100", "200"]}) == [100, 200]


def test_roi_phash_replay_matches_when_full_phash_differs(tmp_path):
    from PIL import Image, ImageDraw
    from agent.recipe.phash_replay import match_step_by_screen_signature
    from agent.vision.screen_signature import compute_target_roi_signature

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        if path == current:
            draw.rectangle([0, 120, 200, 200], fill="black")
        image.save(path)

    roi_signature = compute_target_roi_signature(saved, [150, 20, 170, 40], [200, 200])
    step = {
        "screen_signature": {
            "phash": "f" * 16,
            "size": [200, 200],
        },
        "roi_signature": roi_signature,
        "target": {
            "text": "검색",
            "bbox_ratio": [0.75, 0.1, 0.85, 0.2],
            "center_ratio": [0.8, 0.15],
        },
    }
    current_signature = {
        "phash": "0" * 16,
        "size": [200, 200],
    }
    markers = [{"id": 7, "bbox": [150, 20, 170, 40], "text": "검색"}]

    marker_id, result = match_step_by_screen_signature(
        step,
        current_signature,
        markers,
        current_image_path=str(current),
    )

    assert marker_id == 7
    assert result["matched"] is True
    assert result["mode"] == "roi_phash"
    assert result["reason"] == "roi_matched"


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


def test_set_result_card_queue_stores_visible_card_ratios():
    from agent.graph import nodes

    state = {
        "current_markers": [
            {"id": 10, "bbox": [100, 200, 300, 240], "text": "iOS 개발자"},
            {"id": 11, "bbox": [100, 245, 260, 270], "text": "보이저엑스"},
        ],
        "screen_signature": {
            "phash": "0" * 16,
            "size": [1000, 1000],
            "anchors": ["iOS 개발자", "보이저엑스"],
        },
        "reflex_state_key": "ocr#list",
        "recent_images": ["screen.png"],
        "marked_image": "marked.png",
        "recipe_params": {"target_count": 2},
        "extracted_jd": {},
    }

    result, _jd, _plan, _step = nodes._dispatch_state(
        "set_result_card_queue",
        {"cards": [{"marker_id": 10, "title": "iOS 개발자", "company": "보이저엑스"}]},
        {},
        [],
        0,
        current_url="https://www.wanted.co.kr/search?query=ios",
        state=state,
    )

    assert result["status"] == "success"
    queue = result["_result_card_queue"]
    assert queue[0]["title"] == "iOS 개발자"
    assert queue[0]["company"] == "보이저엑스"
    assert queue[0]["bbox_ratio"] == [0.1, 0.2, 0.3, 0.24]
    assert result["_result_page_memory"]["state_key"] == "ocr#list"


def test_card_queue_replay_after_go_back_uses_cached_bbox():
    from agent.graph import nodes

    state = {
        "result_card_queue": [
            {
                "queue_id": "card-2",
                "status": "pending",
                "title": "두번째 iOS 개발자",
                "company": "넛지",
                "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                "center_ratio": [0.4, 0.425],
                "target": {
                    "text": "두번째 iOS 개발자",
                    "semantic_label": "두번째 iOS 개발자",
                    "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                    "center_ratio": [0.4, 0.425],
                },
            }
        ],
        "result_page_memory": {
            "state_key": "ocr#list",
            "screen_signature": {
                "phash": "0" * 16,
                "anchors": ["두번째 iOS 개발자"],
                "size": [1000, 1000],
            },
        },
    }

    msg, markers, trace = nodes._queue_replay_after_return(
        state,
        {"action": "go_back"},
        "https://www.wanted.co.kr/search?query=ios",
        "ocr#list",
        [],
        {"phash": "0" * 16, "anchors": ["두번째 iOS 개발자"], "size": [1000, 1000]},
    )

    assert msg is not None
    assert trace["hit"] is True
    call = msg.tool_calls[0]
    assert call["name"] == "click_marker"
    assert call["args"]["queue_id"] == "card-2"
    assert markers[0]["bbox"] == [300, 400, 500, 450]


def test_card_queue_replay_waits_until_active_card_is_done():
    from agent.graph import nodes

    state = {
        "active_result_card": {"queue_id": "card-1", "status": "active", "title": "첫번째 iOS 개발자"},
        "result_card_queue": [
            {"queue_id": "card-1", "status": "active", "title": "첫번째 iOS 개발자"},
            {
                "queue_id": "card-2",
                "status": "pending",
                "title": "두번째 iOS 개발자",
                "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
            },
        ],
        "result_page_memory": {
            "state_key": "ocr#list",
            "screen_signature": {"phash": "0" * 16, "anchors": ["두번째 iOS 개발자"], "size": [1000, 1000]},
        },
    }

    msg, _markers, trace = nodes._queue_replay_after_return(
        state,
        {"action": "go_back"},
        "https://www.wanted.co.kr/search?query=ios",
        "ocr#list",
        [],
        {"phash": "0" * 16, "anchors": ["두번째 iOS 개발자"], "size": [1000, 1000]},
    )

    assert msg is None
    assert trace["reason"] == "active_card_not_completed"


def test_card_queue_marks_active_and_done():
    from agent.graph import nodes

    queue = [{"queue_id": "card-1", "status": "pending", "title": "A"}]

    queue, active = nodes._mark_result_card_active(queue, {"queue_id": "card-1"})
    assert queue[0]["status"] == "active"
    assert active["queue_id"] == "card-1"

    queue, active = nodes._complete_active_result_card(queue, active)
    assert queue[0]["status"] == "done"
    assert active == {}


def test_card_queue_marks_active_when_card_click_uses_title_label():
    from agent.graph import nodes

    queue = [{"queue_id": "card-1", "status": "pending", "title": "iOS 핵심 시스템 CTO 및 PM급 엔지니어"}]
    args = {
        "marker_id": 170,
        "target_component": "job_card",
        "target_role": "link",
        "target_label": "iOS 핵심 시스템 CTO 및 PM급 엔지니어",
    }

    assert nodes._result_card_click_matches_queue(queue, args) is True
    queue, active = nodes._mark_result_card_active(queue, args)
    assert queue[0]["status"] == "active"
    assert active["queue_id"] == "card-1"


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


def test_recipe_store_filters_recipe_by_site_and_task_category(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "recipes.db")
    store.commit_recipe(
        "wanted",
        "goal",
        [{"seq": 0, "state_key": "state-a", "action": "click_marker", "target": {"text": "검색"}}],
        metadata={"task_category": "검색"},
    )

    assert store.get_recipe("state-a", site="wanted", task_category="검색") is not None
    assert store.get_recipe("state-a", site="wanted", task_category="로그인") is None
    assert store.get_recipe("state-a", site="other", task_category="검색") is None
    assert len(store.get_site_recipes("wanted", task_category="검색")) == 1
    assert store.get_site_recipes("wanted", task_category="로그인") == []


def test_reflex_node_builds_action_tool_call(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.graph import nodes
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    saved = tmp_path / "saved-search.png"
    current = tmp_path / "current-search.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([10, 10, 70, 40], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [10, 10, 70, 40], [200, 120])

    class FakeStore:
        def get_recipe(self, state_key, site=None, task_category=None):
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
                        roi_signature=roi_signature,
                        target={
                            "text": "검색",
                            "region": "top-left",
                            "ordinal": 0,
                            "bbox_ratio": [0.05, 0.0833, 0.35, 0.3333],
                            "center_ratio": [0.2, 0.2083],
                        },
                        param={"text": "ai 엔지니어"},
                    )
                ],
            )

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = nodes.reflex_node(
        {
            "goal": "ai 엔지니어 공고 찾아줘",
            "current_url": "https://www.wanted.co.kr",
            "reflex_state_key": "state-a",
            "screen_signature": {"size": [200, 120]},
            "recent_images": [current],
            "current_markers": [{"id": 7, "bbox": [10, 10, 70, 40], "text": "검색"}],
        }
    )

    msg = result["last_action_result"]
    assert result["reflex_hit"] is True
    assert msg.tool_calls[0]["name"] == "type_in_marker"
    assert msg.tool_calls[0]["args"] == {"marker_id": 7, "text": "ai 엔지니어"}
    assert len(msg.tool_calls) == 1


def test_reflex_node_uses_roi_signature_when_available(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.graph import nodes
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [150, 20, 170, 40], [200, 200])

    class FakeStore:
        def get_recipe(self, state_key, site=None, task_category=None):
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
                            "size": [200, 200],
                            "anchors": ["검색"],
                        },
                        roi_signature=roi_signature,
                        target={
                            "text": "검색",
                            "bbox_ratio": [0.75, 0.1, 0.85, 0.2],
                            "center_ratio": [0.8, 0.15],
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
                "phash": "0" * 16,
                "size": [200, 200],
                "anchors": ["검색"],
            },
            "recent_images": [current],
            "current_markers": [{"id": 77, "bbox": [150, 20, 170, 40], "text": "검색"}],
        }
    )

    assert result["reflex_hit"] is True
    assert result["last_action_result"].tool_calls[0]["args"] == {"marker_id": 77}
    assert result["reflex_trace"]["hit"] is True
    assert result["reflex_trace"]["lookup"] == "exact"
    call_id = result["last_action_result"].tool_calls[0]["id"]
    assert result["reflex_trace"]["tool_calls"][call_id]["match_mode"] == "roi_phash"
    assert result["reflex_trace"]["tool_calls"][call_id]["phash"]["distance"] == 0


def test_reflex_node_rejects_signed_step_when_roi_missing(monkeypatch):
    from agent.graph import nodes
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    class FakeStore:
        def get_recipe(self, state_key, site=None, task_category=None):
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
    assert result["reflex_trace"]["reason"] == "no_candidate_passed"
    assert result["reflex_trace"]["candidates"][0]["steps"][0]["reason"] == "roi_signature_missing"


def test_reflex_node_does_not_broad_match_legacy_site_recipe(monkeypatch):
    from agent.graph import nodes
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    class FakeStore:
        def get_recipe(self, state_key, site=None, task_category=None):
            return None

        def get_site_recipes(self, site, task_category=None):
            return [
                (
                    "old-scroll",
                    SiteRecipe(
                        site=site,
                        goal="goal",
                        steps=[
                            RecipeStep(
                                seq=0,
                                state_key="old-scroll-state",
                                action="scroll",
                                replay_mode="fixed",
                                param={"direction": "down"},
                            )
                        ],
                    ),
                ),
                (
                    "old-click",
                    SiteRecipe(
                        site=site,
                        goal="goal",
                        steps=[
                            RecipeStep(
                                seq=0,
                                state_key="old-click-state",
                                action="click_marker",
                                replay_mode="fixed",
                                target={"text": "검색", "center_ratio": [0.8, 0.1]},
                            )
                        ],
                    ),
                ),
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = nodes.reflex_node(
        {
            "goal": "ios 개발자 공고 2개 찾아줘",
            "current_url": "https://www.wanted.co.kr",
            "reflex_state_key": "current-state",
            "recipe_params": {"site": "wanted"},
            "screen_signature": {"size": [1000, 1000]},
            "current_markers": [{"id": 77, "bbox": [790, 80, 830, 120], "text": "검색"}],
        }
    )

    assert result["reflex_hit"] is False
    assert result["reflex_trace"]["reason"] == "no_recipe"
    assert result["reflex_trace"]["candidate_count"] == 0


def test_reflex_node_replaces_type_input_slot(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.graph import nodes
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe
    from shared.schema.skill_schema import RecipeSkillMetadata, SkillInputSlot

    saved = tmp_path / "slot-saved.png"
    current = tmp_path / "slot-current.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([10, 10, 70, 40], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [10, 10, 70, 40], [200, 120])

    class FakeStore:
        def get_recipe(self, state_key, site=None, task_category=None):
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
                        roi_signature=roi_signature,
                        target={
                            "text": "Search",
                            "region": "top-left",
                            "ordinal": 0,
                            "bbox_ratio": [0.05, 0.0833, 0.35, 0.3333],
                            "center_ratio": [0.2, 0.2083],
                        },
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
            "screen_signature": {"size": [200, 120]},
            "recent_images": [current],
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


def test_reflex_node_uses_site_recipe_when_exact_state_misses(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw
    from agent.graph import nodes
    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe
    from shared.schema.skill_schema import RecipeSkillMetadata

    saved = tmp_path / "site-saved.png"
    current = tmp_path / "site-current.png"
    for path in [saved, current]:
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        image.save(path)
    roi_signature = compute_target_roi_signature(saved, [150, 20, 170, 40], [200, 200])

    class FakeStore:
        def get_recipe(self, state_key, site=None, task_category=None):
            assert state_key == "new-state"
            return None

        def get_site_recipes(self, site, task_category=None):
            assert site == "wanted"
            return [
                (
                    "recorded-state",
                    SiteRecipe(
                        site="wanted",
                        goal="collect jobs",
                        skill_metadata=RecipeSkillMetadata(task_category="검색"),
                        steps=[
                            RecipeStep(
                                seq=0,
                                state_key="recorded-state",
                                state_anchors=["검색", "채용"],
                                action="click_marker",
                                replay_mode="fixed",
                                roi_signature=roi_signature,
                                target={
                                    "text": "검색",
                                    "bbox_ratio": [0.75, 0.1, 0.85, 0.2],
                                    "center_ratio": [0.8, 0.15],
                                },
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = nodes.reflex_node(
        {
            "goal": "android 개발자 공고 찾아줘",
            "reflex_state_key": "new-state",
            "screen_signature": {"size": [200, 200]},
            "recent_images": [current],
            "current_markers": [{"id": 7, "bbox": [150, 20, 170, 40], "text": "검색"}],
            "recipe_params": {"site": "wanted", "query": "android 개발자", "task_category": "검색"},
        }
    )

    assert result["reflex_hit"] is True
    assert result["reflex_state_key"] == "new-state"
    assert result["reflex_trace"]["lookup"] == "site"
    assert result["last_action_result"].tool_calls[0]["args"] == {"marker_id": 7}


def test_reflex_node_skips_site_recipe_when_task_category_mismatches(monkeypatch):
    from agent.graph import nodes
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe
    from shared.schema.skill_schema import RecipeSkillMetadata

    class FakeStore:
        def get_recipe(self, state_key, site=None, task_category=None):
            return None

        def get_site_recipes(self, site, task_category=None):
            assert site == "wanted"
            return [
                (
                    "login-state",
                    SiteRecipe(
                        site="wanted",
                        goal="login",
                        skill_metadata=RecipeSkillMetadata(task_category="로그인"),
                        steps=[
                            RecipeStep(
                                seq=0,
                                state_key="login-state",
                                action="click_marker",
                                replay_mode="fixed",
                                roi_signature={"phash": "0" * 16},
                                target={"text": "로그인"},
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())

    result = nodes.reflex_node(
        {
            "goal": "android 개발자 공고 찾아줘",
            "reflex_state_key": "new-state",
            "screen_signature": {"size": [200, 200]},
            "current_markers": [{"id": 7, "bbox": [150, 20, 170, 40], "text": "검색"}],
            "recipe_params": {"site": "wanted", "query": "android 개발자", "task_category": "검색"},
        }
    )

    assert result["reflex_hit"] is False
    assert result["reflex_trace"]["reason"] == "no_recipe"
    assert result["reflex_trace"]["task_category_skips"] == 1


def test_reflex_node_rejects_similar_recipe_when_target_does_not_match(monkeypatch):
    from agent.graph import nodes
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    class FakeStore:
        def get_recipe(self, state_key, site=None, task_category=None):
            return None

        def get_site_recipes(self, site, task_category=None):
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
    assert route_after_perception({}) == "reflex"

    monkeypatch.setenv("REFLEX_ENABLED", "0")
    assert route_after_perception({}) == "reasoning"

    monkeypatch.setenv("REFLEX_ENABLED", "1")
    assert route_after_perception({}) == "reflex"
    assert route_after_perception({"transition_status": "pending"}) == "perception"
    assert route_after_perception({"transition_status": "unknown", "transition_source": "reflex"}) == "reasoning"
    assert route_after_perception({"transition_status": "unknown", "transition_source": "page_policy"}) == "reasoning"
    assert route_after_perception({"transition_status": "unknown", "transition_source": "autonomous"}) == "reflex"
    assert route_after_perception({"transition_status": "ready"}) == "reflex"
    assert route_after_perception({"queue_replay_hit": True}) == "action"

    assert route_after_reflex({"reflex_hit": True}) == "action"
    assert route_after_reflex({"reflex_hit": False}) == "reasoning"


def test_detail_ui_context_compacts_ocr_markers_into_ordered_lines(monkeypatch):
    from agent.graph import nodes

    monkeypatch.setenv("VISION_DETAIL_SECTION_MIN_TEXT_MARKERS", "1")
    markers = [
        {"id": 1, "bbox": [80, 80, 130, 110], "text": "채용"},
        {"id": 2, "bbox": [100, 200, 210, 235], "text": "모바일"},
        {"id": 3, "bbox": [220, 200, 360, 235], "text": "엔지니어"},
        {"id": 10, "bbox": [100, 400, 210, 435], "text": "주요업무"},
        {"id": 11, "bbox": [100, 450, 135, 480], "text": "iOS"},
        {"id": 12, "bbox": [150, 450, 180, 480], "text": "앱"},
        {"id": 13, "bbox": [195, 450, 250, 480], "text": "개발"},
        {"id": 20, "bbox": [100, 540, 220, 575], "text": "자격요건"},
        {"id": 21, "bbox": [100, 590, 170, 620], "text": "Swift"},
        {"id": 22, "bbox": [190, 590, 250, 620], "text": "경험"},
        {"id": 30, "bbox": [100, 700, 290, 735], "text": "상세 정보 더 보기"},
        {"id": 31, "bbox": [820, 210, 870, 260], "text": "상호작용 가능한 요소 (button)"},
    ]

    context = nodes._build_ui_context(markers, current_url="https://www.wanted.co.kr/wd/1")

    assert "상세 페이지 OCR 본문" in context
    assert "식별된 텍스트 요소" not in context
    assert "[공고 소개]" not in context
    assert "모바일 엔지니어" in context
    assert "[주요업무]" not in context
    assert "주요업무" in context
    assert "iOS 앱 개발" in context
    assert "[자격요건]" not in context
    assert "자격요건" in context
    assert "Swift 경험" in context
    assert "수집 진행용 클릭 후보" in context
    assert "[id: 30] 상세 정보 더 보기" in context
    assert "채용" not in context


def test_detail_ui_context_keeps_heading_lines_for_llm_judgment(monkeypatch):
    from agent.graph import nodes

    monkeypatch.setenv("VISION_DETAIL_SECTION_MIN_TEXT_MARKERS", "1")
    markers = [
        {"id": 1, "bbox": [100, 200, 210, 235], "text": "기술스택"},
        {"id": 2, "bbox": [100, 250, 170, 280], "text": "Flutter"},
        {"id": 3, "bbox": [190, 250, 260, 280], "text": "WebRTC"},
        {"id": 4, "bbox": [100, 340, 160, 375], "text": "태그"},
        {"id": 5, "bbox": [100, 390, 190, 420], "text": "식대지원"},
        {"id": 6, "bbox": [210, 390, 320, 420], "text": "장비지원"},
    ]

    context = nodes._build_ui_context(markers, current_url="https://www.wanted.co.kr/wd/1")

    assert "[기술스택]" not in context
    assert "[태그/혜택]" not in context
    assert "기술스택" in context
    assert "태그" in context
    assert "Flutter WebRTC" in context
    assert "식대지원 장비지원" in context


def test_non_detail_ui_context_keeps_raw_marker_list(monkeypatch):
    from agent.graph import nodes

    monkeypatch.setenv("VISION_DETAIL_SECTION_MIN_TEXT_MARKERS", "1")
    markers = [
        {"id": 1, "bbox": [100, 200, 180, 230], "text": "검색"},
        {"id": 2, "bbox": [100, 260, 220, 290], "text": "iOS 개발자"},
    ]

    context = nodes._build_ui_context(markers, current_url="https://www.wanted.co.kr/search?query=ios")

    assert "상세 페이지 OCR 섹션 요약" not in context
    assert "식별된 텍스트 요소" in context
    assert "[id: 2] iOS 개발자" in context


def test_detail_lightweight_marked_image_draws_action_candidates(tmp_path, monkeypatch):
    from PIL import Image

    from agent.graph import nodes

    monkeypatch.setenv("VISION_DETAIL_LIGHTWEIGHT_MARKED_IMAGE_ENABLED", "1")
    image_path = tmp_path / "screen_detail.png"
    Image.new("RGB", (420, 360), "white").save(image_path)
    markers = [
        {"id": 1, "bbox": [40, 40, 180, 80], "text": "채용"},
        {"id": 30, "bbox": [100, 270, 260, 310], "text": "상세 정보 더 보기"},
    ]

    output_path = nodes._build_detail_lightweight_marked_image(
        image_path,
        markers,
        "https://www.wanted.co.kr/wd/1",
    )

    assert output_path
    assert output_path.endswith(".jpg")
    assert (tmp_path / "light_marked_screen_detail.jpg").exists()


def test_detail_lightweight_marked_image_skips_non_detail_page(tmp_path, monkeypatch):
    from PIL import Image

    from agent.graph import nodes

    monkeypatch.setenv("VISION_DETAIL_LIGHTWEIGHT_MARKED_IMAGE_ENABLED", "1")
    image_path = tmp_path / "screen_search.png"
    Image.new("RGB", (420, 360), "white").save(image_path)

    output_path = nodes._build_detail_lightweight_marked_image(
        image_path,
        [{"id": 30, "bbox": [100, 270, 260, 310], "text": "상세 정보 더 보기"}],
        "https://www.wanted.co.kr/search?query=ios",
    )

    assert output_path == ""


def test_detail_ocr_buffer_accumulates_unique_detail_lines(monkeypatch):
    from agent.graph import nodes

    monkeypatch.setenv("VISION_DETAIL_OCR_BUFFER_ENABLED", "1")
    markers = [
        {"id": 1, "bbox": [100, 200, 210, 235], "text": "주요업무"},
        {"id": 2, "bbox": [100, 250, 160, 280], "text": "iOS"},
        {"id": 3, "bbox": [180, 250, 250, 280], "text": "개발"},
        {"id": 4, "bbox": [100, 320, 180, 350], "text": "자격요건"},
        {"id": 5, "bbox": [100, 370, 180, 400], "text": "Swift"},
    ]

    first = nodes._update_detail_ocr_buffer(
        {},
        markers,
        "https://www.wanted.co.kr/wd/1",
        "screen_a.png",
    )
    second = nodes._update_detail_ocr_buffer(
        first,
        markers,
        "https://www.wanted.co.kr/wd/1",
        "screen_b.png",
    )

    assert first["stats"]["added_lines_last_screen"] == 4
    assert len(first["lines"]) == 4
    assert second["stats"]["added_lines_last_screen"] == 0
    assert second["stats"]["duplicate_lines_last_screen"] == 4
    assert len(second["lines"]) == 4


def test_detail_ocr_buffer_context_guides_finish_detail_reading(monkeypatch):
    from agent.graph import nodes

    monkeypatch.setenv("VISION_DETAIL_OCR_BUFFER_ENABLED", "1")
    state = {
        "detail_ocr_buffer": {
            "url": "https://www.wanted.co.kr/wd/1",
            "lines": [{"text": "주요업무 iOS 개발"}, {"text": "자격요건 Swift"}],
            "stats": {
                "screen_count": 2,
                "added_lines_last_screen": 1,
                "duplicate_lines_last_screen": 1,
            },
        }
    }

    context = nodes._compact_detail_ocr_buffer_context(state, "https://www.wanted.co.kr/wd/1")

    assert "상세 OCR 누적 상태" in context
    assert "finish_detail_reading" in context
    assert "중간 DB 추출" in context


def test_finish_detail_reading_merges_buffer_extraction_and_clears_buffer(monkeypatch):
    from agent.graph import nodes

    monkeypatch.setattr(
        nodes,
        "_extract_job_from_detail_ocr_buffer",
        lambda state, current_url: {
            "company_name": "보이저엑스",
            "position": "iOS 개발자",
            "url": current_url,
            "requirements": ["Swift"],
        },
    )

    result, current_jd, plan, step = nodes._dispatch_state(
        "finish_detail_reading",
        {"page_role": "job_detail", "detail_complete": True},
        {},
        [],
        0,
        current_url="https://www.wanted.co.kr/wd/1",
        state={
            "detail_ocr_buffer": {
                "url": "https://www.wanted.co.kr/wd/1",
                "lines": [{"text": "자격요건 Swift"}],
            }
        },
    )

    assert result["status"] == "success"
    assert result["_detail_ocr_buffer"] == {}
    assert current_jd["공고목록"][0]["position"] == "iOS 개발자"
    assert plan == []
    assert step == 0


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
    assert submission["task_category"] == ""
    assert submission["skill_metadata_evidence"]["site"] == "wanted"
    assert submission["skill_metadata_evidence"]["task_category"] == ""
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
        "task_category": "검색",
        "keyword": "ai engineer",
        "review_attempt": 0,
        "skill_metadata_evidence": {"site": "wanted", "task_category": "검색"},
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


def test_candidate_reviewer_records_review_without_active_promotion(tmp_path):
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
    assert seen["payload"]["task_category"] == "검색"
    assert seen["payload"]["steps"][0]["state_key"] == "state-a"
    assert seen["payload"]["transition_observations"][0]["action_seq"] == 0
    assert review["decision"] == "accept"
    assert candidate["status"] == "accepted"
    assert recipes == []


def test_candidate_reviewer_promote_requires_roi_signature(tmp_path):
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
        mode="promote",
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
    assert review["decision"] == "accept"
    assert review["promotion"]["promoted"] is False
    assert review["promotion"]["skipped_steps"][0]["reason"] == "roi_signature_missing"
    assert recipes == []


def test_candidate_reviewer_promotes_roi_fixed_steps_only(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _sample_recipe_candidate_submission()
    submission["recorded_steps"][0].update(
        {
            "roi_signature": {
                "algorithm": "roi-phash-dct64-v1",
                "phash": "0" * 16,
                "crop_rect_ratio": [0.7, 0.0, 0.9, 0.2],
            },
            "target": {
                "text": "검색",
                "bbox_ratio": [0.75, 0.1, 0.85, 0.2],
                "center_ratio": [0.8, 0.15],
            },
        }
    )
    submission["recorded_steps"].append(
        {
            "seq": 1,
            "state_key": "state-results",
            "action": "click_marker",
            "roi_signature": {
                "algorithm": "roi-phash-dct64-v1",
                "phash": "f" * 16,
                "crop_rect_ratio": [0.1, 0.2, 0.6, 0.4],
            },
            "target": {"text": "Specific Job Title"},
            "component": "job_card_title",
        }
    )
    submission["recorded_steps"].append(
        {
            "seq": 2,
            "state_key": "state-a",
            "action": "press_key",
            "param": {"key": "enter"},
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
        mode="promote",
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
                    {"seq": 2, "action": "press_key", "replay_mode": "fixed"},
                ],
            },
            "transition_contracts": [
                {
                    "seq": 0,
                    "contract": {
                        "common_ready_cues": [{"kind": "text_any", "values": ["검색어"]}],
                    },
                }
            ],
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(tmp_path / "critic.db").get_by_site("wanted")

    assert review["decision"] == "accept"
    assert review["promotion"]["promoted"] is True
    assert review["promotion"]["promoted_step_count"] == 1
    assert len(recipes) == 1
    assert recipes[0]["steps"][0]["state_key"] == "state-a"
    assert recipes[0]["steps"][0]["replay_mode"] == "fixed"
    assert len(recipes[0]["steps"]) == 1
    assert recipes[0]["skill_metadata"]["task_category"] == "검색"
    assert recipes[0]["steps"][0]["transition_contract"]["common_ready_cues"][0]["values"] == ["검색어"]
    assert {"seq": 2, "action": "press_key", "reason": "non_target_action"} in review["promotion"]["skipped_steps"]


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
    monkeypatch.setattr(rs, "_commit_recipe_candidate", lambda *args, **kwargs: called.append(args) or "candidate-1")
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
        return "candidate-1"

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


def test_realtime_recipe_learning_mode_promote_is_record_only(monkeypatch):
    from agent.tools import realtime_scraping as rs

    monkeypatch.setenv("VISION_RECIPE_LEARNING_MODE", "promote")
    monkeypatch.setattr(rs, "_persist_collected_data", lambda extracted, keyword: 1)
    seen = {}
    def fake_commit_recipe_candidate(submission, review, source, submission_id, mode):
        seen["mode"] = mode
        return "candidate-1"

    monkeypatch.setattr(
        rs,
        "_commit_recipe_candidate",
        fake_commit_recipe_candidate,
    )
    monkeypatch.setattr("agent.recipe.submission_store.SubmissionStore.commit_submission", lambda self, submission, review=None, source="": "worker-run-critic:0")

    _count, submission, _review, _submission_id = rs.persist_accepted_worker_result(
        _sample_worker_result_for_learning_mode(),
        {"decision": "accept", "recipe_candidate": True, "confidence": 0.7},
    )

    assert seen["mode"] == "record"
    assert submission["recipe_learning_mode"] == "record"
    assert submission["recipe_candidate_id"] == "candidate-1"
    assert "recipe_candidate_review" not in submission


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
    assert set(seen) == set(candidate_ids)
    assert RecipeStore(db_path).get_by_site("wanted") == []
    for candidate_id in candidate_ids:
        candidate = store.get_candidate(candidate_id)
        assert candidate["status"] == "accepted"


def test_process_recipe_candidates_promote_mode_writes_active_recipe(tmp_path):
    from agent.recipe.candidate_reviewer import process_recipe_candidates
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    db_path = tmp_path / "batch.db"
    store = RecipeCandidateStore(db_path)
    submission = _sample_recipe_candidate_submission()
    submission["recorded_steps"][0].update(
        {
            "roi_signature": {
                "algorithm": "roi-phash-dct64-v1",
                "phash": "0" * 16,
                "crop_rect_ratio": [0.7, 0.0, 0.9, 0.2],
            },
            "target": {
                "text": "검색",
                "bbox_ratio": [0.75, 0.1, 0.85, 0.2],
                "center_ratio": [0.8, 0.15],
            },
        }
    )
    candidate_id = store.commit_candidate(
        submission,
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
    assert candidate["status"] == "accepted"
    assert len(recipes) == 1
    assert recipes[0]["steps"][0]["replay_mode"] == "fixed"
    assert result["results"][0]["promotion"]["promoted"] is True


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

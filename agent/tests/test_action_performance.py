def test_action_coords_reuse_last_capture_region():
    from agent.tools.actions import ActionTools

    class FakePerception:
        last_region = {"left": 10, "top": 20, "width": 300, "height": 200}
        scale_x = 1.0
        scale_y = 1.0

        def __init__(self):
            self.region_reads = 0

        def _get_browser_region(self):
            self.region_reads += 1
            return {"left": 999, "top": 999, "width": 300, "height": 200}

    perception = FakePerception()
    action_tools = object.__new__(ActionTools)
    action_tools.perception = perception

    assert action_tools._get_absolute_coords([20, 30, 60, 70]) == (50, 70)
    assert perception.region_reads == 0


def test_release_address_bar_focus_relies_on_pyautogui_pause(monkeypatch):
    from agent.tools.perception import PerceptionEngine

    sleeps = []

    class FakePyAutoGUI:
        PAUSE = 0.1

        def __init__(self):
            self.pressed = []

        def press(self, key):
            self.pressed.append(key)

    fake_pyautogui = FakePyAutoGUI()
    engine = object.__new__(PerceptionEngine)
    monkeypatch.setattr("agent.tools.perception.time.sleep", lambda sec: sleeps.append(sec))

    engine.release_address_bar_focus(fake_pyautogui, key_pause=0.02)

    assert fake_pyautogui.pressed == ["esc", "esc"]
    assert sleeps == []
    assert fake_pyautogui.PAUSE == 0.1


def test_open_browser_ignores_current_url_for_decision(monkeypatch):
    from agent.tools.actions import ActionTools

    opened = []

    class FakePerception:
        _browser_window_id = None

    action_tools = object.__new__(ActionTools)
    action_tools.perception = FakePerception()
    monkeypatch.setattr(action_tools, "_bound_browser_window_exists", lambda: False)
    monkeypatch.setattr(
        action_tools,
        "_open_url_in_new_window",
        lambda url: opened.append(url) or {"opened": True, "url": url, "reason": "new_browser_window"},
    )

    result = action_tools.open_browser(
        "https://www.wanted.co.kr",
        current_url="https://www.wanted.co.kr/search?query=data",
    )

    assert result["status"] == "success"
    assert opened == ["https://www.wanted.co.kr"]
    assert result["result"]["opened"] is True
    assert result["result"]["url"] == "https://www.wanted.co.kr"


def test_open_browser_does_not_run_duplicate_ocr_readiness_check(monkeypatch):
    from agent.tools.actions import ActionTools

    opened = []

    class FakePerception:
        _browser_window_id = None

        def capture_screen(self, *args, **kwargs):
            raise AssertionError("open_browser must leave observation to perception_node")

        def analyze_ui(self, *args, **kwargs):
            raise AssertionError("open_browser must not run OCR")

    action_tools = object.__new__(ActionTools)
    action_tools.perception = FakePerception()
    monkeypatch.setattr(action_tools, "_bound_browser_window_exists", lambda: False)
    monkeypatch.setattr(
        action_tools,
        "_open_url_in_new_window",
        lambda url: opened.append(url) or {"opened": True, "url": url, "reason": "new_browser_window"},
    )

    result = action_tools.open_browser("https://www.wanted.co.kr", current_url="")

    assert result["status"] == "success"
    assert opened == ["https://www.wanted.co.kr"]
    assert "visual_ready" not in result["result"]

def test_action_node_handles_open_browser_without_screen_change(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        assert action_name == "open_browser"
        assert current_url == "https://www.wanted.co.kr"
        return {
            "status": "success",
            "action": "open_browser",
            "result": {
                "opened": False,
                "url": "https://www.wanted.co.kr",
                "reason": "ocr_screen_already_ready",
            },
        }

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    result = nodes.action_node({
        "current_markers": [],
        "current_url": "https://www.wanted.co.kr",
        "current_url_stale": False,
        "ui_context": "already captured",
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[{"name": "open_browser", "args": {"url": "https://www.wanted.co.kr"}, "id": "1"}],
        ),
    })

    assert result["last_action_screen_changed"] is False
    assert result["current_url_stale"] is False
    assert result["current_url"] == "https://www.wanted.co.kr"


def test_action_node_records_target_metadata(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        assert get_bbox(args["marker_id"]) == [10, 20, 110, 80]
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    result = nodes.action_node({
        "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "Data Scientist"}],
        "current_url": "https://www.wanted.co.kr/search?query=data",
        "current_url_stale": False,
        "reflex_state_key": "state-results",
        "marked_image": "marked.jpg",
        "recent_images": ["screen.jpg"],
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[{"name": "click_marker", "args": {"marker_id": 1}, "id": "1"}],
        ),
    })

    action = result["action_history"][0]
    assert action["state_key"] == "state-results"
    assert action["before_url"] == "https://www.wanted.co.kr/search?query=data"
    assert action["before_screenshot"] == "screen.jpg"
    assert action["before_marked_image"] == "marked.jpg"
    assert action["target"] == {
        "marker_id": 1,
        "text": "Data Scientist",
        "bbox": [10, 20, 110, 80],
        "center": [60, 50],
    }
    episode = result["feedback_episodes"][0]
    assert episode["proposal"]["action"] == "click_marker"
    assert episode["proposal"]["target"]["text"] == "Data Scientist"
    assert episode["observation"]["before"]["state_key"] == "state-results"
    assert episode["feedback"]["label"] == "partial"


def test_action_node_uses_action_history_seq_for_recorded_steps(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        assert action_name == "click_marker"
        assert get_bbox(args["marker_id"]) == [10, 20, 110, 80]
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    result = nodes.action_node({
        "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "Data Scientist"}],
        "current_url": "https://www.wanted.co.kr/search?query=data",
        "current_url_stale": False,
        "reflex_state_key": "state-results",
        "action_history": [
            {"status": "success", "action": "open_browser", "args": {}, "state_key": "state-home"},
            {"status": "success", "action": "press_key", "args": {"key": "enter"}, "state_key": "state-home"},
        ],
        "recorded_steps": [
            {"seq": 0, "action": "open_browser", "state_key": "state-home"},
        ],
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[
                {"name": "update_plan_progress", "args": {"current_step": 1}, "id": "1"},
                {"name": "click_marker", "args": {"marker_id": 1, "reason": "open first result"}, "id": "2"},
            ],
        ),
    })

    assert [episode["seq"] for episode in result["feedback_episodes"]] == [2, 3]
    assert result["recorded_steps"][0]["seq"] == 3
    assert result["recorded_steps"][0]["intent"] == "open first result"


def test_action_node_carries_reflex_transition_contract_to_next_perception(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    monkeypatch.setattr(
        nodes,
        "_dispatch_ui",
        lambda action_name, args, get_bbox, current_url="": {
            "status": "success",
            "action": action_name,
            "result": "clicked",
        },
    )
    contract = {
        "common_ready_cues": [{"kind": "text_any", "values": ["포지션"]}],
        "outcomes": [{"name": "results_found", "cues": [{"kind": "text_any", "values": ["회사명"]}]}],
        "timeout_sec": 5,
    }
    result = nodes.action_node(
        {
            "goal": "android 개발자 공고",
            "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "검색"}],
            "current_url": "https://www.wanted.co.kr",
            "current_url_stale": False,
            "action_history": [],
            "recorded_steps": [],
            "extracted_jd": {},
            "is_finished": False,
            "collected_data": [],
            "error_count": 0,
            "current_plan_step": 0,
            "plan": [],
            "recipe_params": {"query": "android 개발자"},
            "reflex_hit": True,
            "reflex_trace": {
                "hit": True,
                "recipe_key": "recipe-home",
                "lookup": "exact",
                "similarity": 1.0,
                "tool_calls": {
                    "reflex-call": {
                        "seq": 0,
                        "action": "click_marker",
                        "match_mode": "phash",
                        "marker_id": 1,
                    }
                },
            },
            "reflex_transition_contracts": {"reflex-call": contract},
            "last_action_result": AIMessage(
                content="",
                tool_calls=[{"name": "click_marker", "args": {"marker_id": 1}, "id": "reflex-call"}],
            ),
        }
    )

    pending = result["pending_transition"]
    assert pending["action_seq"] == 0
    assert pending["source"] == "reflex"
    assert pending["contract"] == contract
    assert pending["params"]["query"] == "android 개발자"
    assert result["action_history"][0]["reflex_recipe_key"] == "recipe-home"
    assert result["action_history"][0]["reflex_match"]["match_mode"] == "phash"


def test_action_node_blocks_repeated_same_state_ui_action(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    def fake_dispatch_ui(*_args, **_kwargs):
        raise AssertionError("repeated action should be blocked before dispatch")

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    result = nodes.action_node({
        "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "Data Scientist"}],
        "current_url": "https://www.wanted.co.kr/search?query=data",
        "current_url_stale": False,
        "reflex_state_key": "state-results",
        "action_history": [
            {
                "status": "success",
                "action": "click_marker",
                "args": {"marker_id": 1},
                "state_key": "state-results",
            }
        ],
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[{"name": "click_marker", "args": {"marker_id": 1}, "id": "1"}],
        ),
    })

    action = result["action_history"][0]
    assert action["status"] == "skipped"
    assert action["reason"] == "same_state_repeat_blocked"
    assert action["target"]["text"] == "Data Scientist"
    assert action["observation_required"] is True
    assert result["error_count"] == 0
    assert result["last_action_screen_changed"] is True
    assert result["current_url_stale"] is True
    repeat_episode = result["feedback_episodes"][0]
    assert repeat_episode["feedback"]["label"] == "loop_risk"
    assert repeat_episode["feedback"]["reason"] == "same_state_repeat_blocked"


def test_action_node_stops_ui_chain_after_screen_boundary_action(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    calls = []

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        calls.append((action_name, dict(args)))
        get_bbox(args["marker_id"])
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    result = nodes.action_node({
        "current_markers": [
            {"id": 1, "bbox": [10, 20, 110, 80], "text": "first"},
            {"id": 2, "bbox": [10, 100, 110, 160], "text": "second"},
        ],
        "current_url": "https://www.wanted.co.kr/search?query=data",
        "current_url_stale": False,
        "reflex_state_key": "state-results",
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[
                {"name": "click_marker", "args": {"marker_id": 1}, "id": "1"},
                {"name": "click_marker", "args": {"marker_id": 2}, "id": "2"},
            ],
        ),
    })

    assert calls == [("click_marker", {"marker_id": 1})]
    assert result["action_history"][0]["status"] == "success"
    assert result["action_history"][1]["status"] == "skipped"
    assert result["action_history"][1]["reason"] == "chain_boundary_after_screen_change"
    assert result["last_action_screen_changed"] is True


def test_action_node_allows_type_then_enter_chain(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    calls = []

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        calls.append(action_name)
        if action_name == "type_in_marker":
            assert get_bbox(args["marker_id"]) == [10, 20, 110, 80]
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    result = nodes.action_node({
        "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "검색"}],
        "current_url": "https://www.wanted.co.kr",
        "current_url_stale": False,
        "reflex_state_key": "state-home",
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[
                {"name": "type_in_marker", "args": {"marker_id": 1, "text": "데이터 분석가"}, "id": "1"},
                {"name": "press_key", "args": {"key": "enter"}, "id": "2"},
            ],
        ),
    })

    assert calls == ["type_in_marker", "press_key"]
    assert [a["status"] for a in result["action_history"]] == ["success", "success"]
    assert result["last_action_screen_changed"] is True


def test_reasoning_prompt_lists_forbidden_same_screen_actions():
    from agent.graph import nodes

    messages = nodes._build_reasoning_messages({
        "goal": "collect jobs",
        "plan": [],
        "current_plan_step": 0,
        "extracted_jd": {},
        "ui_context": "[id: 1] Search",
        "current_url": "https://www.wanted.co.kr",
        "marked_image": "",
        "action_history": [
            {
                "status": "success",
                "action": "open_browser",
                "args": {"url": "https://www.wanted.co.kr"},
                "state_key": "state-home",
                "result": {
                    "opened": False,
                    "url": "https://www.wanted.co.kr",
                    "reason": "ocr_screen_already_ready",
                },
            },
            {
                "status": "skipped",
                "action": "open_browser",
                "args": {"url": "https://www.wanted.co.kr"},
                "state_key": "state-home",
                "reason": "same_state_repeat_blocked",
            },
        ],
    }, "")

    human_text = messages[-1].content
    assert "Execution constraints for the current screen" in human_text
    assert "open_browser" in human_text
    assert "https://www.wanted.co.kr" in human_text
    assert "same_state_repeat_blocked" in human_text


def test_reasoning_prompt_lists_visited_cards_and_collection_target():
    from agent.graph import nodes

    messages = nodes._build_reasoning_messages(
        {
            "goal": "collect two jobs",
            "plan": [],
            "current_plan_step": 0,
            "extracted_jd": {"jobs": [{"position": "First Job"}]},
            "ui_context": "[id: 1] First Job\n[id: 2] Second Job",
            "current_url": "https://www.wanted.co.kr/search",
            "marked_image": "",
            "recipe_params": {"target_count": 2},
            "action_history": [
                {
                    "status": "success",
                    "action": "click_marker",
                    "args": {
                        "marker_id": 1,
                        "target_component": "job_card_title",
                        "target_label": "First Job",
                    },
                    "target": {"text": "First Job"},
                }
            ],
        },
        "",
    )

    human_text = messages[-1].content
    assert "목표 공고 수: 2" in human_text
    assert "현재 수집 공고 수: 1" in human_text
    assert "First Job" in human_text
    assert "미방문 공고 제목" in human_text


def test_reasoning_prompt_compacts_large_state_inputs(monkeypatch):
    from agent.graph import nodes

    monkeypatch.setenv("VISION_REASONING_ACTION_HISTORY_LIMIT", "2")
    large_unused_text = "UNRELATED_FULL_TEXT_" + ("x" * 800)
    messages = nodes._build_reasoning_messages(
        {
            "goal": "collect jobs",
            "plan": [
                "초기 검색 완료",
                "현재 공고 상세 정보 수집",
                "목록으로 돌아가 다음 공고 클릭",
                "최종 보고서 작성",
            ],
            "current_plan_step": 1,
            "extracted_jd": {
                "jobs": [
                    {
                        "company_name": "Old Co",
                        "position": "Old Job",
                        "main_tasks": [large_unused_text],
                    },
                    {
                        "company_name": "Current Co",
                        "position": "Current Job",
                        "url": "https://www.wanted.co.kr/wd/current",
                        "main_tasks": ["Build product"],
                        "extra_notes": large_unused_text,
                    },
                ]
            },
            "ui_context": "[id: 1] Current Job",
            "current_url": "https://www.wanted.co.kr/wd/current",
            "marked_image": "",
            "action_history": [
                {
                    "status": "success",
                    "action": "scroll",
                    "args": {"direction": "down"},
                    "reason": "OLD_ACTION_REASON_SHOULD_NOT_BE_INCLUDED",
                },
                {
                    "status": "success",
                    "action": "click_marker",
                    "args": {"marker_id": 1, "target_label": "Old hidden action"},
                },
                {
                    "status": "success",
                    "action": "click_marker",
                    "args": {"marker_id": 2, "target_label": "Recent card"},
                },
                {
                    "status": "success",
                    "action": "update_extracted_info",
                    "args": {
                        "data_json": "{\"jobs\":[{\"company_name\":\"Current Co\"}]}",
                    },
                },
            ],
        },
        "",
    )

    human_text = messages[-1].content
    assert "계획 요약" in human_text
    assert "현재 공고 상세 정보 수집" in human_text
    assert "목록으로 돌아가 다음 공고 클릭" in human_text
    assert "최종 보고서 작성" not in human_text
    assert "수집 데이터 요약" in human_text
    assert "Current Co" in human_text
    assert "Current Job" in human_text
    assert "누락필드" in human_text
    assert "UNRELATED_FULL_TEXT" not in human_text
    assert "최근 행동 요약" in human_text
    assert "Old hidden action" not in human_text
    assert "Recent card" in human_text
    assert "이전 행동 내역" not in human_text


def test_action_node_blocks_same_state_repeat_across_intervening_states(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    def fake_dispatch_ui(*_args, **_kwargs):
        raise AssertionError("same state repeat should be blocked even after visiting another screen")

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    result = nodes.action_node({
        "current_markers": [{"id": 41, "bbox": [10, 20, 110, 80], "text": "Job card"}],
        "current_url": "https://www.wanted.co.kr/search?query=iOS",
        "current_url_stale": False,
        "reflex_state_key": "state-list",
        "action_history": [
            {
                "status": "success",
                "action": "click_marker",
                "args": {"marker_id": 41},
                "state_key": "state-list",
            },
            {
                "status": "success",
                "action": "scroll",
                "args": {"direction": "down"},
                "state_key": "state-detail",
            },
            {
                "status": "success",
                "action": "go_back",
                "args": {},
                "state_key": "state-detail-bottom",
            },
        ],
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="[reflex] cached 1 action(s)",
            tool_calls=[{"name": "click_marker", "args": {"marker_id": 41}, "id": "1"}],
        ),
    })

    action = result["action_history"][0]
    assert action["status"] == "skipped"
    assert action["reason"] == "same_state_repeat_blocked"
    assert result["error_count"] == 0
    assert result["last_action_screen_changed"] is True
    assert result["current_url_stale"] is True
    repeat_episode = result["feedback_episodes"][0]
    assert repeat_episode["feedback"]["label"] == "loop_risk"
    assert repeat_episode["feedback"]["reason"] == "same_state_repeat_blocked"

def test_action_node_blocks_same_state_repeat_when_marker_id_changes_but_target_text_matches(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    def fake_dispatch_ui(*_args, **_kwargs):
        raise AssertionError("semantic repeat should be blocked before dispatch")

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    result = nodes.action_node({
        "current_markers": [{"id": 42, "bbox": [10, 20, 110, 80], "text": "Job card"}],
        "current_url": "https://www.wanted.co.kr/search?query=iOS",
        "current_url_stale": False,
        "reflex_state_key": "state-list",
        "action_history": [
            {
                "status": "success",
                "action": "click_marker",
                "args": {"marker_id": 41},
                "state_key": "state-list",
                "target": {"marker_id": 41, "text": "Job card"},
            }
        ],
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="[reflex] cached 1 action(s)",
            tool_calls=[{"name": "click_marker", "args": {"marker_id": 42}, "id": "1"}],
        ),
    })

    action = result["action_history"][0]
    assert action["status"] == "skipped"
    assert action["reason"] == "same_state_repeat_blocked"
    assert result["error_count"] == 0
    assert result["last_action_screen_changed"] is True
    assert result["current_url_stale"] is True

def test_action_node_allows_state_update_after_screen_boundary(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    calls = []

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        calls.append(action_name)
        get_bbox(args["marker_id"])
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    result = nodes.action_node({
        "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "Senior iOS Developer"}],
        "current_url": "https://www.wanted.co.kr/search?query=iOS",
        "current_url_stale": False,
        "reflex_state_key": "state-list",
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": ["open", "collect"],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[
                {"name": "click_marker", "args": {"marker_id": 1}, "id": "1"},
                {"name": "update_plan_progress", "args": {"current_step": 1}, "id": "2"},
            ],
        ),
    })

    assert calls == ["click_marker"]
    assert [action["status"] for action in result["action_history"]] == ["success", "success"]
    assert result["current_plan_step"] == 1


def test_close_browser_closes_visible_browser_window(monkeypatch):
    from agent.tools import actions
    from agent.tools.actions import ActionTools

    calls = []

    class FakeWindow:
        title = "Wanted - Google Chrome"
        isMinimized = False
        width = 1200
        height = 800

        def activate(self):
            calls.append("activate")

        def close(self):
            calls.append("close")

    fake_window = FakeWindow()

    class FakeGW:
        def getActiveWindow(self):
            return None

        def getAllWindows(self):
            return [fake_window]

    monkeypatch.setattr(actions, "gw", FakeGW())

    action_tools = object.__new__(ActionTools)
    result = action_tools.close_browser()

    assert result["status"] == "success"
    assert result["result"] == {"closed": True, "title": "Wanted - Google Chrome"}
    assert calls == ["activate", "close"]


def test_action_node_executes_close_browser(monkeypatch):
    from langchain_core.messages import AIMessage
    from agent.graph import nodes

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        assert action_name == "close_browser"
        assert args == {}
        return {"status": "success", "action": "close_browser", "result": {"closed": True}}

    monkeypatch.setattr(nodes, "_dispatch_ui", fake_dispatch_ui)

    result = nodes.action_node({
        "current_markers": [],
        "current_url": "https://www.wanted.co.kr",
        "current_url_stale": False,
        "reflex_state_key": "state-home",
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "current_plan_step": 0,
        "plan": [],
        "last_action_result": AIMessage(
            content="",
            tool_calls=[{"name": "close_browser", "args": {}, "id": "1"}],
        ),
    })

    action = result["action_history"][0]
    assert action["status"] == "success"
    assert action["action"] == "close_browser"
    assert result["last_action_screen_changed"] is True
    assert result["current_url_stale"] is True

def test_open_browser_uses_new_window_when_no_browser_is_bound(monkeypatch):
    from pathlib import Path

    from agent.tools import actions
    from agent.tools.actions import ActionTools

    launched = []
    bound_calls = []

    class FakePerception:
        _browser_window_id = None

    def fake_popen(args, stdout=None, stderr=None):
        launched.append(args)
        return object()

    action_tools = object.__new__(ActionTools)
    action_tools.perception = FakePerception()
    action_tools.clipboard_delay_sec = 0

    monkeypatch.setattr(action_tools, "_browser_window_ids", lambda: set())
    browser_exe = Path("C:/Chrome/chrome.exe")
    monkeypatch.setattr(action_tools, "_browser_executable", lambda: browser_exe)
    monkeypatch.setattr(action_tools, "_sleep", lambda seconds: None)
    monkeypatch.setattr(action_tools, "_bind_new_or_active_browser_window", lambda before_ids: bound_calls.append(before_ids) or True)
    monkeypatch.setattr(actions.subprocess, "Popen", fake_popen)

    result = action_tools.open_browser("https://www.wanted.co.kr", current_url="")

    assert result["status"] == "success"
    assert result["result"]["opened"] is True
    assert result["result"]["reason"] == "new_browser_window"
    assert launched == [[str(browser_exe), "--new-window", "https://www.wanted.co.kr"]]
    assert bound_calls == [set()]


def test_open_browser_window_size_is_opt_in(monkeypatch):
    from pathlib import Path

    from agent.tools import actions
    from agent.tools.actions import ActionTools

    launched = []

    class FakePerception:
        _browser_window_id = None

    def fake_popen(args, stdout=None, stderr=None):
        launched.append(args)
        return object()

    action_tools = object.__new__(ActionTools)
    action_tools.perception = FakePerception()
    monkeypatch.setenv("VISION_BROWSER_WINDOW_SIZE", "1")
    monkeypatch.setattr(action_tools, "_browser_window_ids", lambda: set())
    monkeypatch.setattr(action_tools, "_browser_executable", lambda: Path("C:/Chrome/chrome.exe"))
    monkeypatch.setattr(action_tools, "_sleep", lambda seconds: None)
    monkeypatch.setattr(action_tools, "_bind_new_or_active_browser_window", lambda before_ids: True)
    monkeypatch.setattr(actions.subprocess, "Popen", fake_popen)

    result = action_tools.open_browser("https://www.wanted.co.kr", current_url="")

    assert result["status"] == "success"
    assert launched == [[str(Path("C:/Chrome/chrome.exe")), "--new-window", "--window-size=1976,2129", "https://www.wanted.co.kr"]]


def test_open_browser_navigates_bound_window_instead_of_opening_another(monkeypatch):
    from agent.tools import actions
    from agent.tools.actions import ActionTools

    calls = []

    class FakePerception:
        _browser_window_id = 10

        def _get_browser_region(self):
            calls.append("region")
            return {"left": 0, "top": 0, "width": 1000, "height": 800}

    class FakePyAutoGUI:
        PAUSE = 0.1

        def hotkey(self, *keys):
            calls.append(("hotkey", keys))

        def press(self, key):
            calls.append(("press", key))

    fake_pyautogui = FakePyAutoGUI()
    copied = []
    action_tools = object.__new__(ActionTools)
    action_tools.perception = FakePerception()
    action_tools.clipboard_delay_sec = 0

    monkeypatch.setattr(action_tools, "_browser_window_ids", lambda: {10})
    monkeypatch.setattr(action_tools, "_open_url_in_new_window", lambda url: (_ for _ in ()).throw(AssertionError("should reuse bound window")))
    monkeypatch.setattr(action_tools, "_sleep", lambda seconds: None)
    monkeypatch.setattr(actions, "pyautogui", fake_pyautogui)
    monkeypatch.setattr(actions.pyperclip, "copy", lambda text: copied.append(text))

    result = action_tools.open_browser("https://www.wanted.co.kr/search?query=ai", current_url="https://www.wanted.co.kr")

    assert result["status"] == "success"
    assert result["result"] == {
        "opened": True,
        "url": "https://www.wanted.co.kr/search?query=ai",
        "reason": "dedicated_browser_navigated",
    }
    assert copied == ["https://www.wanted.co.kr/search?query=ai"]
    assert calls == ["region", ("hotkey", ("ctrl", "l")), ("hotkey", ("ctrl", "v")), ("press", "enter")]


def test_perception_prefers_bound_browser_window(monkeypatch):
    from agent.tools import perception
    from agent.tools.perception import PerceptionEngine

    class FakeWindow:
        def __init__(self, hwnd, title):
            self._hWnd = hwnd
            self.title = title
            self.visible = True
            self.isMinimized = False
            self.width = 1200
            self.height = 800
            self.top = 10
            self.left = 20
            self.isMaximized = False

    unrelated = FakeWindow(1, "Spreadsheet - Google Chrome")
    wanted = FakeWindow(2, "Wanted - Google Chrome")

    class FakeGW:
        def getAllWindows(self):
            return [unrelated, wanted]

        def getActiveWindow(self):
            return unrelated

    monkeypatch.setattr(perception, "gw", FakeGW())
    engine = object.__new__(PerceptionEngine)
    engine._browser_window_id = 2
    engine.last_region = None

    assert engine._find_browser_window() is wanted

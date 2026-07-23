from agent.graph import worker_execution, worker_reasoning, worker_recording


def _action_request(*, content="", tool_calls=None, source="llm"):
    from agent.graph.action_request import build_action_request

    return build_action_request(source, str(content or ""), list(tool_calls or []))


def test_action_node_uses_same_executor_for_all_action_sources(monkeypatch):

    executed = []

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        executed.append((action_name, args["direction"]))
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)

    for source in ("llm", "reflex", "card_queue"):
        result = worker_execution.action_node(
            {
                "current_url": "https://www.wanted.co.kr/search",
                "current_url_stale": False,
                "action_history": [],
                "pending_action": _action_request(
                    source=source,
                    tool_calls=[
                        {
                            "name": "scroll",
                            "args": {"direction": "down"},
                            "id": f"{source}-scroll",
                        }
                    ],
                ),
            }
        )

        assert result["last_action_result"].source == source
        assert result["action_history"][0]["action_source"] == source

    assert executed == [("scroll", "down")] * 3


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
    monkeypatch.setattr(action_tools, "_reset_browser_zoom", lambda: None)

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
    monkeypatch.setattr(action_tools, "_reset_browser_zoom", lambda: None)

    result = action_tools.open_browser("https://www.wanted.co.kr", current_url="")

    assert result["status"] == "success"
    assert opened == ["https://www.wanted.co.kr"]
    assert "visual_ready" not in result["result"]

def test_action_node_handles_open_browser_without_screen_change(monkeypatch):

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

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)

    result = worker_execution.action_node({
        "current_markers": [],
        "current_url": "https://www.wanted.co.kr",
        "current_url_stale": False,
        "ui_context": "already captured",
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "pending_action": _action_request(
            content="",
            tool_calls=[{"name": "open_browser", "args": {"url": "https://www.wanted.co.kr"}, "id": "1"}],
        ),
    })

    assert result["last_action_result"].screen_changed is False
    assert result["current_url_stale"] is False
    assert result["current_url"] == "https://www.wanted.co.kr"


def test_action_node_records_target_metadata(monkeypatch):

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        assert get_bbox(args["marker_id"]) == [10, 20, 110, 80]
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)

    result = worker_execution.action_node({
        "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "Data Scientist"}],
        "current_url": "https://www.wanted.co.kr/search?query=data",
        "current_url_stale": False,
        "marked_image": "marked.jpg",
        "recent_images": ["screen.jpg"],
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "pending_action": _action_request(
            content="",
            tool_calls=[{"name": "click_marker", "args": {"marker_id": 1}, "id": "1"}],
        ),
    })

    action = result["action_history"][0]
    assert "state_key" not in action
    assert action["before_url"] == "https://www.wanted.co.kr/search?query=data"
    assert action["before_screenshot"] == "screen.jpg"
    assert action["before_marked_image"] == "marked.jpg"
    assert action["target"] == {
        "marker_id": 1,
        "text": "Data Scientist",
        "bbox": [10, 20, 110, 80],
        "center": [60, 50],
    }
    recorded = worker_recording.record_execution_node(result)
    episode = recorded["feedback_episodes"][0]
    assert episode["proposal"]["action"] == "click_marker"
    assert episode["proposal"]["target"]["text"] == "Data Scientist"
    assert "state_key" not in episode["observation"]["before"]
    assert episode["feedback"]["label"] == "partial"


def test_action_node_uses_action_history_seq_for_recorded_steps(monkeypatch):

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        assert action_name == "click_marker"
        assert get_bbox(args["marker_id"]) == [10, 20, 110, 80]
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)

    result = worker_execution.action_node({
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
        "pending_action": _action_request(
            content="",
            tool_calls=[
                {
                    "name": "update_extracted_info",
                    "args": {"data_json": '{"메모":"목록 확인"}'},
                    "id": "1",
                },
                {"name": "click_marker", "args": {"marker_id": 1, "reason": "open first result"}, "id": "2"},
            ],
        ),
    })

    recorded = worker_recording.record_execution_node(
        {
            **result,
            "recorded_steps": [{"seq": 0, "action": "open_browser"}],
        }
    )
    assert [episode["seq"] for episode in recorded["feedback_episodes"]] == [2, 3]
    assert recorded["recorded_steps"][0]["seq"] == 3
    assert recorded["recorded_steps"][0]["intent"] == "open first result"


def test_action_node_carries_reflex_transition_contract_to_next_perception(monkeypatch):

    monkeypatch.setattr(
        worker_execution,
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
    result = worker_execution.action_node(
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
            "recipe_params": {"query": "android 개발자"},
            "reflex_trace": {
                "hit": True,
                "recipe_key": "recipe-home",
                "tool_calls": {
                    "reflex-call": {
                        "seq": 0,
                        "action": "click_marker",
                        "match_mode": "roi_phash",
                        "marker_id": 1,
                    }
                },
            },
            "reflex_transition_contracts": {"reflex-call": contract},
            "pending_action": _action_request(
                content="",
                tool_calls=[{"name": "click_marker", "args": {"marker_id": 1}, "id": "reflex-call"}],
                source="reflex",
            ),
        }
    )

    pending = result["pending_transition"]
    assert pending["action_seq"] == 0
    assert pending["source"] == "reflex"
    assert pending["contract"] == contract
    assert pending["params"]["query"] == "android 개발자"
    assert result["action_history"][0]["reflex_recipe_key"] == "recipe-home"
    assert result["action_history"][0]["reflex_match"]["match_mode"] == "roi_phash"


def test_action_node_does_not_block_repeated_action_by_state_key(monkeypatch):

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)

    result = worker_execution.action_node({
        "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "Data Scientist"}],
        "current_url": "https://www.wanted.co.kr/search?query=data",
        "current_url_stale": False,
        "action_history": [
            {
                "status": "success",
                "action": "click_marker",
                "args": {"marker_id": 1},
            }
        ],
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "pending_action": _action_request(
            content="",
            tool_calls=[{"name": "click_marker", "args": {"marker_id": 1}, "id": "1"}],
        ),
    })

    action = result["action_history"][0]
    assert action["status"] == "success"
    assert action["target"]["text"] == "Data Scientist"
    assert result["error_count"] == 0
    assert result["last_action_result"].screen_changed is True
    assert result["current_url_stale"] is True


def test_action_node_stops_ui_chain_after_screen_boundary_action(monkeypatch):

    calls = []

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        calls.append((action_name, dict(args)))
        get_bbox(args["marker_id"])
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)

    result = worker_execution.action_node({
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
        "pending_action": _action_request(
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
    assert result["last_action_result"].screen_changed is True


def test_action_node_allows_type_then_enter_chain(monkeypatch):

    calls = []

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        calls.append(action_name)
        if action_name == "type_in_marker":
            assert get_bbox(args["marker_id"]) == [10, 20, 110, 80]
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)

    result = worker_execution.action_node({
        "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "검색"}],
        "current_url": "https://www.wanted.co.kr",
        "current_url_stale": False,
        "reflex_state_key": "state-home",
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "pending_action": _action_request(
            content="",
            tool_calls=[
                {"name": "type_in_marker", "args": {"marker_id": 1, "text": "데이터 분석가"}, "id": "1"},
                {
                    "name": "press_key",
                    "args": {"key": "enter"},
                    "id": "2",
                    "metadata": {"transition_source": "reflex_compound"},
                },
            ],
        ),
    })

    assert calls == ["type_in_marker", "press_key"]
    assert [a["status"] for a in result["action_history"]] == ["success", "success"]
    assert result["action_history"][1]["args"] == {"key": "enter"}
    assert result["pending_transition"]["source"] == "reflex_compound"
    assert result["last_action_result"].screen_changed is True


def test_text_input_target_guard_rejects_close_icon_and_accepts_input_container():
    from agent.runtime.action_validation import text_input_target_rejection

    markers = [
        {"id": 0, "bbox": [2350, 218, 2423, 297], "type": "icon"},
        {"id": 1, "bbox": [1398, 317, 2430, 401], "type": "icon"},
        {"id": 2, "bbox": [1451, 341, 1541, 371], "type": "text"},
    ]

    rejection = text_input_target_rejection(markers, 0)

    assert rejection == {
        "reason": "implausible_text_input_target",
        "marker_id": 0,
        "marker_type": "icon",
        "aspect_ratio": 0.924,
    }
    assert text_input_target_rejection(markers, 1) is None
    assert text_input_target_rejection(markers, 2) is None


def test_action_node_rejects_type_on_compact_icon_before_physical_input(monkeypatch):

    def unexpected_dispatch(*args, **kwargs):
        raise AssertionError("거절된 입력 대상에는 물리 입력을 실행하면 안 됩니다.")

    monkeypatch.setattr(worker_execution, "_dispatch_ui", unexpected_dispatch)

    result = worker_execution.action_node(
        {
            "current_markers": [
                {"id": 0, "bbox": [2350, 218, 2423, 297], "type": "icon", "text": "닫기"},
                {"id": 1, "bbox": [1398, 317, 2430, 401], "type": "icon", "text": "입력 영역"},
                {"id": 2, "bbox": [1451, 341, 1541, 371], "type": "text", "text": "검색어"},
            ],
            "current_url": "https://www.wanted.co.kr",
            "current_url_stale": False,
            "extracted_jd": {},
            "is_finished": False,
            "collected_data": [],
            "error_count": 0,
            "pending_action": _action_request(
                content="",
                tool_calls=[
                    {
                        "name": "type_in_marker",
                        "args": {"marker_id": 0, "text": "ios 개발자", "target_role": "search_input"},
                        "id": "1",
                    },
                    {"name": "press_key", "args": {"key": "enter"}, "id": "2"},
                ],
            ),
        }
    )

    assert len(result["action_history"]) == 1
    assert result["action_history"][0]["status"] == "error"
    assert result["action_history"][0]["reason"] == "implausible_text_input_target"
    assert result["last_action_result"].screen_changed is False
    assert result["error_count"] == 1


def test_reasoning_prompt_lists_forbidden_same_screen_actions():

    messages = worker_reasoning._build_reasoning_messages({
        "goal": "collect jobs",
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

    messages = worker_reasoning._build_reasoning_messages(
        {
            "goal": "collect two jobs",
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

    monkeypatch.setenv("VISION_REASONING_ACTION_HISTORY_LIMIT", "2")
    large_unused_text = "UNRELATED_FULL_TEXT_" + ("x" * 800)
    messages = worker_reasoning._build_reasoning_messages(
        {
            "goal": "collect jobs",
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
    assert "수집 데이터 요약" in human_text
    assert "Current Co" in human_text
    assert "Current Job" in human_text
    assert "누락필드" in human_text
    assert "UNRELATED_FULL_TEXT" not in human_text
    assert "최근 행동 요약" in human_text
    assert "Old hidden action" not in human_text
    assert "Recent card" in human_text
    assert "이전 행동 내역" not in human_text


def test_action_node_allows_repeat_after_navigation_without_state_key_guard(monkeypatch):

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)

    result = worker_execution.action_node({
        "current_markers": [{"id": 41, "bbox": [10, 20, 110, 80], "text": "Job card"}],
        "current_url": "https://www.wanted.co.kr/search?query=iOS",
        "current_url_stale": False,
        "action_history": [
            {
                "status": "success",
                "action": "click_marker",
                "args": {"marker_id": 41},
            },
            {
                "status": "success",
                "action": "scroll",
                "args": {"direction": "down"},
            },
            {
                "status": "success",
                "action": "go_back",
                "args": {},
            },
        ],
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "pending_action": _action_request(
            content="[reflex] cached 1 action(s)",
            tool_calls=[{"name": "click_marker", "args": {"marker_id": 41}, "id": "1"}],
        ),
    })

    action = result["action_history"][0]
    assert action["status"] == "success"
    assert result["error_count"] == 0
    assert result["last_action_result"].screen_changed is True
    assert result["current_url_stale"] is True

def test_action_node_records_policy_go_back_step(monkeypatch):

    recorded_actions = []

    def fake_dispatch_state(action_name, _args, _jd, **_kwargs):
        return (
            {"action": action_name, "status": "success", "_detail_ocr_buffer": {}},
            {"공고목록": [{"position": "iOS 개발자"}]},
        )

    def fake_dispatch_ui(action_name, _args, _get_bbox, current_url=""):
        return {"status": "success", "action": action_name, "result": current_url}

    def fake_record_ui_step(_steps, _state, action_name, _args, _seq):
        recorded_actions.append(action_name)
        _steps.append({"seq": _seq, "action": action_name})

    monkeypatch.setattr(worker_execution, "_dispatch_state", fake_dispatch_state)
    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)
    monkeypatch.setattr(worker_recording, "record_ui_step", fake_record_ui_step)

    result = worker_execution.action_node(
        {
            "current_url": "https://www.wanted.co.kr/wd/1",
            "current_url_stale": False,
            "current_markers": [],
            "recipe_params": {"target_count": 2},
            "result_card_queue": [{"queue_id": "card-2", "status": "pending"}],
            "active_result_card": {"queue_id": "card-1"},
            "extracted_jd": {},
            "detail_ocr_buffer": {},
            "action_history": [],
            "collected_data": [],
            "error_count": 0,
            "is_finished": False,
            "pending_action": _action_request(
                content="[page_policy] detail finish",
                tool_calls=[
                    {
                        "name": "finish_detail_reading",
                        "args": {"page_role": "job_detail", "detail_complete": True},
                        "id": "detail-finish",
                    }
                ],
            ),
        }
    )

    assert [action["action"] for action in result["action_history"]] == ["finish_detail_reading"]
    assert result["pending_action"].source == "page_policy"

    followup_state = {
        **result,
        "goal": "iOS 개발자 공고 2개",
        "current_url": "https://www.wanted.co.kr/wd/1",
        "current_url_stale": False,
        "current_markers": [],
        "action_history": result["action_history"],
        "recorded_steps": [],
    }
    followup = worker_execution.action_node(followup_state)
    recorded = worker_recording.record_execution_node({**followup_state, **followup})

    assert followup["action_history"][0]["action_source"] == "page_policy"
    assert followup["pending_transition"]["source"] == "page_policy"
    assert recorded_actions == ["go_back"]
    assert recorded["recorded_steps"][0]["action"] == "go_back"


def test_detail_completion_does_not_repeat_failed_return_action(monkeypatch):

    def fake_dispatch_state(action_name, _args, _jd, **_kwargs):
        return (
            {"action": action_name, "status": "success", "_detail_ocr_buffer": {}},
            {"공고목록": [{"position": "iOS 개발자"}]},
        )

    monkeypatch.setattr(worker_execution, "_dispatch_state", fake_dispatch_state)
    monkeypatch.setattr(
        worker_execution,
        "_dispatch_ui",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("실패한 복귀 행동을 자동 반복하면 안 됩니다.")
        ),
    )

    result = worker_execution.action_node(
        {
            "current_url": "https://www.jobkorea.co.kr/Recruit/GI_Read/1",
            "current_url_stale": False,
            "current_markers": [],
            "recent_images": ["same-detail.png"],
            "transition_status": "unknown",
            "transition_observations": [
                {
                    "action": "go_back",
                    "status": "unknown",
                    "reason": "no_screen_change",
                    "screenshot": "same-detail.png",
                }
            ],
            "recipe_params": {"target_count": 2},
            "result_card_queue": [
                {"queue_id": "card-1", "status": "active"},
                {"queue_id": "card-2", "status": "pending"},
            ],
            "active_result_card": {"queue_id": "card-1"},
            "extracted_jd": {},
            "detail_ocr_buffer": {},
            "action_history": [],
            "collected_data": [],
            "error_count": 0,
            "is_finished": False,
            "pending_action": _action_request(
                content="",
                tool_calls=[
                    {
                        "name": "finish_detail_reading",
                        "args": {"page_role": "job_detail", "detail_complete": True},
                        "id": "detail-finish-after-failed-back",
                    }
                ],
            ),
        }
    )

    assert [action["action"] for action in result["action_history"]] == [
        "finish_detail_reading"
    ]
    assert result["action_history"][0]["detail_policy"] == "return_requires_reasoning"
    assert result["action_history"][0]["failed_return_action"] == "go_back"
    assert result["last_action_result"].screen_changed is False


def test_action_node_finishes_after_last_visible_result_card(monkeypatch):

    def fake_dispatch_state(action_name, _args, _jd, **_kwargs):
        return (
            {"action": action_name, "status": "success", "_detail_ocr_buffer": {}},
            {"공고목록": [{"position": "데이터 엔지니어"}]},
        )

    monkeypatch.setattr(worker_execution, "_dispatch_state", fake_dispatch_state)

    result = worker_execution.action_node(
        {
            "current_url": "https://www.wanted.co.kr/wd/1",
            "current_url_stale": False,
            "current_markers": [],
            "recipe_params": {"target_count": 0, "count_mode": "visible_all"},
            "result_card_queue": [{"queue_id": "card-1", "status": "active"}],
            "active_result_card": {"queue_id": "card-1"},
            "extracted_jd": {},
            "detail_ocr_buffer": {},
            "action_history": [],
            "collected_data": [],
            "error_count": 0,
            "is_finished": False,
            "pending_action": _action_request(
                content="[page_policy] detail finish",
                tool_calls=[
                    {
                        "name": "finish_detail_reading",
                        "args": {"page_role": "job_detail", "detail_complete": True},
                        "id": "detail-finish",
                    }
                ],
            ),
        }
    )

    assert result["is_finished"] is True
    assert [action["action"] for action in result["action_history"]] == ["finish_detail_reading"]
    assert result["result_card_queue"][0]["status"] == "done"


def test_action_node_finishes_visible_all_enum_after_last_card(monkeypatch):
    from shared.schema.collection_intent import CollectionCountMode

    def fake_dispatch_state(action_name, _args, _jd, **_kwargs):
        return (
            {"action": action_name, "status": "success", "_detail_ocr_buffer": {}},
            {"공고목록": [{"position": "데이터 엔지니어"}]},
        )

    monkeypatch.setattr(worker_execution, "_dispatch_state", fake_dispatch_state)

    result = worker_execution.action_node(
        {
            "current_url": "https://www.wanted.co.kr/wd/1",
            "current_markers": [],
            "recipe_params": {
                "target_count": 0,
                "count_mode": CollectionCountMode.VISIBLE_ALL,
            },
            "result_card_queue": [{"queue_id": "card-1", "status": "active"}],
            "active_result_card": {"queue_id": "card-1"},
            "extracted_jd": {},
            "detail_ocr_buffer": {},
            "action_history": [],
            "collected_data": [],
            "error_count": 0,
            "is_finished": False,
            "pending_action": _action_request(
                content="[page_policy] detail finish",
                tool_calls=[
                    {
                        "name": "finish_detail_reading",
                        "args": {"page_role": "job_detail", "detail_complete": True},
                        "id": "detail-finish",
                    }
                ],
            ),
        }
    )

    assert result["is_finished"] is True


def test_action_node_allows_same_text_when_marker_id_changes_without_state_key_guard(monkeypatch):

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)

    result = worker_execution.action_node({
        "current_markers": [{"id": 42, "bbox": [10, 20, 110, 80], "text": "Job card"}],
        "current_url": "https://www.wanted.co.kr/search?query=iOS",
        "current_url_stale": False,
        "action_history": [
            {
                "status": "success",
                "action": "click_marker",
                "args": {"marker_id": 41},
                "target": {"marker_id": 41, "text": "Job card"},
            }
        ],
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "pending_action": _action_request(
            content="[reflex] cached 1 action(s)",
            tool_calls=[{"name": "click_marker", "args": {"marker_id": 42}, "id": "1"}],
        ),
    })

    action = result["action_history"][0]
    assert action["status"] == "success"
    assert result["error_count"] == 0
    assert result["last_action_result"].screen_changed is True
    assert result["current_url_stale"] is True

def test_action_node_stops_before_state_update_after_screen_boundary(monkeypatch):

    calls = []

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        calls.append(action_name)
        get_bbox(args["marker_id"])
        return {"status": "success", "action": action_name, "result": "ok"}

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)

    result = worker_execution.action_node({
        "current_markers": [{"id": 1, "bbox": [10, 20, 110, 80], "text": "Senior iOS Developer"}],
        "current_url": "https://www.wanted.co.kr/search?query=iOS",
        "current_url_stale": False,
        "reflex_state_key": "state-list",
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "pending_action": _action_request(
            content="",
            tool_calls=[
                {"name": "click_marker", "args": {"marker_id": 1}, "id": "1"},
                {
                    "name": "update_extracted_info",
                    "args": {"data_json": '{"메모":"새 화면 정보"}'},
                    "id": "2",
                },
            ],
        ),
    })

    assert calls == ["click_marker"]
    assert [action["status"] for action in result["action_history"]] == ["success", "skipped"]
    assert result["action_history"][1]["reason"] == "chain_boundary_after_screen_change"
    assert result["extracted_jd"] == {}


def test_close_browser_closes_visible_browser_window(monkeypatch):
    from agent.tools import actions
    from agent.tools.actions import ActionTools

    calls = []

    class FakeWindow:
        _hWnd = 20
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
    action_tools.perception = type(
        "FakePerception",
        (),
        {
            "_browser_window_id": 20,
            "clear_browser_window": lambda self: setattr(self, "_browser_window_id", None),
        },
    )()
    result = action_tools.close_browser()

    assert result["status"] == "success"
    assert result["result"] == {"closed": True, "title": "Wanted - Google Chrome"}
    assert calls == ["activate", "close"]


def test_action_node_executes_close_browser(monkeypatch):

    def fake_dispatch_ui(action_name, args, get_bbox, current_url=""):
        assert action_name == "close_browser"
        assert args == {}
        return {"status": "success", "action": "close_browser", "result": {"closed": True}}

    monkeypatch.setattr(worker_execution, "_dispatch_ui", fake_dispatch_ui)

    result = worker_execution.action_node({
        "current_markers": [],
        "current_url": "https://www.wanted.co.kr",
        "current_url_stale": False,
        "reflex_state_key": "state-home",
        "extracted_jd": {},
        "is_finished": False,
        "collected_data": [],
        "error_count": 0,
        "pending_action": _action_request(
            content="",
            tool_calls=[{"name": "close_browser", "args": {}, "id": "1"}],
        ),
    })

    action = result["action_history"][0]
    assert action["status"] == "success"
    assert result["last_action_result"].source == "llm"
    assert result["last_action_result"].status == "success"
    assert result["last_action_result"].tool_results == result["action_history"]
    assert action["action"] == "close_browser"
    assert result["last_action_result"].screen_changed is True
    assert result["current_url_stale"] is True

def test_open_browser_uses_new_window_when_no_browser_is_bound(monkeypatch):
    from pathlib import Path

    from agent.tools import actions
    from agent.tools.actions import ActionTools

    launched = []
    bound_calls = []
    zoom_reset = []

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
    monkeypatch.setattr(action_tools, "_browser_profile_dir", lambda: Path("C:/L2C/browser-profile"))
    monkeypatch.setattr(action_tools, "_sleep", lambda seconds: None)
    monkeypatch.setattr(action_tools, "_bind_new_or_active_browser_window", lambda before_ids: bound_calls.append(before_ids) or True)
    monkeypatch.setattr(action_tools, "_reset_browser_zoom", lambda: zoom_reset.append(True))
    monkeypatch.setattr(actions.subprocess, "Popen", fake_popen)

    result = action_tools.open_browser("https://www.wanted.co.kr", current_url="")

    assert result["status"] == "success"
    assert result["result"]["opened"] is True
    assert result["result"]["reason"] == "new_browser_window"
    assert launched == [[
        str(browser_exe),
        "--new-window",
        "--user-data-dir=C:\\L2C\\browser-profile",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-session-crashed-bubble",
        "--window-size=1976,2129",
        "https://www.wanted.co.kr",
    ]]
    assert bound_calls == [set()]
    assert zoom_reset == [True]


def test_open_browser_window_size_can_be_disabled(monkeypatch):
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
    monkeypatch.setenv("VISION_BROWSER_WINDOW_SIZE", "0")
    monkeypatch.setattr(action_tools, "_browser_window_ids", lambda: set())
    monkeypatch.setattr(action_tools, "_browser_executable", lambda: Path("C:/Chrome/chrome.exe"))
    monkeypatch.setattr(action_tools, "_browser_profile_dir", lambda: Path("C:/L2C/browser-profile"))
    monkeypatch.setattr(action_tools, "_sleep", lambda seconds: None)
    monkeypatch.setattr(action_tools, "_bind_new_or_active_browser_window", lambda before_ids: True)
    monkeypatch.setattr(action_tools, "_reset_browser_zoom", lambda: None)
    monkeypatch.setattr(actions.subprocess, "Popen", fake_popen)

    result = action_tools.open_browser("https://www.wanted.co.kr", current_url="")

    assert result["status"] == "success"
    assert launched == [[
        str(Path("C:/Chrome/chrome.exe")),
        "--new-window",
        "--user-data-dir=C:\\L2C\\browser-profile",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-session-crashed-bubble",
        "https://www.wanted.co.kr",
    ]]


def test_close_browser_prefers_bound_agent_window(monkeypatch):
    from agent.tools import actions
    from agent.tools.actions import ActionTools

    calls = []

    class FakeWindow:
        visible = True
        isMinimized = False
        width = 1200
        height = 800

        def __init__(self, window_id, title):
            self._hWnd = window_id
            self.title = title

        def activate(self):
            calls.append(("activate", self._hWnd))

        def close(self):
            calls.append(("close", self._hWnd))

    user_window = FakeWindow(10, "개인 문서 - Google Chrome")
    agent_window = FakeWindow(20, "Wanted - Google Chrome")

    class FakeGW:
        def getActiveWindow(self):
            return user_window

        def getAllWindows(self):
            return [user_window, agent_window]

    class FakePerception:
        _browser_window_id = 20

        @staticmethod
        def _window_id(window):
            return window._hWnd

        @staticmethod
        def _looks_like_browser_window(window):
            return "Chrome" in window.title

        @staticmethod
        def _is_visible_window(window):
            return window.visible and not window.isMinimized

        def clear_browser_window(self):
            self._browser_window_id = None

    monkeypatch.setattr(actions, "gw", FakeGW())
    action_tools = object.__new__(ActionTools)
    action_tools.perception = FakePerception()

    result = action_tools.close_browser()

    assert result["status"] == "success"
    assert result["result"]["title"] == "Wanted - Google Chrome"
    assert calls == [("activate", 20), ("close", 20)]
    assert action_tools.perception._browser_window_id is None


def test_normalize_browser_window_restores_and_resizes(monkeypatch):
    from agent.tools.actions import ActionTools

    calls = []

    class FakeWindow:
        isMaximized = True
        width = 1976
        height = 2129

        def restore(self):
            calls.append("restore")

        def resizeTo(self, width, height):
            calls.append(("resize", width, height))

    action_tools = object.__new__(ActionTools)
    monkeypatch.setattr(action_tools, "_sleep", lambda seconds: calls.append(("sleep", seconds)))

    assert action_tools._normalize_browser_window(FakeWindow()) is True
    assert calls[0] == "restore"
    assert calls[1] == ("resize", 1976, 2129)


def test_reasoning_screen_guard_detects_changed_screen(tmp_path, monkeypatch):
    from PIL import Image, ImageDraw

    from agent.runtime.action_guard import check_reasoning_screen_stale
    from agent.vision.screen_signature import perceptual_hash

    before = tmp_path / "before.png"
    before_image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(before_image).rectangle([0, 0, 96, 256], fill="black")
    before_image.save(before)
    after_image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(after_image).ellipse([100, 20, 240, 160], fill="black")

    class FakePerception:
        def capture_screen(self, **kwargs):
            assert kwargs["initial_wait_sec"] == 0
            assert kwargs["wait_for_stable"] is False
            temporary = tmp_path / kwargs["filename"]
            after_image.save(temporary)
            return temporary

    monkeypatch.setenv("VISION_REASONING_STALE_PHASH_MAX_DISTANCE", "10")
    result = check_reasoning_screen_stale(
        {"screen_signature": {"phash": perceptual_hash(before)}},
        FakePerception(),
    )

    assert result["checked"] is True
    assert result["stale"] is True
    assert result["reason"] == "screen_changed_during_reasoning"
    assert result["distance"] > result["max_distance"]
    assert not list(tmp_path.glob("pre_action_*.png"))


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


def test_tab_actions_use_bounded_browser_hotkeys(monkeypatch):
    from agent.tools import actions
    from agent.tools.actions import ActionTools

    calls = []

    class FakePerception:
        def _get_browser_region(self):
            calls.append("region")
            return {"left": 0, "top": 0, "width": 100, "height": 100}

        def release_address_bar_focus(self, key_pause=0.02):
            calls.append(("release_focus", key_pause))

    class FakePyAutoGUI:
        def hotkey(self, *keys):
            calls.append(("hotkey", keys))

    monkeypatch.setattr(actions, "pyautogui", FakePyAutoGUI())
    monkeypatch.setattr(actions.platform, "system", lambda: "Windows")

    action_tools = object.__new__(ActionTools)
    action_tools.perception = FakePerception()

    assert action_tools.close_current_tab()["status"] == "success"
    assert action_tools.switch_tab("next")["status"] == "success"
    assert action_tools.switch_tab("previous")["status"] == "success"
    assert calls == [
        "region",
        ("release_focus", 0.02),
        ("hotkey", ("ctrl", "w")),
        "region",
        ("release_focus", 0.02),
        ("hotkey", ("ctrl", "tab")),
        "region",
        ("release_focus", 0.02),
        ("hotkey", ("ctrl", "shift", "tab")),
    ]


def test_targeted_scroll_moves_to_marker_and_uses_wheel(monkeypatch):
    from agent.tools import actions
    from agent.tools.actions import ActionTools

    calls = []

    class FakePerception:
        last_region = {"left": 10, "top": 20, "width": 1000, "height": 800}
        scale_x = 1.0
        scale_y = 1.0

    class FakePyAutoGUI:
        def moveTo(self, x, y, duration=0):
            calls.append(("move", x, y, duration))

        def scroll(self, steps):
            calls.append(("scroll", steps))

        def hscroll(self, steps):
            calls.append(("hscroll", steps))

        def keyDown(self, key):
            calls.append(("key_down", key))

        def keyUp(self, key):
            calls.append(("key_up", key))

    monkeypatch.setattr(actions, "pyautogui", FakePyAutoGUI())
    monkeypatch.setattr(actions.platform, "system", lambda: "Windows")
    action_tools = object.__new__(ActionTools)
    action_tools.perception = FakePerception()

    down = action_tools.scroll("down", bbox=[100, 200, 300, 400], amount="small")
    right = action_tools.scroll("right", bbox=[100, 200, 300, 400], amount="page")

    assert down["status"] == "success"
    assert right["status"] == "success"
    assert calls == [
        ("move", 210, 320, 0.05),
        ("scroll", -360),
        ("move", 210, 320, 0.05),
        ("key_down", "shift"),
        ("scroll", -960),
        ("key_up", "shift"),
    ]


def test_action_node_blocks_repeating_no_effect_navigation(monkeypatch):

    monkeypatch.setattr(
        worker_execution,
        "_dispatch_ui",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("효과 없던 동일 행동을 다시 실행하면 안 됩니다.")
        ),
    )

    result = worker_execution.action_node(
        {
            "current_url": "https://www.jobkorea.co.kr/Recruit/GI_Read/1",
            "current_url_stale": False,
            "current_markers": [],
            "recent_images": ["same-screen.png"],
            "transition_status": "unknown",
            "transition_observations": [
                {
                    "action": "go_back",
                    "status": "unknown",
                    "reason": "reflex_no_screen_change",
                    "screenshot": "same-screen.png",
                }
            ],
            "extracted_jd": {},
            "action_history": [],
            "collected_data": [],
            "error_count": 0,
            "is_finished": False,
            "pending_action": _action_request(
                content="",
                tool_calls=[{"name": "go_back", "args": {}, "id": "repeat-back"}],
            ),
        }
    )

    assert result["action_history"][0]["status"] == "skipped"
    assert result["action_history"][0]["reason"] == "same_screen_no_effect_action_blocked"
    assert result["last_action_result"].screen_changed is False


def test_action_node_allows_different_marker_after_no_effect_click(monkeypatch):

    calls = []
    monkeypatch.setattr(
        worker_execution,
        "_dispatch_ui",
        lambda action_name, args, *_rest, **_kwargs: calls.append(
            (action_name, args["marker_id"])
        )
        or {"action": action_name, "status": "success", "result": "clicked"},
    )
    monkeypatch.setattr(
        worker_execution,
        "_check_current_reasoning_screen",
        lambda _state: {"checked": True, "stale": False},
    )

    result = worker_execution.action_node(
        {
            "current_url": "https://example.com/jobs",
            "current_url_stale": False,
            "current_markers": [
                {"id": 10, "bbox": [10, 10, 100, 40], "text": "첫 링크"},
                {"id": 20, "bbox": [10, 60, 100, 90], "text": "다른 링크"},
            ],
            "recent_images": ["same-screen.png"],
            "transition_status": "unknown",
            "transition_observations": [
                {
                    "action": "click_marker",
                    "step": {"args": {"marker_id": 10}},
                    "status": "unknown",
                    "reason": "no_screen_change",
                    "screenshot": "same-screen.png",
                }
            ],
            "extracted_jd": {},
            "action_history": [],
            "collected_data": [],
            "error_count": 0,
            "is_finished": False,
            "pending_action": _action_request(
                content="",
                tool_calls=[
                    {
                        "name": "click_marker",
                        "args": {"marker_id": 20, "page_role": "search"},
                        "id": "different-marker",
                    }
                ],
            ),
        }
    )

    assert calls == [("click_marker", 20)]
    assert result["action_history"][0]["status"] == "success"


def test_reasoning_prompt_recommends_tab_close_after_back_no_effect():

    messages = worker_reasoning._build_reasoning_messages(
        {
            "goal": "잡코리아 공고 두 개 수집",
            "current_url": "https://www.jobkorea.co.kr/Recruit/GI_Read/1",
            "current_markers": [],
            "recent_images": ["same-screen.png"],
            "marked_image": "",
            "ui_context": "상세 공고",
            "extracted_jd": {},
            "action_history": [],
            "transition_status": "unknown",
            "transition_source": "page_policy",
            "transition_observations": [
                {
                    "action": "go_back",
                    "status": "unknown",
                    "reason": "reflex_no_screen_change",
                    "screenshot": "same-screen.png",
                }
            ],
        },
        "",
    )

    assert "효과가 없었던 행동: go_back" in messages[-1].content
    assert "close_current_tab" in messages[-1].content


def test_transition_cycle_detects_two_distinct_screens_repeating(tmp_path):
    from PIL import Image, ImageDraw

    from agent.runtime.transition_runtime import detect_two_screen_transition_cycle

    screen_a = tmp_path / "screen_a.png"
    screen_b = tmp_path / "screen_b.png"
    image_a = Image.new("RGB", (800, 600), "white")
    image_b = Image.new("RGB", (800, 600), "white")
    ImageDraw.Draw(image_a).rectangle((40, 100, 360, 520), fill="black")
    ImageDraw.Draw(image_b).rectangle((440, 80, 760, 500), fill="black")
    image_a.save(screen_a)
    image_b.save(screen_b)
    observations = [
        {
            "action": "press_key",
            "step": {"args": {"key": "enter"}},
            "screenshot": str(screen_a),
        },
        {
            "action": "press_key",
            "step": {"args": {"key": "esc"}},
            "screenshot": str(screen_b),
        },
        {
            "action": "press_key",
            "step": {"args": {"key": "enter"}},
            "screenshot": str(screen_a),
        },
        {
            "action": "press_key",
            "step": {"args": {"key": "esc"}},
            "screenshot": str(screen_b),
        },
    ]

    result = detect_two_screen_transition_cycle(observations)

    assert result["detected"] is True
    assert result["action_cycle"] == ["press_key:enter", "press_key:esc"]
    assert result["same_screen_distances"] == [0, 0]


def test_transition_cycle_ignores_one_screen_repeated(tmp_path):
    from PIL import Image

    from agent.runtime.transition_runtime import detect_two_screen_transition_cycle

    screen = tmp_path / "same.png"
    Image.new("RGB", (800, 600), "white").save(screen)
    observations = [
        {"action": "press_key", "screenshot": str(screen)}
        for _ in range(4)
    ]

    assert detect_two_screen_transition_cycle(observations) == {"detected": False}

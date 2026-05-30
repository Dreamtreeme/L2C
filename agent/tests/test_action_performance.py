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


def test_open_browser_skips_when_state_url_already_matches():
    from agent.tools.actions import ActionTools

    class FakePerception:
        def _get_browser_region(self):
            raise AssertionError("browser region lookup should be skipped when state URL already matches")

    action_tools = object.__new__(ActionTools)
    action_tools.perception = FakePerception()

    result = action_tools.open_browser(
        "https://www.wanted.co.kr",
        current_url="https://www.wanted.co.kr/search?query=data",
    )

    assert result["status"] == "success"
    assert result["result"] == {
        "opened": False,
        "url": "https://www.wanted.co.kr/search?query=data",
        "reason": "state_url_already_matches",
    }


def test_action_node_does_not_reopen_same_browser_url(monkeypatch):
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
                "reason": "state_url_already_matches",
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

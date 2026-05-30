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

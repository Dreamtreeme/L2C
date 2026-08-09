from agent.tools.perception import PerceptionEngine


class FakeSctImage:
    size = (4, 4)
    width = 4
    height = 4
    bgra = b"\x00\x00\x00\xff" * 16


class FakeSct:
    monitors = [None, {"top": 0, "left": 0, "width": 4, "height": 4}]

    def grab(self, _region):
        return FakeSctImage()


class ReadyLoadingWait:
    def __init__(self):
        self.regions = []

    def wait_until_ready(self, *, region=None):
        self.regions.append(region)
        return {"ready": True, "stable": True, "low_information": False}


class FakeClipboard:
    def __init__(self):
        self.value = ""

    def copy(self, value):
        self.value = value

    def paste(self):
        return self.value


class FakeKeyboard:
    def __init__(self, clipboard, url_on_copy=0):
        self.clipboard = clipboard
        self.url_on_copy = url_on_copy
        self.hotkeys = []
        self.copy_count = 0

    def hotkey(self, *keys):
        self.hotkeys.append(keys)
        if keys[-1] != "c":
            return
        self.copy_count += 1
        if self.copy_count == self.url_on_copy:
            self.clipboard.value = "https://example.com/jobs/1"


def test_capture_screen_saves_bound_browser_region(monkeypatch, tmp_path):
    engine = object.__new__(PerceptionEngine)
    engine.screenshot_dir = tmp_path
    engine.sct = FakeSct()
    engine.scale_x = 1.0
    engine.scale_y = 1.0
    engine.last_region = None
    region = {"top": 0, "left": 0, "width": 4, "height": 4}
    monkeypatch.setattr(engine, "_get_browser_region", lambda: region)

    output = engine.capture_screen(filename="screen.jpg")

    assert output == tmp_path / "screen.jpg"
    assert output.exists()
    assert engine.last_region == region


def test_capture_usable_screen_waits_in_memory_then_saves_once(monkeypatch, tmp_path):
    engine = object.__new__(PerceptionEngine)
    engine.screenshot_dir = tmp_path
    engine.loading_wait = ReadyLoadingWait()
    engine.last_capture_quality = {}
    region = {"top": 0, "left": 0, "width": 4, "height": 4}
    output = tmp_path / "ready.png"
    save_calls = []
    monkeypatch.setattr(engine, "_get_browser_region", lambda: region)
    monkeypatch.setattr(
        engine,
        "_save_capture",
        lambda supplied_region, filename=None: save_calls.append(
            (supplied_region, filename)
        )
        or output,
    )
    monkeypatch.setattr("agent.tools.perception.time.sleep", lambda _seconds: None)

    assert engine.capture_usable_screen(initial_wait_sec=0.7) == output
    assert engine.loading_wait.regions == [region]
    assert save_calls == [(region, None)]
    assert engine.last_capture_quality["ready"] is True


def test_address_bar_url_copy_retries_until_clipboard_changes(monkeypatch):
    engine = object.__new__(PerceptionEngine)
    clipboard = FakeClipboard()
    keyboard = FakeKeyboard(clipboard, url_on_copy=2)
    activations = []
    monkeypatch.setattr(
        engine,
        "_get_browser_region",
        lambda: activations.append(True) or None,
    )
    monkeypatch.setattr("agent.tools.perception.time.sleep", lambda _seconds: None)

    url = engine._copy_address_bar_url(
        keyboard,
        clipboard,
        modifier="ctrl",
        key_pause=0,
        copy_wait=0.01,
        copy_timeout=0,
        max_attempts=2,
    )

    assert url == "https://example.com/jobs/1"
    assert len(activations) == 2


def test_address_bar_url_copy_stops_after_attempt_limit(monkeypatch):
    engine = object.__new__(PerceptionEngine)
    clipboard = FakeClipboard()
    keyboard = FakeKeyboard(clipboard)
    activations = []
    monkeypatch.setattr(
        engine,
        "_get_browser_region",
        lambda: activations.append(True) or None,
    )
    monkeypatch.setattr("agent.tools.perception.time.sleep", lambda _seconds: None)

    url = engine._copy_address_bar_url(
        keyboard,
        clipboard,
        modifier="ctrl",
        key_pause=0,
        copy_wait=0.01,
        copy_timeout=0,
        max_attempts=2,
    )

    assert url == ""
    assert len(activations) == 2
    assert keyboard.copy_count == 2

from PIL import Image


class FakeSctImage:
    size = (4, 4)
    width = 4
    height = 4
    bgra = b"\x00\x00\x00\xff" * 16


class FakeSct:
    monitors = [None, {"top": 0, "left": 0, "width": 4, "height": 4}]

    def grab(self, _region):
        return FakeSctImage()


def test_paddle_worker_emits_phases_and_request_scoped_result(monkeypatch, capsys):
    import io
    import json

    from agent.tools import paddle_ocr_runner as runner

    class FakeOcr:
        def predict(self, _image_path):
            return [
                {
                    "res": {
                        "rec_texts": ["검색"],
                        "rec_scores": [0.9],
                        "rec_boxes": [[0, 0, 10, 10]],
                    }
                }
            ]

    monkeypatch.setattr(runner, "build_ocr", lambda: FakeOcr())
    monkeypatch.setattr(
        runner.sys,
        "stdin",
        io.StringIO(json.dumps({"request_id": "req-1", "image_path": "screen.png"}) + "\n"),
    )

    runner.worker_main()

    output_lines = capsys.readouterr().out.splitlines()
    phases = [
        json.loads(line.removeprefix("__OCR_EVENT__ "))["phase"]
        for line in output_lines
        if line.startswith("__OCR_EVENT__ ")
    ]
    result_line = next(line for line in output_lines if line.startswith("__OCR_JSON_RESULT__ "))
    result = json.loads(result_line.removeprefix("__OCR_JSON_RESULT__ "))

    assert phases == [
        "request_received",
        "inference_started",
        "inference_completed",
        "result_serialized",
    ]
    assert result["request_id"] == "req-1"
    assert result["results"][0]["text"] == "검색"
    assert result["timings"]["inference_sec"] >= 0


def test_paddle_worker_emits_explicit_error_instead_of_empty_result(monkeypatch, capsys):
    import io
    import json

    from agent.tools import paddle_ocr_runner as runner

    class FailingOcr:
        def predict(self, _image_path):
            raise RuntimeError("predictor stopped")

    monkeypatch.setattr(runner, "build_ocr", lambda: FailingOcr())
    monkeypatch.setattr(
        runner.sys,
        "stdin",
        io.StringIO(json.dumps({"request_id": "req-2", "image_path": "screen.png"}) + "\n"),
    )

    runner.worker_main()

    captured = capsys.readouterr()
    output_lines = captured.out.splitlines()
    error_line = next(line for line in output_lines if line.startswith("__OCR_WORKER_ERROR__ "))
    error = json.loads(error_line.removeprefix("__OCR_WORKER_ERROR__ "))

    assert error["request_id"] == "req-2"
    assert error["phase"] == "inference_started"
    assert error["error_type"] == "RuntimeError"
    assert not any(line.startswith("__OCR_JSON_RESULT__ ") for line in output_lines)
    assert "predictor stopped" in captured.err


def test_wait_stable_uses_supplied_region_without_window_lookup():
    from agent.utils.wait_stable import WaitStable

    class FakePerception:
        sct = FakeSct()

        def __init__(self):
            self.region_calls = 0

        def _get_browser_region(self):
            self.region_calls += 1
            return {"top": 0, "left": 0, "width": 4, "height": 4}

    perception = FakePerception()
    wait_stable = WaitStable(perception)

    img = wait_stable._capture_memory_image(
        region={"top": 0, "left": 0, "width": 4, "height": 4},
        sample_width=2,
    )

    assert img.size == (2, 2)
    assert perception.region_calls == 0


def test_opencv_frame_compare_separates_change_ratio_and_stability():
    import numpy as np

    from agent.vision.frame_compare import (
        changed_pixel_ratio,
        mean_difference_percent,
    )

    before = np.zeros((200, 200), dtype=np.uint8)
    after = before.copy()
    after[50:150, 50:150] = 255

    assert changed_pixel_ratio(
        before,
        after,
        intensity_threshold=20,
    ) > 0.2
    assert mean_difference_percent(before, before) == 0.0
    assert mean_difference_percent(before, after) > 20.0


def test_capture_screen_reuses_region_for_wait(monkeypatch, tmp_path):
    from agent.tools.perception import PerceptionEngine

    sleeps = []
    waited_regions = []

    class FakeWaitStable:
        def wait(self, region=None):
            waited_regions.append(region)
            return True

    engine = object.__new__(PerceptionEngine)
    engine.screenshot_dir = tmp_path
    engine.sct = FakeSct()
    engine.scale_x = 1.0
    engine.scale_y = 1.0
    engine.last_region = None
    engine._wait_stable = FakeWaitStable()

    region = {"top": 0, "left": 0, "width": 4, "height": 4}
    region_calls = {"count": 0}

    def fake_get_browser_region():
        region_calls["count"] += 1
        return region

    monkeypatch.setattr(engine, "_get_browser_region", fake_get_browser_region)
    monkeypatch.setattr("agent.tools.perception.time.sleep", lambda sec: sleeps.append(sec))

    output = engine.capture_screen(filename="screen.jpg", initial_wait_sec=0.01)

    assert output == tmp_path / "screen.jpg"
    assert output.exists()
    assert sleeps == [0.01]
    assert waited_regions == [region]
    assert region_calls["count"] == 1


def test_address_bar_url_copy_retries_after_new_tab_focus_delay(monkeypatch):
    from agent.tools.perception import PerceptionEngine

    class FakeClipboard:
        def __init__(self):
            self.value = ""

        def copy(self, value):
            self.value = value

        def paste(self):
            return self.value

    class FakePyAutoGUI:
        def __init__(self, clipboard):
            self.clipboard = clipboard
            self.hotkeys = []
            self.copy_count = 0

        def hotkey(self, *keys):
            self.hotkeys.append(keys)
            if keys[-1] == "c":
                self.copy_count += 1
                if self.copy_count == 2:
                    self.clipboard.value = "https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=1"

    engine = object.__new__(PerceptionEngine)
    clipboard = FakeClipboard()
    pyautogui = FakePyAutoGUI(clipboard)
    activation_count = {"value": 0}

    def activate_browser():
        activation_count["value"] += 1
        return {"left": 0, "top": 0, "width": 100, "height": 100}

    monkeypatch.setattr(engine, "_get_browser_region", activate_browser)
    monkeypatch.setattr("agent.tools.perception.time.sleep", lambda _seconds: None)

    url = engine._copy_address_bar_url(
        pyautogui,
        clipboard,
        modifier="ctrl",
        key_pause=0.05,
        copy_wait=0.01,
        copy_timeout=0,
        max_attempts=2,
    )

    assert url.endswith("rec_idx=1")
    assert activation_count["value"] == 2
    assert pyautogui.hotkeys == [
        ("ctrl", "l"),
        ("ctrl", "c"),
        ("ctrl", "l"),
        ("ctrl", "c"),
    ]


def test_address_bar_url_copy_returns_empty_after_bounded_attempts(monkeypatch):
    from agent.tools.perception import PerceptionEngine

    class FakeClipboard:
        def copy(self, _value):
            pass

        def paste(self):
            return ""

    class FakePyAutoGUI:
        def __init__(self):
            self.hotkeys = []

        def hotkey(self, *keys):
            self.hotkeys.append(keys)

    engine = object.__new__(PerceptionEngine)
    pyautogui = FakePyAutoGUI()
    activation_count = {"value": 0}

    def activate_browser():
        activation_count["value"] += 1
        return None

    monkeypatch.setattr(engine, "_get_browser_region", activate_browser)
    monkeypatch.setattr("agent.tools.perception.time.sleep", lambda _seconds: None)

    url = engine._copy_address_bar_url(
        pyautogui,
        FakeClipboard(),
        modifier="ctrl",
        key_pause=0,
        copy_wait=0.01,
        copy_timeout=0,
        max_attempts=2,
    )

    assert url == ""
    assert activation_count["value"] == 2
    assert pyautogui.hotkeys.count(("ctrl", "c")) == 2


def test_screen_quality_distinguishes_blank_body_from_content(monkeypatch, tmp_path):
    from PIL import ImageDraw
    from agent.tools.perception import PerceptionEngine

    blank_path = tmp_path / "blank.png"
    content_path = tmp_path / "content.png"

    blank = Image.new("RGB", (800, 900), (75, 92, 101))
    ImageDraw.Draw(blank).rectangle((0, 0, 800, 139), fill="white")
    blank.save(blank_path)

    content = Image.new("RGB", (800, 900), "white")
    draw = ImageDraw.Draw(content)
    for row in range(8):
        y = 180 + row * 70
        draw.rectangle((60, y, 340, y + 35), fill=(40 + row * 15, 80, 160))
        draw.rectangle((400, y, 730, y + 20), fill=(20, 20, 20))
    content.save(content_path)

    engine = object.__new__(PerceptionEngine)
    monkeypatch.delenv("VISION_PAGE_BLANK_MAX_STDDEV", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MAX_EDGE_MEAN", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MIN_DOMINANT_RATIO", raising=False)

    assert engine.screen_quality(blank_path)["low_information"] is True
    assert engine.screen_quality(content_path)["low_information"] is False


def test_screen_quality_excludes_dynamic_browser_chrome_from_blank_page(monkeypatch, tmp_path):
    from agent.tools.perception import PerceptionEngine

    image_path = tmp_path / "blank-browser.png"
    image = Image.new("RGB", (800, 900), "white")
    image.paste((45, 58, 65), (0, 0, 800, 185))
    image.save(image_path)

    engine = object.__new__(PerceptionEngine)
    monkeypatch.delenv("VISION_PAGE_CONTENT_TOP_PX", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MAX_STDDEV", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MAX_EDGE_MEAN", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MIN_DOMINANT_RATIO", raising=False)

    quality = engine.screen_quality(image_path)

    assert quality["low_information"] is True


def test_screen_quality_tolerates_small_loading_edges(monkeypatch, tmp_path):
    from PIL import ImageDraw
    from agent.tools.perception import PerceptionEngine

    image_path = tmp_path / "blank-with-loading-edges.png"
    image = Image.new("RGB", (800, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 799, 184), fill=(45, 58, 65))
    for index in range(18):
        x = 250 + (index % 9) * 35
        y = 210 + (index // 9) * 35
        draw.ellipse((x, y, x + 10, y + 10), outline=(80, 80, 80), width=2)
    image.save(image_path)

    engine = object.__new__(PerceptionEngine)
    monkeypatch.delenv("VISION_PAGE_CONTENT_TOP_PX", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MAX_STDDEV", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MAX_EDGE_MEAN", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MIN_DOMINANT_RATIO", raising=False)

    quality = engine.screen_quality(image_path)

    assert 1.0 < quality["edge_mean"] < 1.2
    assert quality["low_information"] is True


def test_screen_quality_preserves_sparse_search_ui(monkeypatch, tmp_path):
    from PIL import ImageDraw
    from agent.tools.perception import PerceptionEngine

    image_path = tmp_path / "sparse-search.png"
    image = Image.new("RGB", (800, 900), "white")
    image.paste((45, 58, 65), (0, 0, 800, 185))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((290, 220, 510, 260), radius=8, fill=(245, 245, 245))
    draw.rectangle((300, 232, 400, 240), fill=(150, 150, 150))
    for row in range(4):
        y = 300 + row * 35
        draw.rectangle((300, y, 360, y + 7), fill=(80, 80, 80))
        draw.rectangle((440, y, 500, y + 7), fill=(80, 80, 80))
    image.save(image_path)

    engine = object.__new__(PerceptionEngine)
    monkeypatch.delenv("VISION_PAGE_CONTENT_TOP_PX", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MAX_STDDEV", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MAX_EDGE_MEAN", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MIN_DOMINANT_RATIO", raising=False)

    quality = engine.screen_quality(image_path)

    assert quality["dominant_ratio"] < 0.97
    assert quality["low_information"] is False


def test_screen_quality_rejects_low_contrast_loading_shell(monkeypatch, tmp_path):
    from PIL import ImageDraw
    from agent.tools.perception import PerceptionEngine

    image_path = tmp_path / "loading-shell.png"
    image = Image.new("RGB", (800, 900), "white")
    image.paste((45, 58, 65), (0, 0, 800, 185))
    draw = ImageDraw.Draw(image)
    for row in range(2):
        y = 225 + row * 42
        draw.rounded_rectangle(
            (250, y, 550, y + 18),
            radius=7,
            fill=(180, 180, 180),
        )
    image.save(image_path)

    engine = object.__new__(PerceptionEngine)
    monkeypatch.delenv("VISION_PAGE_CONTENT_TOP_PX", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MAX_STDDEV", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MAX_EDGE_MEAN", raising=False)
    monkeypatch.delenv("VISION_PAGE_BLANK_MIN_DOMINANT_RATIO", raising=False)

    quality = engine.screen_quality(image_path)

    assert 6.0 < quality["stddev"] <= 12.0
    assert quality["low_information"] is True


def test_capture_usable_screen_retries_before_ocr(monkeypatch, tmp_path):
    from agent.tools.perception import PerceptionEngine

    engine = object.__new__(PerceptionEngine)
    paths = [tmp_path / "blank.png", tmp_path / "ready.png"]
    calls = []

    def fake_capture_screen(filename=None):
        calls.append(filename)
        return paths[len(calls) - 1]

    monkeypatch.setenv("VISION_PAGE_CAPTURE_RETRY_SEC", "0")
    monkeypatch.setenv("VISION_PAGE_READY_TIMEOUT_SEC", "1")
    monkeypatch.setattr(engine, "capture_screen", fake_capture_screen)
    monkeypatch.setattr(
        engine,
        "screen_quality",
        lambda path: {"low_information": path.name == "blank.png"},
    )

    assert engine.capture_usable_screen() == paths[1]
    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] == "blank.png"


def test_capture_usable_screen_retries_when_frame_stability_times_out(
    monkeypatch,
    tmp_path,
):
    from agent.tools.perception import PerceptionEngine

    engine = object.__new__(PerceptionEngine)
    paths = [tmp_path / "moving.png", tmp_path / "ready.png"]
    calls = []

    class FakeWaitStable:
        last_wait_result = {}

    wait_stable = FakeWaitStable()
    engine._wait_stable = wait_stable

    def fake_capture_screen(filename=None):
        calls.append(filename)
        stable = len(calls) > 1
        wait_stable.last_wait_result = {
            "stable": stable,
            "reason": (
                "consecutive_frames_stable"
                if stable
                else "stability_timeout"
            ),
            "probe_count": 2,
            "stable_frames": 2 if stable else 0,
            "diff_percent": 0.1 if stable else 2.5,
        }
        return paths[len(calls) - 1]

    monkeypatch.setenv("VISION_PAGE_CAPTURE_RETRY_SEC", "0")
    monkeypatch.setenv("VISION_PAGE_READY_TIMEOUT_SEC", "1")
    monkeypatch.setattr(engine, "capture_screen", fake_capture_screen)
    monkeypatch.setattr(
        engine,
        "screen_quality",
        lambda _path: {"low_information": False},
    )

    assert engine.capture_usable_screen() == paths[1]
    assert calls == [None, "moving.png"]
    assert engine.last_capture_quality["stable"] is True
    assert engine.last_capture_quality["stability_confirmations"] == 2


def test_capture_usable_screen_honors_single_attempt_override(monkeypatch, tmp_path):
    from agent.tools.perception import PerceptionEngine

    engine = object.__new__(PerceptionEngine)
    blank_path = tmp_path / "blank.png"
    calls = []

    monkeypatch.setattr(
        engine,
        "capture_screen",
        lambda filename=None: calls.append(filename) or blank_path,
    )
    monkeypatch.setattr(engine, "screen_quality", lambda _path: {"low_information": True})

    assert engine.capture_usable_screen(max_attempts=1) == blank_path
    assert calls == [None]


def test_capture_usable_screen_forwards_input_settle_wait(monkeypatch, tmp_path):
    from agent.tools.perception import PerceptionEngine

    engine = object.__new__(PerceptionEngine)
    ready_path = tmp_path / "ready.png"
    calls = []

    def fake_capture_screen(filename=None, initial_wait_sec=None):
        calls.append((filename, initial_wait_sec))
        return ready_path

    monkeypatch.setattr(engine, "capture_screen", fake_capture_screen)
    monkeypatch.setattr(engine, "screen_quality", lambda _path: {"low_information": False})

    assert engine.capture_usable_screen(max_attempts=1, initial_wait_sec=0.7) == ready_path
    assert calls == [(None, 0.7)]


def test_prepare_som_image_excludes_browser_toolbar_and_bookmarks(monkeypatch, tmp_path):
    from agent.tools.perception import PerceptionEngine

    image_path = tmp_path / "screen.png"
    Image.new("RGB", (800, 900), "white").save(image_path)
    engine = object.__new__(PerceptionEngine)
    monkeypatch.delenv("VISION_SOM_CROP_TOP", raising=False)

    cropped_path, crop_top = engine._prepare_som_image(image_path)

    assert crop_top == 140
    with Image.open(cropped_path) as cropped:
        assert cropped.size == (800, 760)


def test_prepare_som_image_detects_browser_content_boundary(monkeypatch, tmp_path):
    from agent.tools.perception import PerceptionEngine

    image_path = tmp_path / "browser.png"
    image = Image.new("RGB", (800, 900), (65, 80, 88))
    image.paste((250, 250, 250), (0, 185, 800, 900))
    image.save(image_path)
    engine = object.__new__(PerceptionEngine)
    monkeypatch.delenv("VISION_SOM_CROP_TOP", raising=False)

    cropped_path, crop_top = engine._prepare_som_image(image_path)

    assert crop_top == 185
    with Image.open(cropped_path) as cropped:
        assert cropped.size == (800, 715)


def test_som_engine_normalizes_paddleocr_results_and_scales_boxes():
    from agent.tools.som_engine import SomEngine

    engine = object.__new__(SomEngine)
    results = [
        {"bbox": [2.0, 4.0, 6.0, 8.0], "text": "Search", "confidence": 0.9},
        {"bbox": [0.0, 0.0, 1.0, 1.0], "text": "noise", "confidence": 0.1},
    ]

    boxes = engine._normalize_paddleocr_results(results, scale=0.5)

    assert boxes == [
        {
            "bbox": [4.0, 8.0, 12.0, 16.0],
            "type": "text",
            "text": "Search",
            "conf": 0.9,
        }
    ]


def test_som_engine_scales_only_large_images_for_ocr(monkeypatch):
    from agent.config import get_settings
    from agent.tools.som_engine import SomEngine

    engine = object.__new__(SomEngine)
    monkeypatch.delenv("SOM_OCR_MAX_DIM", raising=False)

    assert engine._ocr_scale_for_image(1024, 900) == 1.0
    assert round(engine._ocr_scale_for_image(1976, 1200), 3) == round(1152 / 1976, 3)
    assert round(engine._ocr_scale_for_image(3846, 2094), 3) == round(1152 / 3846, 3)

    monkeypatch.setenv("SOM_OCR_MAX_DIM", "1600")
    get_settings.cache_clear()
    assert round(engine._ocr_scale_for_image(3846, 2094), 3) == round(1600 / 3846, 3)

def test_som_engine_removes_icon_containers_that_duplicate_ocr_text():
    from agent.tools.som_engine import SomEngine

    engine = object.__new__(SomEngine)
    text_boxes = [
        {"bbox": [20, 20, 80, 40], "type": "text", "text": "채용", "conf": 0.9}
    ]
    icon_boxes = [
        {"bbox": [10, 10, 100, 50], "type": "icon", "text": "icon", "conf": 0.8},
        {"bbox": [120, 10, 150, 40], "type": "icon", "text": "icon", "conf": 0.8},
        {"bbox": [70, 20, 110, 40], "type": "icon", "text": "icon", "conf": 0.8},
    ]

    filtered = engine._remove_text_covered_icons(icon_boxes, text_boxes)

    assert filtered == icon_boxes[1:]


def test_som_engine_ensure_ocr_worker_ready_starts_reusable_worker(monkeypatch):
    from agent.tools.som_engine import SomEngine

    engine = object.__new__(SomEngine)
    worker = object()
    started = []
    monkeypatch.setattr(engine, "_start_ocr_worker", lambda: started.append(True) or worker)

    assert engine.ensure_ocr_worker_ready() is worker
    assert started == [True]


def test_som_engine_propagates_worker_failure_without_oneshot_fallback(monkeypatch, tmp_path):
    from agent.tools.som_engine import SomEngine

    engine = object.__new__(SomEngine)
    image_path = tmp_path / "screen.png"
    image_path.touch()
    calls = []

    def fail_worker(path):
        calls.append(path)
        raise RuntimeError("worker failed")

    monkeypatch.setattr(engine, "_run_paddle_ocr_worker", fail_worker)

    try:
        engine._run_paddle_ocr(image_path)
    except RuntimeError as exc:
        assert str(exc) == "worker failed"
    else:
        raise AssertionError("OCR 작업자 실패를 일회성 프로세스로 숨기면 안 됩니다.")

    assert calls == [image_path]
    assert not hasattr(SomEngine, "_run_paddle_ocr_once")


def test_som_engine_resolves_ocr_python_from_separate_environment(monkeypatch, tmp_path):
    from agent.tools.som_engine import SomEngine

    engine = object.__new__(SomEngine)
    engine.root_dir = tmp_path
    ocr_python = tmp_path / ".venv-ocr" / "Scripts" / "python.exe"
    ocr_python.parent.mkdir(parents=True)
    ocr_python.touch()
    monkeypatch.setenv("PADDLE_OCR_PYTHON", str(ocr_python))

    assert engine._resolve_ocr_python() == ocr_python


def test_som_engine_rejects_missing_ocr_environment(monkeypatch, tmp_path):
    from agent.tools.som_engine import SomEngine

    engine = object.__new__(SomEngine)
    engine.root_dir = tmp_path
    monkeypatch.setenv("PADDLE_OCR_PYTHON", str(tmp_path / "missing-python.exe"))

    try:
        engine._resolve_ocr_python()
    except FileNotFoundError as exc:
        assert "scripts/setup_runtime.ps1" in str(exc)
    else:
        raise AssertionError("Missing OCR environment must fail before worker launch")


def test_som_engine_uses_bounded_ocr_resize_from_yolo(monkeypatch, tmp_path):
    from agent.tools.som_engine import SomEngine

    image_path = tmp_path / "screen.jpg"
    Image.new("RGB", (3200, 1200), "white").save(image_path)

    engine = object.__new__(SomEngine)
    seen = {}
    monkeypatch.setenv("SOM_INFERENCE_MAX_DIM", "1024")
    monkeypatch.setenv("SOM_OCR_MAX_DIM", "1280")

    def fake_ocr(image, scale=1.0):
        seen["ocr_is_image"] = isinstance(image, Image.Image)
        seen["ocr_scale"] = scale
        seen["ocr_size"] = image.size
        return [{"bbox": [10, 10, 60, 30], "type": "text", "text": "Search", "conf": 0.9}]

    def fake_yolo(_img, scale):
        seen["yolo_scale"] = scale
        return []

    monkeypatch.setattr(engine, "_run_paddle_ocr", fake_ocr)
    monkeypatch.setattr(engine, "_run_yolo", fake_yolo)

    _marked, bboxes, elements = engine.process_image(image_path, output_filename="marked.jpg")

    assert seen["ocr_is_image"] is True
    assert seen["ocr_scale"] == 1280 / 3200
    assert seen["ocr_size"] == (1280, 480)
    assert round(seen["yolo_scale"], 3) == round(1024 / 3200, 3)
    assert bboxes[0] == [10, 10, 60, 30]
    assert elements[0]["text"] == "Search"


def test_wait_for_change_detects_transition_without_ocr(monkeypatch, tmp_path):
    import numpy as np

    from agent.utils.wait_stable import WaitStable

    reference_path = tmp_path / "before.png"
    Image.new("RGB", (200, 200), "white").save(reference_path)

    wait_stable = object.__new__(WaitStable)
    wait_stable.perception = object()
    monkeypatch.setattr(
        wait_stable,
        "_capture_memory_frame",
        lambda **_kwargs: np.zeros((200, 200), dtype=np.uint8),
    )

    assert wait_stable.wait_for_change(
        str(reference_path),
        max_wait_sec=0.1,
        check_interval_sec=0,
        region={"top": 0, "left": 0, "width": 200, "height": 200},
    ) is True


def test_wait_stable_requires_consecutive_stable_frames(monkeypatch):
    import numpy as np

    from agent.utils.wait_stable import WaitStable

    white = np.full((200, 200), 255, dtype=np.uint8)
    black = np.zeros((200, 200), dtype=np.uint8)
    frames = iter([white, white, black, black, black])
    capture_count = {"value": 0}

    wait_stable = object.__new__(WaitStable)
    wait_stable.perception = object()
    wait_stable.last_wait_result = {}

    def capture_frame(**_kwargs):
        capture_count["value"] += 1
        return next(frames)

    monkeypatch.setattr(
        wait_stable,
        "_capture_memory_frame",
        capture_frame,
    )
    monkeypatch.setattr(
        "agent.utils.wait_stable.time.sleep",
        lambda _seconds: None,
    )

    assert wait_stable.wait(
        max_wait_sec=1,
        check_interval_sec=0,
        threshold_percent=1,
        required_stable_frames=2,
        region={"top": 0, "left": 0, "width": 200, "height": 200},
    ) is True
    assert capture_count["value"] == 5
    assert wait_stable.last_wait_result["probe_count"] == 4
    assert wait_stable.last_wait_result["stable_frames"] == 2

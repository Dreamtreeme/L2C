from pathlib import Path

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


def test_capture_usable_screen_retries_before_ocr(monkeypatch, tmp_path):
    from agent.tools.perception import PerceptionEngine

    engine = object.__new__(PerceptionEngine)
    paths = [tmp_path / "blank.png", tmp_path / "ready.png"]
    calls = []

    def fake_capture_screen(filename=None):
        calls.append(filename)
        return paths[len(calls) - 1]

    monkeypatch.setenv("VISION_PAGE_CAPTURE_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("VISION_PAGE_CAPTURE_RETRY_SEC", "0")
    monkeypatch.setattr(engine, "capture_screen", fake_capture_screen)
    monkeypatch.setattr(
        engine,
        "screen_quality",
        lambda path: {"low_information": path.name == "blank.png"},
    )

    assert engine.capture_usable_screen() == paths[1]
    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1].startswith("screen_retry_")


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


def test_som_engine_normalizes_paddleocr_results_and_scales_boxes():
    from agent.tools.som_engine import SomEngine

    engine = object.__new__(SomEngine)
    results = [
        [
            ([[2.0, 4.0], [6.0, 4.0], [6.0, 8.0], [2.0, 8.0]], ("Search", 0.9)),
            ([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], ("noise", 0.1)),
        ]
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
    from agent.tools.som_engine import SomEngine

    engine = object.__new__(SomEngine)
    monkeypatch.delenv("SOM_OCR_RESIZE", raising=False)
    monkeypatch.delenv("SOM_OCR_MAX_DIM", raising=False)

    assert engine._ocr_scale_for_image(1024, 900) == 1.0
    assert round(engine._ocr_scale_for_image(1976, 1200), 3) == round(1280 / 1976, 3)
    assert round(engine._ocr_scale_for_image(3846, 2094), 3) == round(1280 / 3846, 3)

    monkeypatch.setenv("SOM_OCR_MAX_DIM", "1600")
    assert round(engine._ocr_scale_for_image(3846, 2094), 3) == round(1600 / 3846, 3)

    monkeypatch.setenv("SOM_OCR_RESIZE", "0")
    assert engine._ocr_scale_for_image(3846, 2094) == 1.0


def test_som_engine_recycles_ocr_worker_after_request_budget(monkeypatch):
    from agent.tools.som_engine import SomEngine

    class FakeWorker:
        pid = 1234

        def poll(self):
            return None

    engine = object.__new__(SomEngine)
    engine._ocr_worker = FakeWorker()
    engine._ocr_worker_request_count = 5
    stopped = []

    monkeypatch.setenv("SOM_OCR_WORKER_MAX_REQUESTS", "5")
    monkeypatch.setattr(engine, "_stop_ocr_worker", lambda: stopped.append(True))

    assert engine._recycle_ocr_worker_if_needed() is True
    assert stopped == [True]


def test_som_engine_does_not_recycle_before_request_budget(monkeypatch):
    from agent.tools.som_engine import SomEngine

    class FakeWorker:
        def poll(self):
            return None

    engine = object.__new__(SomEngine)
    engine._ocr_worker = FakeWorker()
    engine._ocr_worker_request_count = 4
    stopped = []

    monkeypatch.setenv("SOM_OCR_WORKER_MAX_REQUESTS", "5")
    monkeypatch.setattr(engine, "_stop_ocr_worker", lambda: stopped.append(True))

    assert engine._recycle_ocr_worker_if_needed() is False
    assert stopped == []


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

    _marked, _coords, bboxes, elements = engine.process_image(image_path, output_filename="marked.jpg")

    assert seen["ocr_is_image"] is True
    assert seen["ocr_scale"] == 1280 / 3200
    assert seen["ocr_size"] == (1280, 480)
    assert round(seen["yolo_scale"], 3) == round(1024 / 3200, 3)
    assert bboxes[0] == [10, 10, 60, 30]
    assert elements[0]["text"] == "Search"

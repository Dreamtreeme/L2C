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


def test_som_engine_normalizes_easyocr_results_and_scales_boxes():
    from agent.tools.som_engine import SomEngine

    engine = object.__new__(SomEngine)
    results = [
        ([[2.0, 4.0], [6.0, 4.0], [6.0, 8.0], [2.0, 8.0]], "Search", 0.9),
        ([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "noise", 0.1),
    ]

    boxes = engine._normalize_easyocr_results(results, scale=0.5)

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


def test_som_engine_uses_bounded_ocr_resize_from_yolo(monkeypatch, tmp_path):
    from agent.tools.som_engine import SomEngine

    image_path = tmp_path / "screen.jpg"
    Image.new("RGB", (3200, 1200), "white").save(image_path)

    engine = object.__new__(SomEngine)
    seen = {}
    monkeypatch.setenv("SOM_INFERENCE_MAX_DIM", "1024")
    monkeypatch.setenv("SOM_OCR_MAX_DIM", "1280")

    def fake_ocr(path, scale=1.0):
        seen["ocr_path"] = Path(path)
        seen["ocr_scale"] = scale
        with Image.open(path) as img:
            seen["ocr_size"] = img.size
        return [{"bbox": [10, 10, 60, 30], "type": "text", "text": "Search", "conf": 0.9}]

    def fake_yolo(_img, scale):
        seen["yolo_scale"] = scale
        return []

    monkeypatch.setattr(engine, "_run_easy_ocr", fake_ocr)
    monkeypatch.setattr(engine, "_run_yolo", fake_yolo)

    _marked, _coords, bboxes, elements = engine.process_image(image_path, output_filename="marked.jpg")

    assert seen["ocr_path"] != image_path
    assert seen["ocr_scale"] == 1280 / 3200
    assert seen["ocr_size"] == (1280, 480)
    assert round(seen["yolo_scale"], 3) == round(1024 / 3200, 3)
    assert bboxes[0] == [10, 10, 60, 30]
    assert elements[0]["text"] == "Search"

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


def test_som_engine_reuses_ocr_worker_and_scales_boxes(monkeypatch, tmp_path):
    from agent.tools.som_engine import SomEngine

    image_path = tmp_path / "ocr.jpg"
    Image.new("RGB", (10, 10), "white").save(image_path)

    engine = object.__new__(SomEngine)
    engine._ocr_worker = None
    monkeypatch.setenv("SOM_OCR_WORKER_REUSE", "true")
    monkeypatch.setattr(engine, "_run_paddle_ocr_worker", lambda _path: [
        {"text": "검색", "confidence": 0.9, "bbox": [2.0, 4.0, 6.0, 8.0]},
        {"text": "노이즈", "confidence": 0.1, "bbox": [0.0, 0.0, 1.0, 1.0]},
    ])

    boxes = engine._run_paddle_ocr(image_path, scale=0.5)

    assert boxes == [{
        "bbox": [4.0, 8.0, 12.0, 16.0],
        "type": "text",
        "text": "검색",
        "conf": 0.9,
    }]


def test_som_engine_falls_back_when_ocr_worker_fails(monkeypatch, tmp_path):
    from agent.tools.som_engine import SomEngine

    image_path = Path(tmp_path / "ocr.jpg")
    Image.new("RGB", (10, 10), "white").save(image_path)

    engine = object.__new__(SomEngine)
    engine._ocr_worker = None
    monkeypatch.setenv("SOM_OCR_WORKER_REUSE", "true")

    def fail_worker(_path):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(engine, "_run_paddle_ocr_worker", fail_worker)
    monkeypatch.setattr(engine, "_run_paddle_ocr_once", lambda _path: [
        {"text": "fallback", "confidence": 0.8, "bbox": [1.0, 2.0, 3.0, 4.0]},
    ])

    boxes = engine._run_paddle_ocr(image_path)

    assert boxes[0]["text"] == "fallback"

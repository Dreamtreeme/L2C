from PIL import Image

from agent.config import get_settings
from agent.tools.ocr_engine import OcrEngine, remove_text_containers
from agent.tools.omni_parser import OmniParser
from agent.tools.paddle_ocr import PaddleOcr


def test_paddle_ocr_normalizes_confident_results_to_text_markers():
    results = [
        {"bbox": [2.0, 4.0, 6.0, 8.0], "text": "Search", "confidence": 0.9},
        {"bbox": [0.0, 0.0, 1.0, 1.0], "text": "noise", "confidence": 0.1},
    ]

    assert PaddleOcr.normalize_results(results, scale=0.5) == [
        {
            "bbox": [4.0, 8.0, 12.0, 16.0],
            "type": "text",
            "text": "Search",
            "conf": 0.9,
        }
    ]


def test_paddle_and_omni_use_their_own_image_limits(monkeypatch):
    monkeypatch.setenv("PADDLE_OCR_MAX_DIM", "1600")
    monkeypatch.setenv("OMNIPARSER_MAX_DIM", "1024")
    get_settings.cache_clear()

    assert round(PaddleOcr.scale_for_image(3846, 2094), 3) == round(1600 / 3846, 3)
    assert round(OmniParser.scale_for_image(3846, 2094), 3) == round(1024 / 3846, 3)


def test_ocr_engine_removes_icon_containers_that_duplicate_text():
    text_boxes = [
        {"bbox": [20, 20, 80, 40], "type": "text", "text": "채용", "conf": 0.9}
    ]
    icon_boxes = [
        {"bbox": [10, 10, 100, 50], "type": "icon", "text": "icon", "conf": 0.8},
        {"bbox": [120, 10, 150, 40], "type": "icon", "text": "icon", "conf": 0.8},
        {"bbox": [70, 20, 110, 40], "type": "icon", "text": "icon", "conf": 0.8},
    ]

    assert remove_text_containers(icon_boxes, text_boxes) == icon_boxes[1:]


def test_paddle_ocr_does_not_hide_worker_failure(monkeypatch):
    paddle = object.__new__(PaddleOcr)

    def fail_worker(_path):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(paddle, "_request", fail_worker)

    try:
        paddle.detect(Image.new("RGB", (20, 20), "white"))
    except RuntimeError as exc:
        assert str(exc) == "worker failed"
    else:
        raise AssertionError("OCR 작업자 실패가 상위 호출자에게 전달되어야 합니다.")


def test_paddle_ocr_rejects_missing_worker_environment(monkeypatch, tmp_path):
    paddle = object.__new__(PaddleOcr)
    monkeypatch.setenv(
        "PADDLE_OCR_PYTHON",
        str(tmp_path / "missing-python.exe"),
    )

    try:
        paddle._resolve_python()
    except FileNotFoundError as exc:
        assert "scripts/setup_runtime.ps1" in str(exc)
    else:
        raise AssertionError("OCR Python이 없으면 작업자 실행 전에 실패해야 합니다.")


def test_paddle_worker_does_not_inherit_parent_python_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "app-site-packages")
    monkeypatch.setenv("PYTHONHOME", "app-python-home")
    get_settings.cache_clear()
    paddle = PaddleOcr(root_dir=tmp_path)

    worker_env = paddle._worker_env()

    assert "PYTHONPATH" not in worker_env
    assert "PYTHONHOME" not in worker_env


def test_ocr_engine_combines_paddle_and_omni_results(tmp_path):
    image_path = tmp_path / "screen.jpg"
    Image.new("RGB", (800, 900), "white").save(image_path)

    class FakePaddle:
        worker_pid = 123

        def detect(self, image):
            assert image.size == (800, 715)
            return [
                {
                    "bbox": [10, 10, 60, 30],
                    "type": "text",
                    "text": "Search",
                    "conf": 0.9,
                }
            ]

        def close(self):
            pass

        def ensure_ready(self):
            pass

    class FakeOmni:
        def detect(self, image):
            assert image.size == (800, 715)
            return [
                {
                    "bbox": [100, 10, 130, 40],
                    "type": "icon",
                    "text": "icon",
                    "conf": 0.8,
                }
            ]

    engine = OcrEngine(paddle=FakePaddle(), omni=FakeOmni())

    marked_path, bboxes, elements = engine.process_image(
        image_path,
        output_filename="marked.jpg",
        content_top=185,
    )

    assert marked_path.exists()
    assert bboxes == {0: [10, 195, 60, 215], 1: [100, 195, 130, 225]}
    assert [element["type"] for element in elements] == ["text", "icon"]


def test_ocr_engine_detects_only_requested_roi_and_restores_screen_coordinates(
    tmp_path,
):
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    calls = {"paddle": 0, "omni": 0}

    class FakePaddle:
        worker_pid = 123

        def detect(self, image):
            calls["paddle"] += 1
            assert image.size == (100, 50)
            return [
                {
                    "bbox": [10, 5, 30, 15],
                    "type": "text",
                    "text": "JOB검색",
                    "conf": 0.9,
                }
            ]

        def close(self):
            pass

        def ensure_ready(self):
            pass

    class FakeOmni:
        def detect(self, image):
            calls["omni"] += 1
            assert image.size == (640, 640)
            return [
                {
                    "bbox": [310, 305, 325, 320],
                    "type": "icon",
                    "text": "icon",
                    "conf": 0.8,
                }
            ]

    engine = OcrEngine(paddle=FakePaddle(), omni=FakeOmni())
    text = engine.detect_region(
        image_path,
        [0.25, 0.2, 0.75, 0.7],
        "text",
    )

    assert calls == {"paddle": 1, "omni": 0}
    assert text[0]["bbox"] == [60, 25, 80, 35]

    icon = engine.detect_region(
        image_path,
        [0.25, 0.2, 0.75, 0.7],
        "icon",
    )

    assert calls == {"paddle": 1, "omni": 1}
    assert icon[0]["bbox"] == [90, 30, 105, 45]

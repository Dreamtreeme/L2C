import io
import json
import threading
from collections import deque
from pathlib import Path

import pytest

from agent.config import get_settings
from agent.tools import paddle_ocr_runner as runner
from agent.tools.paddle_ocr import PaddleOcr
from agent.tools.paddle_ocr_runner import (
    _validate_paddle_version,
    _validate_paddleocr_version,
    extract_text_boxes_from_results,
)


def test_paddle_runtime_accepts_declared_version():
    class FakePaddle:
        __version__ = "3.3.1"

    _validate_paddle_version(FakePaddle())


def test_paddle_runtime_rejects_undeclared_version():
    class FakePaddle:
        __version__ = "2.6.2"

    try:
        _validate_paddle_version(FakePaddle())
    except RuntimeError as exc:
        assert "installed=2.6.2, required=3.3.1" in str(exc)
    else:
        raise AssertionError("Undeclared PaddlePaddle runtime must fail")


def test_paddleocr_runtime_accepts_declared_version():
    class FakePaddleOcr:
        __version__ = "3.7.0"

    _validate_paddleocr_version(FakePaddleOcr())


def test_paddleocr_runtime_rejects_legacy_api_version():
    class FakePaddleOcr:
        __version__ = "2.10.0"

    try:
        _validate_paddleocr_version(FakePaddleOcr())
    except RuntimeError as exc:
        assert "installed=2.10.0, required=3.7.0" in str(exc)
    else:
        raise AssertionError("Legacy PaddleOCR runtime must fail")


def test_paddleocr_v3_result_is_normalized_to_existing_marker_contract():
    results = [
        {
            "res": {
                "rec_texts": ["검색", "채용"],
                "rec_scores": [0.98, 0.87],
                "rec_boxes": [[10, 20, 30, 40], [50, 60, 90, 100]],
            }
        }
    ]

    assert extract_text_boxes_from_results(results) == [
        {
            "text": "검색",
            "confidence": 0.98,
            "bbox": [10.0, 20.0, 30.0, 40.0],
        },
        {
            "text": "채용",
            "confidence": 0.87,
            "bbox": [50.0, 60.0, 90.0, 100.0],
        },
    ]


def test_paddleocr_v3_polygon_falls_back_to_axis_aligned_box():
    results = [
        {
            "res": {
                "rec_texts": ["로그인"],
                "rec_scores": [0.91],
                "rec_polys": [[[8, 12], [44, 10], [46, 28], [7, 30]]],
            }
        }
    ]

    assert extract_text_boxes_from_results(results)[0]["bbox"] == [7.0, 10.0, 46.0, 30.0]


def test_paddle_worker_emits_phases_and_request_scoped_result(monkeypatch, capsys):
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
        io.StringIO(
            json.dumps({"request_id": "req-1", "image_path": "screen.png"}) + "\n"
        ),
    )

    runner.worker_main()

    lines = capsys.readouterr().out.splitlines()
    phases = [
        json.loads(line.removeprefix("__OCR_EVENT__ "))["phase"]
        for line in lines
        if line.startswith("__OCR_EVENT__ ")
    ]
    result_line = next(line for line in lines if line.startswith("__OCR_JSON_RESULT__ "))
    result = json.loads(result_line.removeprefix("__OCR_JSON_RESULT__ "))

    assert phases == [
        "request_received",
        "inference_started",
        "inference_completed",
        "result_serialized",
    ]
    assert result["request_id"] == "req-1"
    assert result["results"][0]["text"] == "검색"


def test_paddle_worker_reports_inference_error(monkeypatch, capsys):
    class FailingOcr:
        def predict(self, _image_path):
            raise RuntimeError("predictor stopped")

    monkeypatch.setattr(runner, "build_ocr", lambda: FailingOcr())
    monkeypatch.setattr(
        runner.sys,
        "stdin",
        io.StringIO(
            json.dumps({"request_id": "req-2", "image_path": "screen.png"}) + "\n"
        ),
    )

    runner.worker_main()

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    error_line = next(line for line in lines if line.startswith("__OCR_WORKER_ERROR__ "))
    error = json.loads(error_line.removeprefix("__OCR_WORKER_ERROR__ "))

    assert error["request_id"] == "req-2"
    assert error["phase"] == "inference_started"
    assert error["error_type"] == "RuntimeError"
    assert "predictor stopped" in captured.err


def test_paddle_client_reuses_live_worker():
    class LiveWorker:
        def poll(self):
            return None

    worker = LiveWorker()
    client = object.__new__(PaddleOcr)
    client._worker = worker
    client._lifecycle_lock = threading.RLock()

    assert client._start_worker() is worker


def test_paddle_client_restarts_worker_after_timeout(monkeypatch):
    monkeypatch.setenv("PADDLE_OCR_WORKER_MAX_ATTEMPTS", "2")
    get_settings.cache_clear()
    client = object.__new__(PaddleOcr)
    workers = [object(), object()]
    starts = []
    stops = []
    attempts = []

    def start_worker():
        worker = workers[len(starts)]
        starts.append(worker)
        return worker

    def request_once(worker, image_path, *, attempt, timeout_sec):
        attempts.append((worker, image_path, attempt, timeout_sec))
        if attempt == 1:
            raise TimeoutError("stalled inference")
        return [{"text": "검색"}]

    monkeypatch.setattr(client, "_start_worker", start_worker)
    monkeypatch.setattr(client, "_stop_worker", lambda: stops.append(True))
    monkeypatch.setattr(client, "_request_once", request_once)

    result = client._request(Path("screen.png"))

    assert result == [{"text": "검색"}]
    assert starts == workers
    assert stops == [True]
    assert [item[2] for item in attempts] == [1, 2]


def test_paddle_client_timeout_reports_last_worker_phase(monkeypatch):
    client = object.__new__(PaddleOcr)
    client._last_phase = {
        "request_id": "request-1",
        "phase": "inference_started",
    }
    client._stderr_lines = deque(["predictor stalled"], maxlen=40)
    monkeypatch.setattr(client, "_next_line", lambda _timeout: None)

    with pytest.raises(TimeoutError) as exc_info:
        client._wait_for_result(
            "request-1",
            timeout_sec=1.0,
            started=0.0,
        )

    message = str(exc_info.value)
    assert "last_phase=inference_started" in message
    assert "predictor stalled" in message

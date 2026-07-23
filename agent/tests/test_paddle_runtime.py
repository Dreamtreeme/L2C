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

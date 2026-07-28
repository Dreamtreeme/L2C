def test_product_chat_e2e_frame_parser_accepts_structured_sse():
    from benchmark.run_product_chat_e2e import _frame_payload

    assert _frame_payload(
        'data: [FINAL] {"status":"completed","text":"답변 [job_id:1]"}',
        "FINAL",
    ) == {
        "status": "completed",
        "text": "답변 [job_id:1]",
    }
    assert _frame_payload("data: [DONE]", "FINAL") is None
    assert _frame_payload("data: [FINAL] invalid", "FINAL") is None


def test_runtime_reuse_quality_requires_same_worker_and_closed_browser():
    from benchmark.run_runtime_reuse_e2e import _reuse_quality

    runs = [
        {
            "quality": {"passed": True},
            "ocr_startup_count": 1,
            "resource_snapshot": {
                "ocr_worker_pid": 7007,
                "browser_window_bound": False,
                "ui_model_variant_count": 1,
            },
        },
        {
            "quality": {"passed": True},
            "ocr_startup_count": 0,
            "resource_snapshot": {
                "ocr_worker_pid": 7007,
                "browser_window_bound": False,
                "ui_model_variant_count": 1,
            },
        },
    ]

    assert _reuse_quality(runs) == {
        "request_quality_passed": True,
        "same_ocr_worker_pid": True,
        "first_request_started_ocr": True,
        "later_requests_skipped_ocr_startup": True,
        "browser_closed_after_each_request": True,
        "reasoning_model_cache_reused": True,
    }

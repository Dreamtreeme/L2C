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


def test_product_chat_matrix_accepts_database_only_answer_with_valid_citations():
    from benchmark.run_product_chat_matrix import _scenario_quality

    jobs = [
        {"id": 1, "company_name": "A", "position": "iOS 개발자"},
        {"id": 2, "company_name": "B", "position": "iOS 엔지니어"},
    ]
    quality = _scenario_quality(
        {
            "expected_status": "completed",
            "collection": "forbidden",
            "clarification": "forbidden",
            "database_mutation": "forbidden",
            "minimum_citations": 2,
            "minimum_database_jobs": 2,
        },
        {
            "final": {
                "status": "completed",
                "text": "두 공고를 비교했습니다. [job_id:1] [job_id:2]",
            },
            "events": [{"event": "answering_started"}],
            "error": "",
        },
        jobs,
        jobs,
    )

    assert quality["passed"] is True
    assert quality["citation_ids"] == [1, 2]
    assert quality["collection_observed"] is False


def test_product_chat_matrix_accepts_structured_clarification():
    from benchmark.run_product_chat_matrix import _scenario_quality

    quality = _scenario_quality(
        {
            "expected_status": "waiting_input",
            "collection": "forbidden",
            "clarification": "required",
            "database_mutation": "forbidden",
            "minimum_clarification_options": 2,
        },
        {
            "final": {
                "status": "waiting_input",
                "text": "어떤 직무를 찾을까요?",
                "clarification": {
                    "options": [
                        {"option_id": "office"},
                        {"option_id": "manufacturing"},
                    ]
                },
            },
            "events": [{"event": "clarification_required"}],
            "error": "",
        },
        [],
        [],
    )

    assert quality["passed"] is True
    assert quality["clarification_observed"] is True
    assert quality["clarification_option_count"] == 2


def test_product_chat_matrix_rejects_missing_required_collection():
    from benchmark.run_product_chat_matrix import _scenario_quality

    quality = _scenario_quality(
        {
            "expected_status": "completed",
            "collection": "required",
            "clarification": "forbidden",
            "minimum_citations": 1,
        },
        {
            "final": {
                "status": "completed",
                "text": "기존 공고입니다. [job_id:1]",
            },
            "events": [{"event": "answering_started"}],
            "error": "",
        },
        [{"id": 1}],
        [{"id": 1}],
    )

    assert quality["passed"] is False
    assert quality["checks"]["collection_contract"] is False


def test_product_chat_matrix_rejects_stale_citation_after_collection():
    from benchmark.run_product_chat_matrix import _scenario_quality

    quality = _scenario_quality(
        {
            "expected_status": "completed",
            "collection": "required",
            "clarification": "forbidden",
            "citation_scope": "collection",
            "minimum_citations": 1,
        },
        {
            "final": {
                "status": "completed",
                "text": "기존 DB 공고입니다. [job_id:1]",
            },
            "events": [
                {"event": "collection_started"},
                {
                    "event": "collection_completed",
                    "data": {"document_ids": [2]},
                },
            ],
            "error": "",
        },
        [{"id": 1}],
        [{"id": 1}, {"id": 2}],
    )

    assert quality["passed"] is False
    assert quality["checks"]["citation_integrity"] is True
    assert quality["checks"]["citation_scope"] is False
    assert quality["collection_document_ids"] == [2]


def test_product_chat_matrix_accepts_citation_from_current_collection():
    from benchmark.run_product_chat_matrix import _scenario_quality

    quality = _scenario_quality(
        {
            "expected_status": "completed",
            "collection": "required",
            "clarification": "forbidden",
            "citation_scope": "collection",
            "minimum_citations": 1,
        },
        {
            "final": {
                "status": "completed",
                "text": "이번에 확인한 공고입니다. [job_id:2]",
            },
            "events": [
                {"event": "collection_started"},
                {
                    "event": "collection_completed",
                    "data": {"document_ids": [2]},
                },
            ],
            "error": "",
        },
        [{"id": 1}],
        [{"id": 1}, {"id": 2}],
    )

    assert quality["passed"] is True
    assert quality["checks"]["citation_scope"] is True


def test_product_chat_matrix_rejects_runtime_and_token_outlier():
    from benchmark.run_product_chat_matrix import _scenario_quality

    quality = _scenario_quality(
        {
            "expected_status": "completed",
            "maximum_execution_time_sec": 30,
            "maximum_total_tokens": 10000,
        },
        {
            "final": {
                "status": "completed",
                "text": "답변",
                "metrics": {
                    "duration_sec": 250,
                    "llm": {"totals": {"total_tokens": 214241}},
                },
            },
            "events": [],
            "error": "",
        },
        [],
        [],
    )

    assert quality["passed"] is False
    assert quality["checks"]["execution_time_budget"] is False
    assert quality["checks"]["token_budget"] is False
    assert quality["execution_time_sec"] == 250
    assert quality["total_tokens"] == 214241


def test_product_chat_matrix_snapshots_sqlite_database(tmp_path):
    import sqlite3

    from benchmark.run_product_chat_matrix import _snapshot_database

    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        connection.execute("INSERT INTO sample(value) VALUES ('copied')")

    _snapshot_database(source, target)

    with sqlite3.connect(target) as connection:
        value = connection.execute("SELECT value FROM sample").fetchone()
    assert value == ("copied",)


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

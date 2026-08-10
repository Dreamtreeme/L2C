import pytest

from benchmark.run_product_chat_matrix import (
    _frame_payload,
    _scenario_quality,
)


def _request(text, *, events=(), duration=0, tokens=0):
    return {
        "final": {
            "status": "completed",
            "text": text,
            "metrics": {
                "duration_sec": duration,
                "llm": {"totals": {"total_tokens": tokens}},
            },
        },
        "events": list(events),
        "error": "",
    }


def _collection_events(document_id=2):
    return [
        {"event": "collection_started"},
        {"event": "collection_completed", "data": {"document_ids": [document_id]}},
    ]


def test_product_chat_e2e_frame_parser_accepts_structured_sse():
    assert _frame_payload(
        'data: [FINAL] {"status":"completed","text":"답변 [job_id:1]"}',
        "FINAL",
    ) == {"status": "completed", "text": "답변 [job_id:1]"}
    assert _frame_payload("data: [DONE]", "FINAL") is None
    assert _frame_payload("data: [FINAL] invalid", "FINAL") is None


@pytest.mark.parametrize(
    ("contract", "request_result", "before", "after", "passed", "failed_check"),
    [
        (
            {
                "expected_status": "completed",
                "collection": "forbidden",
                "clarification": "forbidden",
                "database_mutation": "forbidden",
                "minimum_citations": 2,
                "minimum_database_jobs": 2,
            },
            _request("비교 결과 [job_id:1] [job_id:2]"),
            [{"id": 1}, {"id": 2}],
            [{"id": 1}, {"id": 2}],
            True,
            "",
        ),
        (
            {"expected_status": "completed", "collection": "required"},
            _request("기존 공고 [job_id:1]"),
            [{"id": 1}],
            [{"id": 1}],
            False,
            "collection_contract",
        ),
        (
            {
                "expected_status": "completed",
                "collection": "required",
                "citation_scope": "collection",
                "minimum_citations": 1,
            },
            _request("기존 공고 [job_id:1]", events=_collection_events()),
            [{"id": 1}],
            [{"id": 1}, {"id": 2}],
            False,
            "citation_scope",
        ),
        (
            {
                "expected_status": "completed",
                "collection": "required",
                "citation_scope": "collection",
                "minimum_citations": 1,
            },
            _request("신규 공고 [job_id:2]", events=_collection_events()),
            [{"id": 1}],
            [{"id": 1}, {"id": 2}],
            True,
            "",
        ),
        (
            {
                "expected_status": "completed",
                "maximum_execution_time_sec": 30,
                "maximum_total_tokens": 10_000,
            },
            _request("답변", duration=250, tokens=214_241),
            [],
            [],
            False,
            "execution_time_budget",
        ),
    ],
)
def test_product_chat_matrix_contracts(
    contract, request_result, before, after, passed, failed_check
):
    quality = _scenario_quality(contract, request_result, before, after)

    assert quality["passed"] is passed
    if failed_check:
        assert quality["checks"][failed_check] is False
    if contract.get("maximum_total_tokens"):
        assert quality["checks"]["token_budget"] is False
        assert quality["total_tokens"] == 214_241


def test_product_chat_matrix_accepts_structured_clarification():
    quality = _scenario_quality(
        {
            "expected_status": "waiting_input",
            "collection": "forbidden",
            "clarification": "required",
            "minimum_clarification_options": 2,
        },
        {
            "final": {
                "status": "waiting_input",
                "text": "어떤 직무를 찾을까요?",
                "clarification": {
                    "options": [{"option_id": "office"}, {"option_id": "factory"}]
                },
            },
            "events": [{"event": "clarification_required"}],
            "error": "",
        },
        [],
        [],
    )

    assert quality["passed"] is True
    assert quality["clarification_option_count"] == 2

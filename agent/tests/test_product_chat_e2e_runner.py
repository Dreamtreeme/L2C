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

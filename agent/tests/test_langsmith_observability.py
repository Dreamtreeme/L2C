from __future__ import annotations


def test_feedback_is_skipped_when_langsmith_tracing_is_disabled(monkeypatch):
    from agent.observability import langsmith_adapter

    monkeypatch.setattr(langsmith_adapter, "langsmith_tracing_enabled", lambda: False)

    result = langsmith_adapter.publish_langsmith_feedback(
        "trace-1",
        [{"key": "e2e_success", "score": 1}],
    )

    assert result == {
        "status": "skipped",
        "reason": "tracing_disabled",
        "published": 0,
    }


def test_feedback_uses_root_trace_id_and_flushes(monkeypatch):
    import langsmith

    from agent.observability import langsmith_adapter

    calls = []
    flushes = []

    class FakeClient:
        def create_feedback(self, **kwargs):
            calls.append(kwargs)

        def flush(self, timeout=None):
            flushes.append(timeout)

    monkeypatch.setattr(langsmith_adapter, "langsmith_tracing_enabled", lambda: True)
    monkeypatch.setattr(langsmith, "Client", FakeClient)
    monkeypatch.setenv("L2C_LANGSMITH_E2E_FEEDBACK", "1")
    monkeypatch.setenv("LANGSMITH_FLUSH_TIMEOUT_SEC", "2.5")

    result = langsmith_adapter.publish_langsmith_feedback(
        "trace-1",
        [
            {"key": "e2e_success", "score": 1},
            {"key": "e2e_outcome", "value": "success"},
        ],
    )

    assert result == {"status": "published", "reason": "", "published": 2}
    assert [item["trace_id"] for item in calls] == ["trace-1", "trace-1"]
    assert calls[0]["score"] == 1
    assert calls[1]["value"] == "success"
    assert flushes == [2.5]


def test_trace_setup_failure_does_not_interrupt_the_work(monkeypatch):
    import langsmith

    from agent.observability import langsmith_adapter

    monkeypatch.setattr(langsmith_adapter, "langsmith_tracing_enabled", lambda: True)

    def fail_trace(*args, **kwargs):
        raise RuntimeError("trace unavailable")

    monkeypatch.setattr(langsmith, "trace", fail_trace)

    with langsmith_adapter.langsmith_trace("test") as trace:
        assert trace is None


def test_success_outcome_clears_recovered_failure_from_terminal_state(monkeypatch):
    from agent.application import run_context as run_context_module

    monkeypatch.setattr(run_context_module, "langsmith_tracing_enabled", lambda: False)
    context = run_context_module.RunContext(run_id="run-1")
    context.record_step(
        "ocr_request",
        1.0,
        success=False,
        failure_code="ocr_timeout",
    )

    context.set_outcome("success")
    outcome = context.snapshot()["outcome"]

    assert outcome["status"] == "success"
    assert outcome["failure_stage"] == ""
    assert outcome["failure_code"] == ""

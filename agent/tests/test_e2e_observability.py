from benchmark.e2e_observability import (
    build_e2e_observability,
    build_langsmith_feedback,
)


def _summary(*, status="completed", passed=True, steps=None):
    return {
        "status": status,
        "execution_time_sec": 12.5,
        "quality": {
            "passed": passed,
            "target_fulfillment": 1.0,
            "persistence_rate": 1.0,
            "persisted_count": 2,
        },
        "result": {"persisted_count": 2},
        "metrics": {
            "steps": steps or [],
            "outcome": {
                "failure_stage": "",
                "failure_code": "",
            },
            "llm": {
                "totals": {"total_tokens": 1200},
                "cost": {"estimated_total": 0.12},
                "calls": [
                    {"component": "vision_reasoning"},
                    {"component": "detail_extraction"},
                ],
            },
        },
    }


def test_success_keeps_recovered_ocr_failure_separate_from_terminal_failure():
    result = build_e2e_observability(
        _summary(
            steps=[
                {
                    "component": "ocr_request",
                    "stage": "perception",
                    "success": False,
                    "failure_code": "ocr_timeout",
                },
                {
                    "component": "ocr_request",
                    "stage": "perception",
                    "success": True,
                },
            ]
        )
    )

    assert result["outcome"] == "success"
    assert result["terminal_failure_stage"] == ""
    assert result["terminal_failure_code"] == ""
    assert result["ocr_request_count"] == 2
    assert result["ocr_timeout_count"] == 1
    assert result["recovered_failure_count"] == 1
    assert result["recovery_success"] == 1
    assert result["tokens_per_persisted_item"] == 600.0


def test_quality_failure_is_reported_as_quality_gate():
    result = build_e2e_observability(_summary(passed=False))

    assert result["outcome"] == "partial"
    assert result["e2e_success"] == 0
    assert result["terminal_failure_stage"] == "quality_gate"
    assert result["terminal_failure_code"] == "quality_not_passed"
    assert result["wasted_tokens"] == 1200


def test_failed_ocr_step_is_terminal_failure_stage():
    result = build_e2e_observability(
        _summary(
            status="failed",
            passed=False,
            steps=[
                {
                    "component": "ocr_request",
                    "success": False,
                    "failure_code": "ocr_timeout",
                }
            ],
        )
    )

    assert result["outcome"] == "failed"
    assert result["terminal_failure_stage"] == "perception"
    assert result["terminal_failure_code"] == "ocr_timeout"


def test_replay_hits_use_graph_action_sources_only():
    result = build_e2e_observability(
        _summary(
            steps=[
                {"component": "graph:reflex", "action_source": "reflex"},
                {"component": "graph:selection", "action_source": "card_queue"},
                {"component": "graph:reflex", "hit": True},
                {"component": "graph:perception", "queue_replay_hit": True},
            ]
        )
    )

    assert result["reflex_hits"] == 1
    assert result["queue_replay_hits"] == 1


def test_langsmith_feedback_contains_numeric_and_category_values():
    observability = build_e2e_observability(_summary())
    feedback = build_langsmith_feedback(observability)
    by_key = {item["key"]: item for item in feedback}

    assert by_key["e2e_success"]["score"] == 1
    assert by_key["total_tokens"]["score"] == 1200
    assert by_key["e2e_outcome"]["value"] == "success"

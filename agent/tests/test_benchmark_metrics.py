import json


def test_profile_summary_uses_structured_runtime_metrics(tmp_path):
    from benchmark.profile_reflex_trace import profile_summary

    path = tmp_path / "run.summary.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "completed",
                "execution_time_sec": 10.0,
                "metrics": {
                    "steps": [
                        {
                            "component": "graph:ocr",
                            "duration_sec": 2.0,
                            "analysis_mode": "full",
                        },
                        {
                            "component": "graph:selection",
                            "duration_sec": 0.01,
                            "action_source": "card_queue",
                        },
                        {
                            "component": "graph:reflex",
                            "duration_sec": 0.1,
                            "action_source": "reflex",
                        },
                    ],
                    "llm": {
                        "totals": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                        "cost": {"estimated_total": 0.001},
                        "calls": [
                            {"component": "vision_reasoning", "duration_sec": 3.0}
                        ],
                    },
                },
                "quality": {"target_fulfillment": 1.0},
            }
        ),
        encoding="utf-8",
    )

    result = profile_summary(path)

    assert result["nodes"]["ocr"]["total"] == 2.0
    assert result["reflex_hits"] == 1
    assert result["queue_replay_hits"] == 1
    assert result["llm_usage"]["total_tokens"] == 15


def test_profile_path_rejects_unstructured_log(tmp_path):
    from benchmark.profile_reflex_trace import profile_path

    path = tmp_path / "run.log"
    path.write_text("Reasoning Node completed in 1.00 seconds", encoding="utf-8")

    try:
        profile_path(path)
    except ValueError as exc:
        assert ".summary.json" in str(exc)
    else:
        raise AssertionError("구조화되지 않은 로그는 성능 입력으로 허용하면 안 됩니다.")

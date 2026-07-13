import json


def test_profile_log_reports_tail_latency_and_ocr(tmp_path):
    from benchmark.profile_reflex_trace import profile_log

    path = tmp_path / "run.log"
    path.write_text(
        "\n".join(
            [
                "Perception Node completed in 1.00 seconds",
                "Perception Node completed in 9.00 seconds",
                "PaddleOCR worker request completed duration=0.40s",
                "PaddleOCR worker ready startup=2.50s",
                "EXECUTION_TIME_SEC=12.3",
            ]
        ),
        encoding="utf-8",
    )

    result = profile_log(path)

    assert result["execution_time_sec"] == 12.3
    assert result["nodes"]["perception"]["p50"] == 1.0
    assert result["nodes"]["perception"]["p95"] == 9.0
    assert result["nodes"]["ocr_request"]["max"] == 0.4


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
                        {"component": "graph:perception", "duration_sec": 2.0, "queue_replay_hit": True},
                        {"component": "graph:reflex", "duration_sec": 0.1, "hit": True},
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

    assert result["nodes"]["perception"]["total"] == 2.0
    assert result["reflex_hits"] == 1
    assert result["queue_replay_hits"] == 1
    assert result["llm_usage"]["total_tokens"] == 15


def test_profile_log_reads_json_structlog_duration(tmp_path):
    from benchmark.profile_reflex_trace import profile_log

    path = tmp_path / "run.log"
    path.write_text(
        "\n".join(
            [
                json.dumps({"event": "Perception Node completed", "duration_sec": 2.25}),
                json.dumps({"event": "PaddleOCR worker request completed", "duration": "0.51s"}),
                json.dumps(
                    {
                        "event": "SoM analysis stages completed",
                        "mode": "full",
                        "ocr_duration_sec": 0.55,
                        "yolo_duration_sec": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "event": "Reasoning Node completed",
                        "duration_sec": 1.2,
                        "reasoning_mode": "general",
                    }
                ),
                json.dumps(
                    {
                        "event": "Runtime step completed",
                        "component": "ocr_request",
                        "duration_sec": 20.0,
                        "success": False,
                    }
                ),
                json.dumps({"event": "Reflex miss: no candidate passed marker matching"}),
            ]
        ),
        encoding="utf-8",
    )

    result = profile_log(path)

    assert result["nodes"]["perception"]["total"] == 2.25
    assert result["nodes"]["ocr_request"]["total"] == 20.51
    assert result["nodes"]["ocr_request_failed"]["total"] == 20.0
    assert result["nodes"]["som_ocr"]["total"] == 0.55
    assert result["perception_modes"] == {"full": 1}
    assert result["reasoning_modes"] == {"general": 1}
    assert result["reflex_misses"] == {"no candidate passed marker matching": 1}

from benchmark.run_realtime_e2e import _apply_run_mode_environment
import sqlite3

from benchmark.run_regression_matrix import (
    _clear_jobs_for_warm_run,
    _metric_summary,
    _scenario_environment,
)


def test_metric_summary_prefers_ocr_request_metrics() -> None:
    summary = _metric_summary(
        {
            "status": "success",
            "quality": {"passed": True},
            "metrics": {
                "steps": [
                    {"component": "ocr_startup", "stage": "perception", "duration_sec": 6.0},
                    {"component": "ocr_request", "stage": "perception", "duration_sec": 1.0},
                    {"component": "ocr_request", "stage": "perception", "duration_sec": 3.0},
                    {"stage": "ocr", "duration_sec": 1.0},
                    {"stage": "ocr", "duration_sec": 3.0},
                    {"stage": "reasoning", "duration_sec": 2.5},
                    {"stage": "reflex", "action_source": "reflex"},
                    {"stage": "selection", "action_source": "card_queue"},
                ],
                "llm": {
                    "totals": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12,
                    }
                },
            },
        }
    )

    assert summary["ocr_count"] == 2
    assert summary["ocr_time_sec"] == 4.0
    assert summary["ocr_p50_sec"] == 1.0
    assert summary["ocr_p95_sec"] == 3.0
    assert summary["ocr_startup_sec"] == 6.0
    assert summary["reasoning_count"] == 1
    assert summary["reflex_count"] == 1
    assert summary["queue_count"] == 1
    assert summary["total_tokens"] == 12


def test_run_modes_control_reflex_environment(monkeypatch) -> None:
    monkeypatch.setenv("REFLEX_ENABLED", "1")
    _apply_run_mode_environment("cold")
    assert _scenario_environment({"run_mode": "cold"})["REFLEX_ENABLED"] == "0"

    _apply_run_mode_environment("warm")
    assert _scenario_environment({"run_mode": "warm"})["REFLEX_ENABLED"] == "1"


def test_scenario_environment_uses_isolated_database(tmp_path) -> None:
    db_path = tmp_path / "regression.db"

    environment = _scenario_environment({"run_mode": "cold"}, db_path=db_path)

    assert environment["DB_PATH"] == str(db_path)


def test_warm_reset_keeps_recipes_and_removes_jobs(tmp_path) -> None:
    db_path = tmp_path / "regression.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, title TEXT)")
        connection.execute("CREATE TABLE recipes (recipe_key TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO jobs (title) VALUES ('iOS 개발자')")
        connection.execute("INSERT INTO recipes (recipe_key) VALUES ('roi2#search')")

    assert _clear_jobs_for_warm_run(db_path) == 1

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM recipes").fetchone()[0] == 1

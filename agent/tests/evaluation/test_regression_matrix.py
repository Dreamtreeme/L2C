import sqlite3

from benchmark.run_realtime_e2e import (
    _apply_execution_mode_environment,
    _finalize_experience_guided_preconditions,
)
from benchmark.run_regression_matrix import (
    _attach_promotion_metrics,
    _clear_jobs_for_collection_run,
    _command,
    _metric_summary,
    _mode_contract_passed,
    _mode_pair_efficiency,
    _paired_autonomous_failed,
    _promote_autonomous_candidate,
    _scenario_environment,
    _scenario_pair_key,
    _scenario_workload_key,
)


def test_e2e_command_uses_execution_mode_option(tmp_path) -> None:
    command = _command(
        {
            "id": "wanted-ios-experience-guided",
            "site": "wanted",
            "search_keyword": "iOS 개발자",
            "target_count": 2,
            "count_mode": "explicit",
            "execution_mode": "experience_guided",
        },
        tmp_path / "run.log",
        tmp_path / "run.summary.json",
    )

    assert "--execution-mode" in command
    assert command[command.index("--search-keyword") + 1] == "iOS 개발자"
    assert "--query" not in command
    assert command[command.index("--execution-mode") + 1] == ("experience_guided")
    assert "--run-mode" not in command


def test_metric_summary_prefers_ocr_request_metrics() -> None:
    summary = _metric_summary(
        {
            "status": "success",
            "quality": {"passed": True},
            "metrics": {
                "steps": [
                    {
                        "component": "ocr_startup",
                        "stage": "perception",
                        "duration_sec": 6.0,
                    },
                    {
                        "component": "ocr_request",
                        "stage": "perception",
                        "duration_sec": 1.0,
                    },
                    {
                        "component": "ocr_request",
                        "stage": "perception",
                        "duration_sec": 3.0,
                    },
                    {"stage": "ocr", "duration_sec": 1.0},
                    {"stage": "ocr", "duration_sec": 3.0},
                    {"stage": "reasoning", "duration_sec": 2.5},
                    {"stage": "reflex", "action_source": "reflex"},
                    {"stage": "selection", "action_source": "job_card_queue"},
                    {"stage": "execution", "action_source": "reflex"},
                    {"stage": "execution", "action_source": "job_card_queue"},
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
    experience_metrics = {
        **summary,
        "reflex_path_completed_count": 1,
        "experience_guided_performance_comparable": True,
    }
    assert _mode_contract_passed(
        {"execution_mode": "experience_guided"},
        experience_metrics,
        {},
    )
    assert not _mode_contract_passed(
        {"execution_mode": "experience_guided"},
        {**experience_metrics, "reflex_path_completed_count": 0},
        {},
    )


def test_experience_guided_comparison_rejects_existing_jobs() -> None:
    preconditions = {
        "required": True,
        "roi_recipes": 2,
        "performance_comparable": True,
        "reasons": [],
    }

    result = _finalize_experience_guided_preconditions(
        preconditions,
        {"observed_existing_count": 3},
    )

    assert result["performance_comparable"] is False
    assert result["reasons"] == ["existing_jobs_observed"]


def test_execution_modes_control_reflex_environment(monkeypatch) -> None:
    monkeypatch.setenv("REFLEX_ENABLED", "1")
    _apply_execution_mode_environment("autonomous")
    assert (
        _scenario_environment({"execution_mode": "autonomous"})["REFLEX_ENABLED"] == "0"
    )

    _apply_execution_mode_environment("experience_guided")
    assert (
        _scenario_environment({"execution_mode": "experience_guided"})["REFLEX_ENABLED"]
        == "1"
    )


def test_scenario_environment_uses_isolated_database(tmp_path) -> None:
    db_path = tmp_path / "regression.db"

    environment = _scenario_environment(
        {"execution_mode": "autonomous"},
        db_path=db_path,
    )

    assert environment["DB_PATH"] == str(db_path)
    assert environment["VISION_RECIPE_AUTO_PROMOTE"] == "0"


def test_collection_run_reset_keeps_recipes_and_removes_jobs(
    tmp_path,
) -> None:
    db_path = tmp_path / "regression.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, title TEXT)")
        connection.execute("CREATE TABLE recipes (recipe_key TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO jobs (title) VALUES ('iOS 개발자')")
        connection.execute("INSERT INTO recipes (recipe_key) VALUES ('roi2#search')")

    assert _clear_jobs_for_collection_run(db_path) == 1

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM recipes").fetchone()[0] == 1


def test_scenario_workload_key_matches_both_execution_modes() -> None:
    autonomous = {
        "site": "Wanted",
        "search_keyword": " iOS 개발자 ",
        "target_count": 2,
        "count_mode": "explicit",
        "execution_mode": "autonomous",
    }
    experience_guided = {
        **autonomous,
        "site": "wanted",
        "search_keyword": "iOS 개발자",
        "execution_mode": "experience_guided",
    }

    workload_key = _scenario_workload_key(autonomous)
    assert workload_key == _scenario_workload_key(experience_guided)
    pair_key = _scenario_pair_key(experience_guided)
    assert _paired_autonomous_failed(
        experience_guided,
        {pair_key: False},
    )
    assert not _paired_autonomous_failed(
        experience_guided,
        {pair_key: True},
    )
    assert not _paired_autonomous_failed(
        autonomous,
        {workload_key: False},
    )
    assert _scenario_pair_key({**autonomous, "repeat_index": 2}).endswith("#repeat=2")


def test_promotion_metrics_are_added_to_collection_metrics() -> None:
    review_metrics = {
        "attempt_count": 2,
        "duration_sec": 42.0,
        "input_tokens": 180,
        "output_tokens": 20,
        "total_tokens": 200,
        "estimated_cost": 0.003,
    }
    combined = _attach_promotion_metrics(
        {"total_tokens": 500, "estimated_cost": 0.01},
        {"review_metrics": review_metrics},
    )

    assert review_metrics["attempt_count"] == 2
    assert review_metrics["duration_sec"] == 42.0
    assert review_metrics["total_tokens"] == 200
    assert review_metrics["estimated_cost"] == 0.003
    assert combined["workflow_total_tokens"] == 700
    assert combined["workflow_estimated_cost"] == 0.013


def test_mode_pair_efficiency_includes_critic_break_even() -> None:
    base = {
        "site": "wanted",
        "search_keyword": "iOS 개발자",
        "target_count": 2,
        "count_mode": "explicit",
        "repeat_index": 1,
    }
    report = _mode_pair_efficiency(
        [
            {
                "scenario": {
                    **base,
                    "execution_mode": "autonomous",
                },
                "mode_contract_passed": True,
                "metrics": {
                    "quality_passed": True,
                    "execution_time_sec": 120,
                    "reasoning_count": 20,
                    "total_tokens": 1000,
                    "estimated_cost": 0.02,
                    "promotion_estimated_cost": 0.006,
                },
            },
            {
                "scenario": {
                    **base,
                    "execution_mode": "experience_guided",
                },
                "mode_contract_passed": True,
                "metrics": {
                    "quality_passed": True,
                    "experience_guided_performance_comparable": True,
                    "execution_time_sec": 90,
                    "reasoning_count": 10,
                    "total_tokens": 600,
                    "estimated_cost": 0.014,
                },
            },
        ]
    )

    assert report[0]["median_execution_time_saved_sec"] == 30
    assert report[0]["median_tokens_saved"] == 400
    assert report[0]["break_even_repeat_count"] == 1.0


def test_autonomous_promotion_uses_worker_retry_and_persists_attempts(
    tmp_path,
    monkeypatch,
) -> None:
    from agent.application import recipe_candidate_review_service
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.submission_store import SubmissionStore
    from shared.schema.feedback_schema import WorkerSubmission

    db_path = tmp_path / "regression.db"
    store = RecipeCandidateStore(db_path)
    submission_store = SubmissionStore(db_path)

    def commit_candidate(submission):
        run_id = submission_store.commit_submission(submission)
        return store.commit_candidate(
            submission,
            run_id=run_id,
        )

    other_run_id = commit_candidate(
        WorkerSubmission(
            run_id="other-run",
            goal="다른 후보",
            collection_intent={
                "site": "saramin",
                "search_keyword": "백엔드",
                "task_category": "검색",
            },
            action_events=[
                {
                    "seq": 0,
                    "candidate_action": {
                        "source_seq": 0,
                        "action": "click_marker",
                    },
                    "transition": {
                        "seq": 0,
                        "before": {"observation_id": "other:1"},
                        "actions": [{"source_seq": 0, "action": "click_marker"}],
                        "after": {"observation_id": "other:2"},
                        "evidence": {"result_status": "success", "status": "ready"},
                    },
                }
            ],
        ),
    )
    assert store.enqueue_review(other_run_id) is True
    run_id = commit_candidate(
        WorkerSubmission(
            run_id="autonomous-run",
            goal="iOS 개발자 공고 수집",
            collection_intent={
                "site": "wanted",
                "search_keyword": "iOS 개발자",
                "task_category": "검색",
            },
            action_events=[
                {
                    "seq": 0,
                    "candidate_action": {
                        "source_seq": 0,
                        "action": "type_in_marker",
                        "roi_signature": {"phash": "0" * 16},
                        "target": {"text": "검색"},
                    },
                    "transition": {
                        "seq": 0,
                        "before": {"observation_id": "autonomous:1"},
                        "actions": [
                            {
                                "source_seq": 0,
                                "action": "type_in_marker",
                                "roi_signature": {"phash": "0" * 16},
                                "target": {"text": "검색"},
                            }
                        ],
                        "after": {"observation_id": "autonomous:2"},
                        "evidence": {"result_status": "success", "status": "ready"},
                    },
                }
            ],
        ),
    )
    calls = []

    def review(value, db_path=None, mode="review", raise_on_critic_error=False):
        calls.append(value)
        if len(calls) == 1:
            raise TimeoutError("critic timeout")
        promotion = {
            "enabled": True,
            "promoted": True,
            "saved_count": 1,
            "promoted_action_count": 1,
            "skipped_steps": [],
        }
        RecipeCandidateStore(db_path).update_status(
            value,
            "accepted",
            validation={
                "review": {"decision": "accept"},
                "promotion": promotion,
            },
        )
        return {
            "decision": "accept",
            "promotion": promotion,
        }

    monkeypatch.setattr(
        recipe_candidate_review_service,
        "review_and_apply_candidate",
        review,
    )

    result = _promote_autonomous_candidate(
        {"result": {"worker_run_id": run_id}},
        db_path=db_path,
    )
    candidate = store.get_candidate(run_id)
    other_candidate = store.get_candidate(other_run_id)

    assert calls == [run_id, run_id]
    assert result["promoted"] is True
    assert result["review_status"] == "accepted"
    assert result["review_attempts"] == 2
    assert result["review_metrics"]["attempt_count"] == 2
    assert candidate is not None
    assert other_candidate is not None
    assert candidate.review_attempts == 2
    assert candidate.review_error == ""
    assert other_candidate.status == "pending_review"
    assert other_candidate.review_attempts == 0

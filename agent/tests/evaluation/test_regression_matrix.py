import sqlite3

import pytest

from benchmark.run_realtime_e2e import (
    _apply_execution_mode_environment,
    _experience_guided_preconditions,
    _stored_job_snapshots,
)
from benchmark.quality_eval import evaluate_expected_source_urls
from benchmark.run_regression_matrix import (
    _attach_promotion_metrics,
    _clear_jobs_for_collection_run,
    _command,
    _expand_scenarios,
    _metric_summary,
    _mode_contract_passed,
    _experience_reuse_effectiveness,
    _paired_autonomous_failed,
    _promote_autonomous_candidate,
    _require_new_test_database,
    _scenario_environment,
    _scenario_pair_key,
    _scenario_workload_key,
    _target_contract_passed,
)
from shared.db.database import Database
from shared.schema.jd_schema import JobCollectionEvidence, JobPosting


def test_e2e_summary_preserves_stored_job_fields_before_database_reset(
    tmp_path,
) -> None:
    db_path = tmp_path / "jobs.db"
    job_id = Database(db_path).upsert(
        JobPosting(
            company_name="테스트 회사",
            position="AI 엔지니어",
            url="https://example.com/jobs/1",
            main_tasks=["AI 에이전트 개발"],
            requirements=["Python 경험"],
            tech_stack=["Python"],
            raw_ocr_text="주요 업무 AI 에이전트 개발 자격요건 Python 경험",
        ),
        evidence=JobCollectionEvidence(screenshot_path="screen.png"),
    )

    snapshots = _stored_job_snapshots(
        {"document_ids": [job_id]},
        db_path=db_path,
    )

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["company_name"] == "테스트 회사"
    assert snapshot["position"] == "AI 엔지니어"
    assert snapshot["main_tasks"] == ["AI 에이전트 개발"]
    assert snapshot["requirements"] == ["Python 경험"]
    assert snapshot["raw_ocr_text"] == (
        "주요 업무 AI 에이전트 개발 자격요건 Python 경험"
    )
    assert snapshot["screenshot_path"] == "screen.png"
    assert snapshot["evidence_hash"]


def test_expected_target_contract_requires_each_fixed_url_once() -> None:
    expected_urls = [
        "https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=11",
        "https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=22",
    ]

    def persisted_items(*rec_indexes: int) -> list[dict[str, str]]:
        return [
            {
                "url": (
                    "https://www.saramin.co.kr/zf_user/jobs/relay/view"
                    f"?view_type=search&rec_idx={rec_index}&searchword=AI"
                )
            }
            for rec_index in rec_indexes
        ]

    matched = evaluate_expected_source_urls(expected_urls, persisted_items(11, 22))
    duplicated = evaluate_expected_source_urls(expected_urls, persisted_items(11, 11))
    extra = evaluate_expected_source_urls(
        expected_urls,
        persisted_items(11, 22, 33),
    )
    duplicate_variant = evaluate_expected_source_urls(
        expected_urls[:1],
        [
            {
                "url": (
                    "https://www.saramin.co.kr/zf_user/jobs/relay/view"
                    f"?rec_idx=11&search_uuid={search_uuid}"
                )
            }
            for search_uuid in ("first", "second")
        ],
    )

    assert matched["passed"] is True
    assert matched["matched_count"] == 2
    assert duplicated["passed"] is False
    assert duplicated["matched_count"] == 1
    assert duplicated["missing_urls"] == [expected_urls[1]]
    assert duplicated["unexpected_urls"] == [persisted_items(11)[0]["url"]]
    assert extra["passed"] is False
    assert extra["unexpected_urls"] == [persisted_items(33)[0]["url"]]
    assert duplicate_variant["passed"] is False
    assert duplicate_variant["matched_count"] == 1
    assert len(duplicate_variant["unexpected_urls"]) == 1


def test_target_contract_must_be_explicit() -> None:
    assert _target_contract_passed({}) is False
    assert _target_contract_passed({"target_contract": {}}) is False
    assert (
        _target_contract_passed(
            {"target_contract": {"required": False, "passed": True}}
        )
        is True
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
            "expected_source_urls": [
                "https://www.wanted.co.kr/wd/11",
                "https://www.wanted.co.kr/wd/22",
            ],
        },
        tmp_path / "run.log",
        tmp_path / "run.summary.json",
    )

    assert "--execution-mode" in command
    assert command[command.index("--search-keyword") + 1] == "iOS 개발자"
    assert "--query" not in command
    assert command[command.index("--execution-mode") + 1] == ("experience_guided")
    assert [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--expected-source-url"
    ] == [
        "https://www.wanted.co.kr/wd/11",
        "https://www.wanted.co.kr/wd/22",
    ]
    assert command[-1] == str(tmp_path / "run.summary.json")
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
        "experience_guided_replay_ready": True,
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


def test_collection_success_does_not_require_recipe_promotion() -> None:
    assert _mode_contract_passed(
        {
            "execution_mode": "autonomous",
            "require_recipe_promotion": False,
        },
        {},
        {},
    )
    assert not _mode_contract_passed(
        {
            "execution_mode": "autonomous",
            "require_recipe_promotion": True,
        },
        {},
        {"promoted": False},
    )
    assert _mode_contract_passed(
        {
            "execution_mode": "autonomous",
            "require_recipe_promotion": True,
        },
        {},
        {"promoted": True},
    )


def test_repeated_modes_are_expanded_as_same_round_pairs() -> None:
    scenarios = _expand_scenarios(
        [
            {
                "id": "wanted-autonomous",
                "execution_mode": "autonomous",
                "repeat": 2,
            },
            {
                "id": "wanted-experience",
                "execution_mode": "experience_guided",
                "repeat": 2,
            },
        ]
    )

    assert [scenario["id"] for scenario in scenarios] == [
        "wanted-autonomous-r1",
        "wanted-experience-r1",
        "wanted-autonomous-r2",
        "wanted-experience-r2",
    ]


def test_experience_guided_preconditions_use_active_experience_rules(
    tmp_path,
    monkeypatch,
) -> None:
    from agent.recipe.store import ExperienceRuleStore

    monkeypatch.setattr(
        ExperienceRuleStore,
        "active_counts",
        lambda _store, _site: {"experience_rules": 1, "total": 1},
    )

    result = _experience_guided_preconditions(
        "experience_guided",
        "wanted",
        db_path=tmp_path / "regression.db",
    )

    assert result["replay_ready"] is True
    assert result["reasons"] == []
    assert result["experience_rules"] == 1


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


def test_collection_matrix_requires_a_new_empty_database_path(tmp_path) -> None:
    new_db_path = tmp_path / "new" / "regression.db"
    _require_new_test_database(new_db_path)
    assert new_db_path.exists() is False
    assert new_db_path.parent.is_dir()

    existing_db_path = tmp_path / "existing.db"
    existing_db_path.touch()
    with pytest.raises(SystemExit, match="이미 존재합니다"):
        _require_new_test_database(existing_db_path)


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


def test_experience_reuse_effectiveness_counts_validated_reasoning_bypass() -> None:
    base = {
        "site": "wanted",
        "search_keyword": "iOS 개발자",
        "target_count": 2,
        "count_mode": "explicit",
        "repeat_index": 1,
    }
    report = _experience_reuse_effectiveness(
        [
            {
                "scenario": {
                    **base,
                    "execution_mode": "autonomous",
                },
                "mode_contract_passed": True,
                "target_contract": {"required": False, "passed": True},
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
                "target_contract": {"required": False, "passed": True},
                "metrics": {
                    "quality_passed": True,
                    "experience_guided_replay_ready": True,
                    "execution_time_sec": 90,
                    "reasoning_count": 10,
                    "total_tokens": 600,
                    "estimated_cost": 0.014,
                    "reflex_reasoning_call_reduction": 2,
                    "reflex_source_reasoning_replaced_count": 3,
                    "reflex_path_started_count": 1,
                    "reflex_path_completed_count": 1,
                    "reflex_path_failed_count": 0,
                    "reflex_path_fallback_count": 0,
                },
            },
        ]
    )

    assert report[0]["experience_run_count"] == 1
    assert report[0]["validated_run_count"] == 1
    assert report[0]["validated_reasoning_call_reduction"] == 2
    assert report[0]["validated_source_reasoning_replaced_count"] == 3
    assert report[0]["reflex_path_completion_rate"] == 1.0
    assert "median_execution_time_saved_sec" not in report[0]


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
            "pruned_nodes": [],
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

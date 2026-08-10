import pytest

from benchmark.manual_evaluation import RunManualJudgement, evaluate_manual_run
from benchmark.quality_eval import (
    evaluate_collection_summary,
    evaluate_job_records,
)
from benchmark.site_adaptation_eval import (
    SiteAdaptationManifest,
    evaluate_site_adaptation,
)
from benchmark.user_study_eval import UserStudyManifest, evaluate_user_study


def _collection(*, target=1, resolved=1, status="completed", items=None):
    persisted = len(items or [])
    return {
        "result": {
            "target_count": target,
            "collected_count": persisted,
            "persisted_count": persisted,
            "resolved_count": resolved,
            "status": status,
            "worker_finished": True,
            "persisted_items": items or [],
        }
    }


def _job_judgement(url, *, semantic=True):
    return {
        "url": url,
        "semantic_match": semantic,
        "company_name": "pass",
        "job_title": "pass",
        "responsibilities": "pass",
        "requirements": "pass",
    }


def _manual(jobs):
    return RunManualJudgement.model_validate(
        {
            "run_id": "manual-run",
            "summary_path": "unused.json",
            "site": "example",
            "query": "AI 엔지니어",
            "search_conditions_correct": True,
            "count_handling_correct": True,
            "no_out_of_scope_actions": True,
            "jobs": jobs,
        }
    )


def test_job_quality_uses_canonical_fields_and_detects_duplicate_urls():
    result = evaluate_job_records(
        {
            "jobs": [
                {
                    "company_name": "A",
                    "position": "iOS",
                    "url": "https://example.com/1",
                },
                {
                    "company_name": "B",
                    "position": "서버",
                    "url": "https://example.com/1?ref=x",
                },
            ]
        }
    )

    assert result["record_count"] == 2
    assert result["required_field_coverage"] == 1.0
    assert result["unique_url_rate"] == 0.5


def test_job_quality_uses_exact_reference_identity():
    result = evaluate_job_records(
        [
            {
                "company_name": "VoyagerX",
                "position": "iOS Engineer",
                "url": "https://example.com/jobs/1?source=test",
                "requirements": ["Swift"],
            }
        ],
        [
            {
                "company_name": "VoyagerX",
                "position": "iOS Developer",
                "url": "https://example.com/jobs/1",
                "requirements": ["Swift"],
            }
        ],
    )["reference"]

    assert result["url_recall"] == 1.0
    assert result["identity_exact_rate"] == 0.5
    assert result["content_exact_rate"] == 1.0


def test_collection_quality_requires_resolved_detail_records():
    cases = [
        (
            {
                "target_count": 2,
                "collected_count": 2,
                "persisted_count": 1,
                "resolved_count": 1,
                "status": "partial",
                "worker_finished": True,
            },
            {"target_fulfillment": 0.5, "passed": False},
        ),
        (
            {
                "target_count": 2,
                "resolved_count": 2,
                "observed_job_ids": [7, 8],
                "status": "completed",
                "worker_finished": True,
            },
            {
                "observed_existing_count": 2,
                "target_fulfillment": 1.0,
                "passed": True,
            },
        ),
        *[
            (
                {
                    "target_count": 1,
                    "collected_count": 1,
                    "persisted_count": 1,
                    "resolved_count": 1,
                    "status": "completed",
                    "worker_finished": True,
                    "persisted_items": [{"job_id": 1, "url": url}],
                },
                {
                    "passed": passed,
                    "source_url_integrity": float(passed),
                },
            )
            for url, passed in (
                ("https://www.saramin.co.kr/zf_user/search?searchword=ML", False),
                (
                    "https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=2",
                    True,
                ),
            )
        ],
    ]

    for summary, expected in cases:
        result = evaluate_collection_summary(summary)
        for field, value in expected.items():
            assert result[field] == value, (summary, field)


@pytest.mark.parametrize(
    ("summary", "judged_jobs", "automatic", "manual", "outcome"),
    [
        (
            _collection(
                items=[{"job_id": 1, "url": "https://example.com/jobs/1"}]
            ),
            [_job_judgement("https://example.com/jobs/1", semantic=False)],
            True,
            False,
            "failure",
        ),
        (
            _collection(
                target=2,
                resolved=2,
                items=[
                    {"job_id": 1, "url": "https://example.com/jobs/1"},
                    {"job_id": 2, "url": "https://example.com/jobs/2"},
                ],
            ),
            [_job_judgement("https://example.com/jobs/1")],
            True,
            False,
            "failure",
        ),
        (
            _collection(
                target=2,
                resolved=1,
                status="partial",
                items=[{"job_id": 1, "url": "https://example.com/jobs/1"}],
            ),
            [_job_judgement("https://example.com/jobs/1")],
            False,
            True,
            "partial",
        ),
    ],
)
def test_manual_evaluation_contracts(summary, judged_jobs, automatic, manual, outcome):
    result = evaluate_manual_run(summary, _manual(judged_jobs))

    assert result["automatic_contract_passed"] is automatic
    assert result["manual_contract_passed"] is manual
    assert result["outcome"] == outcome


def _adaptation_record(approach, **overrides):
    record = {
        "site": "example",
        "approach": approach,
        "implementation_minutes": 10,
        "site_specific_code_lines": 0,
        "modified_file_count": 1,
        "common_runtime_code_lines": 0,
        "fix_iteration_count": 0,
        "successful_runs": 0,
        "attempted_runs": 3,
        "runtime_sec": [],
    }
    record.update(overrides)
    return record


def test_site_adaptation_reports_common_runtime_work_and_validity():
    valid = SiteAdaptationManifest.model_validate(
        {
            "commit_sha": "abc123",
            "task_contract": {"target_count": 2},
            "records": [
                _adaptation_record(
                    "classic",
                    implementation_minutes=90,
                    site_specific_code_lines=120,
                    successful_runs=3,
                    runtime_sec=[10, 11, 12],
                ),
                _adaptation_record(
                    "vision",
                    implementation_minutes=30,
                    common_runtime_code_lines=20,
                    successful_runs=3,
                    runtime_sec=[90, 95, 100],
                ),
            ],
        }
    )
    incomplete = SiteAdaptationManifest.model_validate(
        {
            "commit_sha": "abc123",
            "task_contract": {},
            "records": [_adaptation_record(mode) for mode in ("classic", "vision")],
        }
    )

    valid_result = evaluate_site_adaptation(valid)["sites"][0]
    incomplete_result = evaluate_site_adaptation(incomplete)["sites"][0]
    assert valid_result["implementation_minutes_saved"] == 60
    assert valid_result["vision_profile_only"] is False
    assert valid_result["comparison_valid"] is True
    assert incomplete_result["comparison_valid"] is False


def _study_record(mode, task_id, *, total=100, active=80):
    return {
        "participant_id": "P1",
        "task_id": task_id,
        "mode": mode,
        "order": 1,
        "total_completion_sec": total,
        "human_active_sec": active,
        "result_review_sec": 10,
        "suitable_job_count": 1,
        "duplicate_count": 0,
        "missing_field_count": 0,
        "factual_error_count": 0,
        "citation_link_rate": 1,
        "usefulness_score": 4,
        "trust_score": 4,
        "correction_count": 0,
    }


def test_user_study_reports_efficiency_only_for_complete_crossover():
    complete = UserStudyManifest.model_validate(
        {
            "study_contract": {},
            "records": [
                _study_record("manual", "T1", total=300, active=300),
                _study_record("l2c", "T1", total=180, active=60),
            ],
        }
    )
    incomplete = UserStudyManifest.model_validate(
        {
            "study_contract": {"participants": 3, "tasks": ["T1", "T2"]},
            "records": [
                _study_record("manual", "T1"),
                _study_record("l2c", "T2"),
            ],
        }
    )

    complete_result = evaluate_user_study(complete)
    incomplete_result = evaluate_user_study(incomplete)
    assert complete_result["human_activity_reduction_rate"] == 0.8
    assert incomplete_result["design_complete"] is False
    assert incomplete_result["human_activity_reduction_rate"] is None

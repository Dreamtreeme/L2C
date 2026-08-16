import pytest

from benchmark.e2e_observability import build_e2e_observability
from benchmark.manual_evaluation import RunManualJudgement, evaluate_manual_run
from benchmark.quality_eval import (
    evaluate_collection_summary,
    evaluate_job_records,
)
from benchmark.site_adaptation_eval import (
    SiteAdaptationManifest,
    evaluate_site_adaptation,
)
from benchmark.site_onboarding_acceptance import (
    evaluate_site_onboarding_acceptance,
)


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


def test_fixed_target_failure_reaches_e2e_observability():
    result = {
        "target_count": 1,
        "collected_count": 1,
        "persisted_count": 1,
        "resolved_count": 1,
        "status": "completed",
        "worker_finished": True,
        "persisted_items": [
            {
                "job_id": 1,
                "url": ("https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=22"),
            }
        ],
    }
    quality = evaluate_collection_summary(
        result,
        expected_source_urls=[
            "https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=11"
        ],
    )
    observability = build_e2e_observability(
        {
            "status": "completed",
            "quality": quality,
            "result": result,
            "metrics": {},
        }
    )

    assert quality["passed"] is False
    assert quality["target_contract"]["missing_urls"]
    assert quality["target_contract"]["unexpected_urls"]
    assert observability["e2e_success"] == 0
    assert observability["outcome"] == "partial"
    assert observability["terminal_failure_code"] == "quality_not_passed"


def test_observability_counts_current_vision_reasoning_tiers():
    observability = build_e2e_observability(
        {
            "status": "completed",
            "quality": {"passed": True},
            "result": {},
            "metrics": {
                "llm": {
                    "calls": [
                        {"component": "vision_reasoning_lightweight"},
                        {"component": "vision_reasoning_primary"},
                        {"component": "job_card_selection"},
                        {"component": "detail_review"},
                    ]
                }
            },
        }
    )

    assert observability["llm_call_count"] == 4
    assert observability["reasoning_call_count"] == 2


@pytest.mark.parametrize(
    ("summary", "judged_jobs", "automatic", "manual", "outcome"),
    [
        (
            _collection(items=[{"job_id": 1, "url": "https://example.com/jobs/1"}]),
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
        "homepage": "https://example.com",
        "approach": approach,
        "baseline_sha": "abc123",
        "result_sha": "def456",
        "codex_model": "same-model",
        "prompt_sha256": "prompt-hash",
        "started_at": "2026-08-16T00:00:00Z",
        "first_success_at": "2026-08-16T00:00:10Z",
        "finished_at": "2026-08-16T00:01:30Z",
        "status": "completed",
        "site_specific_changed_loc": 0,
        "modified_product_files": ["adapter.py"],
        "acceptance_runs": [
            {
                "query": "검증 검색어",
                "summary_path": "summary.json",
                "passed": True,
                "runtime_sec": 10,
            }
        ],
    }
    record.update(overrides)
    return record


def test_site_adaptation_reports_common_runtime_work_and_validity():
    valid = SiteAdaptationManifest.model_validate(
        {
            "baseline_sha": "abc123",
            "prompt_sha256": "prompt-hash",
            "task_contract": {
                "target_count": 2,
                "acceptance_queries": ["검증 검색어"],
            },
            "foundation": {
                "started_at": "2026-08-15T23:55:00Z",
                "finished_at": "2026-08-16T00:00:00Z",
                "changed_loc": 100,
                "modified_files": ["classic/automation/collection.py"],
                "acceptance_path": "synthetic.json",
            },
            "records": [
                _adaptation_record(
                    "classic",
                    site_specific_changed_loc=120,
                ),
                _adaptation_record(
                    "vision",
                    finished_at="2026-08-16T00:00:30Z",
                    common_runtime_changed_loc=20,
                ),
            ],
        }
    )
    incomplete = SiteAdaptationManifest.model_validate(
        {
            **valid.model_dump(mode="json"),
            "records": [
                _adaptation_record("classic", status="running"),
                _adaptation_record("vision", status="running"),
            ],
        }
    )

    valid_result = evaluate_site_adaptation(valid)["sites"][0]
    incomplete_result = evaluate_site_adaptation(incomplete)["sites"][0]
    assert valid_result["prompt_to_acceptance_sec_saved"] == 60
    assert valid_result["site_specific_changed_loc_saved"] == 120
    assert valid_result["comparison_valid"] is True
    assert incomplete_result["comparison_valid"] is False


def test_site_adaptation_maps_a_substitute_to_its_contract_query():
    manifest = SiteAdaptationManifest.model_validate(
        {
            "baseline_sha": "abc123",
            "prompt_sha256": "prompt-hash",
            "task_contract": {
                "target_count": 2,
                "acceptance_queries": ["프론트엔드 개발자"],
            },
            "foundation": {
                "started_at": "2026-08-15T23:55:00Z",
                "finished_at": "2026-08-16T00:00:00Z",
                "changed_loc": 100,
                "modified_files": ["classic/automation/collection.py"],
                "acceptance_path": "synthetic.json",
            },
            "records": [
                _adaptation_record(
                    approach,
                    acceptance_runs=[
                        {
                            "query": "QA 엔지니어",
                            "contract_query": "프론트엔드 개발자",
                            "substitution_reason": "원 검색어에서 목표 수 미달",
                            "summary_path": "summary.json",
                            "passed": True,
                            "runtime_sec": 10,
                        }
                    ],
                )
                for approach in ("classic", "vision")
            ],
        }
    )

    result = evaluate_site_adaptation(manifest)

    assert result["sites"][0]["comparison_valid"] is True


def test_site_onboarding_acceptance_checks_schema_domain_and_hardcoding():
    summary = _collection(
        items=[{"job_id": 1, "url": "https://jobs.example.com/jobs/1"}]
    )
    jobs = [
        {
            "company_name": "예시회사",
            "position": "백엔드 개발자",
            "url": "https://jobs.example.com/jobs/1",
            "main_tasks": ["API 개발"],
            "requirements": ["Python"],
        }
    ]
    required_fields = [
        "company_name",
        "position",
        "url",
        "main_tasks",
        "requirements",
    ]

    passed = evaluate_site_onboarding_acceptance(
        summary,
        homepage="https://example.com",
        required_fields=required_fields,
        jobs=jobs,
    )
    hardcoded = evaluate_site_onboarding_acceptance(
        summary,
        homepage="https://example.com",
        required_fields=required_fields,
        jobs=jobs,
        patch_text='TARGET = "https://jobs.example.com/jobs/1"',
    )

    assert passed["passed"] is True
    assert hardcoded["passed"] is False
    assert hardcoded["hardcoding_quality"]["matched_literals"]

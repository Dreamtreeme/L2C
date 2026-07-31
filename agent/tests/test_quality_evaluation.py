def test_job_quality_accepts_korean_aliases_and_detects_duplicate_urls():
    from benchmark.quality_eval import evaluate_job_records

    result = evaluate_job_records(
        {
            "공고목록": [
                {"회사명": "A", "직무명": "iOS", "공고url": "https://example.com/1"},
                {"회사명": "B", "직무명": "서버", "공고url": "https://example.com/1?ref=x"},
            ]
        }
    )

    assert result["record_count"] == 2
    assert result["required_field_coverage"] == 1.0
    assert result["unique_url_rate"] == 0.5


def test_job_quality_uses_strict_reference_identity_without_jaccard():
    from benchmark.quality_eval import evaluate_job_records

    reference = [
        {
            "company_name": "VoyagerX",
            "position": "iOS Developer",
            "url": "https://example.com/jobs/1",
            "requirements": ["Swift"],
        }
    ]
    actual = [
        {
            "company_name": "VoyagerX",
            "position": "iOS Engineer",
            "url": "https://example.com/jobs/1?source=test",
            "requirements": ["Swift"],
        }
    ]

    result = evaluate_job_records(actual, reference)

    assert result["reference"]["url_recall"] == 1.0
    assert result["reference"]["identity_exact_rate"] == 0.5
    assert result["reference"]["content_exact_rate"] == 1.0


def test_collection_and_citation_quality_are_separate_metrics():
    from benchmark.quality_eval import (
        evaluate_answer_citations,
        evaluate_collection_summary,
    )

    collection = evaluate_collection_summary(
        {
            "target_count": 2,
            "item_count": 2,
            "persisted_count": 1,
            "review": {"decision": "accept"},
            "is_finished": True,
        }
    )
    citations = evaluate_answer_citations(
        "결과 [job_id:1], 잘못된 값 [job_id:9]",
        valid_ids=[1, 2],
        expected_ids=[1, 2],
    )

    assert collection["target_fulfillment"] == 0.5
    assert collection["persistence_rate"] == 0.5
    assert collection["passed"] is False
    assert citations["citation_validity"] == 0.5
    assert citations["expected_citation_coverage"] == 0.5


def test_collection_quality_counts_existing_database_jobs_as_resolved():
    from benchmark.quality_eval import evaluate_collection_summary

    collection = evaluate_collection_summary(
        {
            "target_count": 2,
            "item_count": 0,
            "persisted_count": 0,
            "observed_job_ids": [7, 8],
            "review": {"decision": "accept"},
            "is_finished": True,
        }
    )

    assert collection["observed_existing_count"] == 2
    assert collection["resolved_count"] == 2
    assert collection["target_fulfillment"] == 1.0
    assert collection["passed"] is True


def test_collection_quality_rejects_search_url_saved_as_saramin_job():
    from benchmark.quality_eval import evaluate_collection_summary

    base = {
        "target_count": 1,
        "item_count": 1,
        "persisted_count": 1,
        "review": {"decision": "accept"},
        "is_finished": True,
    }
    search_url_result = {
        **base,
        "persistence_validation": {
            "persisted_items": [
                {
                    "job_id": 1,
                    "url": "https://www.saramin.co.kr/zf_user/search?searchword=ML",
                }
            ]
        },
    }
    detail_url_result = {
        **base,
        "persistence_validation": {
            "persisted_items": [
                {
                    "job_id": 2,
                    "url": "https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=2",
                }
            ]
        },
    }

    failed = evaluate_collection_summary(search_url_result)
    passed = evaluate_collection_summary(detail_url_result)

    assert failed["source_url_integrity"] == 0.0
    assert failed["passed"] is False
    assert passed["source_url_integrity"] == 1.0
    assert passed["passed"] is True


def test_manual_evaluation_requires_semantic_and_field_judgement():
    from benchmark.manual_evaluation import (
        RunManualJudgement,
        evaluate_manual_run,
    )

    summary = {
        "result": {
            "target_count": 1,
            "item_count": 1,
            "persisted_count": 1,
            "review": {"decision": "accept"},
            "is_finished": True,
        }
    }
    judgement = RunManualJudgement.model_validate(
        {
            "run_id": "wanted-ios-r1",
            "summary_path": "unused.json",
            "site": "wanted",
            "query": "iOS 개발자",
            "search_conditions_correct": True,
            "count_handling_correct": True,
            "no_out_of_scope_actions": True,
            "jobs": [
                {
                    "url": "https://example.com/jobs/1",
                    "semantic_match": False,
                    "company_name": "pass",
                    "job_title": "pass",
                    "responsibilities": "pass",
                    "requirements": "pass",
                }
            ],
        }
    )

    evaluated = evaluate_manual_run(summary, judgement)

    assert evaluated["automatic_contract_passed"] is True
    assert evaluated["manual_contract_passed"] is False
    assert evaluated["outcome"] == "failure"


def test_manual_evaluation_requires_every_resolved_job_to_be_reviewed():
    from benchmark.manual_evaluation import (
        RunManualJudgement,
        evaluate_manual_run,
    )

    summary = {
        "result": {
            "target_count": 2,
            "item_count": 2,
            "persisted_count": 2,
            "review": {"decision": "accept"},
            "is_finished": True,
            "persistence_validation": {
                "persisted_items": [
                    {
                        "job_id": 1,
                        "url": "https://example.com/jobs/1",
                    },
                    {
                        "job_id": 2,
                        "url": "https://example.com/jobs/2",
                    },
                ]
            },
        }
    }
    judgement = RunManualJudgement.model_validate(
        {
            "run_id": "two-jobs",
            "summary_path": "unused.json",
            "site": "example",
            "query": "AI 엔지니어",
            "search_conditions_correct": True,
            "count_handling_correct": True,
            "no_out_of_scope_actions": True,
            "jobs": [
                {
                    "url": "https://example.com/jobs/1",
                    "semantic_match": True,
                    "company_name": "pass",
                    "job_title": "pass",
                    "responsibilities": "pass",
                    "requirements": "pass",
                }
            ],
        }
    )

    evaluated = evaluate_manual_run(summary, judgement)

    assert evaluated["automatic_contract_passed"] is True
    assert evaluated["manual"]["coverage_passed"] is False
    assert evaluated["outcome"] == "failure"


def test_manual_evaluation_uses_partial_only_for_valid_count_shortfall():
    from benchmark.manual_evaluation import (
        RunManualJudgement,
        evaluate_manual_run,
    )

    summary = {
        "result": {
            "target_count": 2,
            "item_count": 1,
            "persisted_count": 1,
            "review": {"decision": "accept"},
            "is_finished": True,
            "persistence_validation": {
                "persisted_items": [
                    {
                        "job_id": 1,
                        "url": "https://example.com/jobs/1",
                    }
                ]
            },
        }
    }
    judgement = RunManualJudgement.model_validate(
        {
            "run_id": "shortfall",
            "summary_path": "unused.json",
            "site": "example",
            "query": "AI 엔지니어",
            "search_conditions_correct": True,
            "count_handling_correct": True,
            "no_out_of_scope_actions": True,
            "jobs": [
                {
                    "url": "https://example.com/jobs/1",
                    "semantic_match": True,
                    "company_name": "pass",
                    "job_title": "pass",
                    "responsibilities": "pass",
                    "requirements": "pass",
                }
            ],
        }
    )

    evaluated = evaluate_manual_run(summary, judgement)

    assert evaluated["automatic_contract_passed"] is False
    assert evaluated["manual_contract_passed"] is True
    assert evaluated["outcome"] == "partial"


def test_site_adaptation_evaluation_keeps_common_runtime_changes_visible():
    from benchmark.site_adaptation_eval import (
        SiteAdaptationManifest,
        evaluate_site_adaptation,
    )

    manifest = SiteAdaptationManifest.model_validate(
        {
            "commit_sha": "abc123",
            "task_contract": {"target_count": 2},
            "records": [
                {
                    "site": "example",
                    "approach": "classic",
                    "implementation_minutes": 90,
                    "site_specific_code_lines": 120,
                    "modified_file_count": 3,
                    "common_runtime_code_lines": 0,
                    "fix_iteration_count": 2,
                    "successful_runs": 3,
                    "runtime_sec": [10, 11, 12],
                },
                {
                    "site": "example",
                    "approach": "vision",
                    "implementation_minutes": 30,
                    "site_specific_code_lines": 0,
                    "modified_file_count": 1,
                    "common_runtime_code_lines": 20,
                    "fix_iteration_count": 1,
                    "successful_runs": 3,
                    "runtime_sec": [90, 95, 100],
                },
            ],
        }
    )

    result = evaluate_site_adaptation(manifest)["sites"][0]

    assert result["implementation_minutes_saved"] == 60
    assert result["vision_profile_only"] is False
    assert result["comparison_valid"] is True


def test_site_adaptation_does_not_claim_profile_only_before_success():
    from benchmark.site_adaptation_eval import (
        SiteAdaptationManifest,
        evaluate_site_adaptation,
    )

    common = {
        "site": "example",
        "implementation_minutes": 10,
        "site_specific_code_lines": 0,
        "modified_file_count": 1,
        "common_runtime_code_lines": 0,
        "fix_iteration_count": 0,
        "successful_runs": 0,
        "attempted_runs": 3,
        "runtime_sec": [],
    }
    manifest = SiteAdaptationManifest.model_validate(
        {
            "commit_sha": "abc123",
            "task_contract": {},
            "records": [
                {**common, "approach": "classic"},
                {**common, "approach": "vision"},
            ],
        }
    )

    result = evaluate_site_adaptation(manifest)["sites"][0]

    assert result["comparison_valid"] is False
    assert result["vision_profile_only"] is False


def test_user_study_evaluation_uses_human_active_time():
    from benchmark.user_study_eval import (
        UserStudyManifest,
        evaluate_user_study,
    )

    common = {
        "participant_id": "P1",
        "task_id": "T1",
        "order": 1,
        "result_review_sec": 10,
        "suitable_job_count": 2,
        "duplicate_count": 0,
        "missing_field_count": 0,
        "factual_error_count": 0,
        "citation_link_rate": 1,
        "usefulness_score": 4,
        "trust_score": 4,
        "correction_count": 0,
    }
    manifest = UserStudyManifest.model_validate(
        {
            "study_contract": {},
            "records": [
                {
                    **common,
                    "mode": "manual",
                    "total_completion_sec": 300,
                    "human_active_sec": 300,
                },
                {
                    **common,
                    "mode": "l2c",
                    "total_completion_sec": 180,
                    "human_active_sec": 60,
                },
            ],
        }
    )

    result = evaluate_user_study(manifest)

    assert result["design_complete"] is True
    assert result["human_activity_reduction_rate"] == 0.8


def test_user_study_hides_efficiency_for_incomplete_crossover():
    from benchmark.user_study_eval import (
        UserStudyManifest,
        evaluate_user_study,
    )

    base = {
        "participant_id": "P1",
        "order": 1,
        "total_completion_sec": 100,
        "human_active_sec": 80,
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
    manifest = UserStudyManifest.model_validate(
        {
            "study_contract": {
                "participants": 3,
                "tasks": ["T1", "T2"],
            },
            "records": [
                {**base, "task_id": "T1", "mode": "manual"},
                {**base, "task_id": "T2", "mode": "l2c"},
            ],
        }
    )

    result = evaluate_user_study(manifest)

    assert result["design_complete"] is False
    assert result["human_activity_reduction_rate"] is None

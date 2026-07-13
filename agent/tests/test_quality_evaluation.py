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

    assert collection["target_fulfillment"] == 1.0
    assert collection["persistence_rate"] == 0.5
    assert collection["passed"] is False
    assert citations["citation_validity"] == 0.5
    assert citations["expected_citation_coverage"] == 0.5

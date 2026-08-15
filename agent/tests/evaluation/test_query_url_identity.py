from benchmark.quality_eval import evaluate_job_records


def test_job_quality_preserves_query_based_record_identity():
    result = evaluate_job_records(
        {
            "jobs": [
                {
                    "company_name": "A",
                    "position": "서버",
                    "url": "https://example.com/job?posting=first&src=search",
                },
                {
                    "company_name": "B",
                    "position": "데이터",
                    "url": "https://example.com/job?src=search&posting=second",
                },
            ]
        }
    )

    assert result["unique_url_rate"] == 1.0

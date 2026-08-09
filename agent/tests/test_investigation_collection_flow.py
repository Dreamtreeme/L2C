"""조사 그래프의 수집·저장 결과 계약을 검증한다."""

import pytest

from agent.graph.investigation_collection_nodes import build_collection_result
from shared.schema.collection_intent import CollectionIntent
from shared.schema.collection_run import CollectionBatch, PersistedCollection
from shared.schema.feedback_schema import WorkerSubmission


@pytest.mark.parametrize(
    ("worker", "persistence", "target", "status", "resolved", "exhausted"),
    [
        (
            {"observed_job_ids": [], "is_finished": True},
            {"persisted_count": 1, "persisted_items": [{"job_id": 1}]},
            2,
            "partial",
            1,
            False,
        ),
        (
            {"observed_job_ids": [7, 8], "is_finished": True},
            {"persisted_count": 0, "persisted_items": []},
            2,
            "completed",
            2,
            False,
        ),
        (
            {"observed_job_ids": [], "is_finished": False},
            {"persisted_count": 1, "persisted_items": [{"job_id": 7}]},
            10,
            "completed",
            1,
            True,
        ),
    ],
)
def test_collection_result_combines_worker_observation_and_persistence(
    worker,
    persistence,
    target,
    status,
    resolved,
    exhausted,
):
    summary = (
        {
            "job_results_availability": {
                "available_job_count": 1,
                "count_evidence": "포지션 1",
                "count_confidence": 0.97,
            }
        }
        if exhausted
        else {}
    )
    submission = WorkerSubmission(
        run_id="worker-1",
        is_finished=worker["is_finished"],
        hit_recursion_limit=not worker["is_finished"],
        collected_count=persistence["persisted_count"],
        observed_job_ids=worker["observed_job_ids"],
        extracted_summary=summary,
    )
    batch = CollectionBatch(
        submission=submission,
        site_name="Wanted",
        site_slug="wanted",
    )
    persisted = PersistedCollection(
        submission=submission,
        submission_id="submission-1",
        persistence={**persistence, "rejected_count": 0},
    )
    intent = CollectionIntent(
        site="wanted",
        search_keyword="iOS 개발자",
        target_count=target,
    )

    result = build_collection_result(intent, batch, persisted)

    assert result.status == status
    assert result.resolved_count == resolved
    assert result.scope_exhausted is exhausted

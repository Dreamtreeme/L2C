"""조사 그래프의 수집·저장 결과 계약을 검증한다."""

import pytest

from agent.graph.investigation_collection_nodes import (
    InvestigationCollectionNodes,
    build_collection_result,
)
from agent.graph.investigation_context import create_investigation_state
from shared.schema.collection_intent import CollectionIntent
from shared.schema.collection_run import (
    CollectionBatch,
    CollectionExperienceResult,
    PersistenceReport,
    PostprocessedCollection,
    RecipeLearningResult,
)
from shared.schema.feedback_schema import WorkerSubmission
from shared.schema.investigation_schema import InvestigationPlanStep, InvestigationRequest


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
    intent = CollectionIntent(
        site="wanted",
        search_keyword="iOS 개발자",
        target_count=target,
    )
    submission = WorkerSubmission(
        run_id="worker-1",
        is_finished=worker["is_finished"],
        hit_recursion_limit=not worker["is_finished"],
        collected_count=persistence["persisted_count"],
        observed_job_ids=worker["observed_job_ids"],
        extracted_summary=summary,
        collection_intent=intent,
    )
    batch = CollectionBatch(
        submission=submission,
        site_name="Wanted",
    )
    experience = CollectionExperienceResult(
        submission_id="submission-1",
        recipe_learning=RecipeLearningResult(status="not_eligible"),
    )
    report = PersistenceReport(persisted_items=persistence["persisted_items"])
    result = build_collection_result(batch, report, experience)

    assert result.status == status
    assert result.resolved_count == resolved
    assert result.scope_exhausted is exhausted


def _collection_state():
    state = create_investigation_state(
        InvestigationRequest(
            investigation_id="investigation-1",
            original_query="iOS 개발자 공고 수집",
        )
    )
    state["execution"]["plan"] = [
        InvestigationPlanStep(
            step_id="collect-1",
            tool_name="realtime_scraping:wanted",
            arguments=CollectionIntent(
                site="wanted",
                search_keyword="iOS 개발자",
                target_count=1,
            ),
        )
    ]
    return state


def _raise(message):
    def fail(*_args, **_kwargs):
        raise RuntimeError(message)

    return fail


def test_collection_node_routes_worker_failure_to_evidence_inspection():
    nodes = InvestigationCollectionNodes(
        _raise("worker failed"),
        _raise("unused"),
        _raise("unused"),
        _raise("unused"),
    )
    state = _collection_state()

    update = nodes.collect(state)
    state["execution"].update(update["execution"])

    result = state["execution"]["collection_results"][0]
    assert result.error_code == "collection_worker_failed"
    assert nodes.route_after_collect(state) == "inspect_evidence"


def test_collection_node_routes_postprocessing_failure_to_evidence_inspection():
    state = _collection_state()
    batch = CollectionBatch(
        submission=WorkerSubmission(
            run_id="worker-1",
            collection_intent=state["execution"]["plan"][0].arguments,
        ),
        site_name="Wanted",
    )
    state["execution"]["pending_collection"] = batch
    nodes = InvestigationCollectionNodes(
        lambda _intent: batch,
        _raise("postprocess failed"),
        _raise("unused"),
        _raise("unused"),
    )

    update = nodes.postprocess(state)
    state["execution"].update(update["execution"])

    result = state["execution"]["collection_results"][0]
    assert result.error_code == "postprocessing_error:RuntimeError"
    assert nodes.route_after_postprocess(state) == "inspect_evidence"


def test_collection_node_reports_persistence_failure():
    state = _collection_state()
    batch = CollectionBatch(
        submission=WorkerSubmission(
            run_id="worker-1",
            collection_intent=state["execution"]["plan"][0].arguments,
        ),
        site_name="Wanted",
    )
    processed = PostprocessedCollection(
        submission=batch.submission,
        site_name=batch.site_name,
    )
    state["execution"].update(
        pending_collection=batch,
        postprocessed_collection=processed,
    )
    nodes = InvestigationCollectionNodes(
        lambda _intent: batch,
        lambda _batch: processed,
        _raise("persistence failed"),
        _raise("unused"),
    )

    update = nodes.persist(state)

    result = update["execution"]["collection_results"][0]
    assert result.error_code == "persistence_error:RuntimeError"


def test_collection_node_keeps_saved_result_when_experience_recording_fails():
    state = _collection_state()
    batch = CollectionBatch(
        submission=WorkerSubmission(
            run_id="worker-1",
            run_status="finished",
            is_finished=True,
            observed_job_ids=[7],
            collection_intent=state["execution"]["plan"][0].arguments,
        ),
        site_name="Wanted",
    )
    processed = PostprocessedCollection(
        submission=batch.submission,
        site_name=batch.site_name,
    )
    state["execution"].update(
        pending_collection=batch,
        postprocessed_collection=processed,
    )
    nodes = InvestigationCollectionNodes(
        lambda _intent: batch,
        lambda _batch: processed,
        lambda _processed: PersistenceReport(),
        _raise("experience failed"),
    )

    update = nodes.persist(state)

    result = update["execution"]["collection_results"][0]
    assert result.status == "completed"
    assert result.document_ids == [7]
    assert result.submission_id == ""

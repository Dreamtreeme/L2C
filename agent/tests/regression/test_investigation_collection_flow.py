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
    RecipeLearningResult,
)
from shared.schema.feedback_schema import WorkerSubmission
from shared.schema.investigation_schema import InvestigationPlanStep, InvestigationRequest


@pytest.mark.parametrize(
    ("worker", "persistence", "target", "status", "resolved", "exhausted"),
    [
        (
            {"observed_job_ids": [], "run_status": "finished"},
            {"persisted_count": 1, "persisted_items": [{"job_id": 1}]},
            2,
            "partial",
            1,
            False,
        ),
        (
            {"observed_job_ids": [7, 8], "run_status": "finished"},
            {"persisted_count": 0, "persisted_items": []},
            2,
            "completed",
            2,
            False,
        ),
        (
            {"observed_job_ids": [], "run_status": "recursion_limit"},
            {"persisted_count": 1, "persisted_items": [{"job_id": 7}]},
            10,
            "failed",
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
        run_status=worker["run_status"],
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
        recipe_learning=RecipeLearningResult(status="not_eligible"),
    )
    report = PersistenceReport(persisted_items=persistence["persisted_items"])
    result = build_collection_result(batch, report, experience)

    assert result.status == status
    assert result.resolved_count == resolved
    assert result.scope_exhausted is exhausted
    assert result.execution_status == (
        "failed" if worker["run_status"] == "recursion_limit" else "completed"
    )


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
            expected_evidence=["jobs"],
        )
    ]
    return state


def _raise(message):
    def fail(*_args, **_kwargs):
        raise RuntimeError(message)

    return fail


def test_collection_node_routes_worker_failure_to_next_plan():
    nodes = InvestigationCollectionNodes(
        _raise("worker failed"),
        _raise("unused"),
        _raise("unused"),
    )
    state = _collection_state()
    state["execution"]["plan"].append(
        InvestigationPlanStep(
            step_id="collect-2",
            tool_name="realtime_scraping:wanted",
            arguments=CollectionIntent(
                site="wanted",
                search_keyword="서버 개발자",
                target_count=1,
            ),
            expected_evidence=["jobs"],
        )
    )

    update = nodes.collect(state)
    state["execution"].update(update["execution"])

    result = state["execution"]["collection_results"][0]
    assert result.error_code == "collection_worker_failed"
    assert nodes.route_after_collect(state) == "collect"


def test_collection_node_routes_final_worker_failure_to_answer():
    nodes = InvestigationCollectionNodes(
        _raise("worker failed"),
        _raise("unused"),
        _raise("unused"),
    )
    state = _collection_state()

    update = nodes.collect(state)
    state["execution"].update(update["execution"])

    assert nodes.route_after_collect(state) == "answer"

def test_collection_node_reports_persistence_failure():
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
            observed_job_ids=[7],
            collection_intent=state["execution"]["plan"][0].arguments,
        ),
        site_name="Wanted",
    )
    state["execution"]["pending_collection"] = batch
    nodes = InvestigationCollectionNodes(
        lambda _intent: batch,
        lambda _processed: PersistenceReport(),
        _raise("experience failed"),
    )

    update = nodes.persist(state)

    result = update["execution"]["collection_results"][0]
    assert result.status == "completed"
    assert result.document_ids == [7]
    assert result.worker_run_id == "worker-1"

    state["execution"].update(update["execution"])
    assert nodes.route_after_persist(state) == "inspect_evidence"


def test_empty_collection_batch_is_persisted_without_rechecking_same_database():
    state = _collection_state()
    batch = CollectionBatch(
        submission=WorkerSubmission(
            run_id="worker-empty",
            run_status="finished",
            collection_intent=state["execution"]["plan"][0].arguments,
        ),
        site_name="Wanted",
    )
    state["execution"]["pending_collection"] = batch
    nodes = InvestigationCollectionNodes(
        lambda _intent: batch,
        lambda _processed: PersistenceReport(),
        lambda _batch, _report: CollectionExperienceResult(
            recipe_learning=RecipeLearningResult(status="not_eligible")
        ),
    )

    update = nodes.persist(state)
    state["execution"].update(update["execution"])

    assert state["execution"]["collection_results"][-1].document_ids == []
    assert nodes.route_after_persist(state) == "answer"

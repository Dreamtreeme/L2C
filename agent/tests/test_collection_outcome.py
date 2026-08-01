from agent.application.collection_outcome import (
    build_collection_outcome,
    collection_run_status,
)
from agent.application.run_contracts import RunStatus


def test_collection_outcome_keeps_stage_results_separate():
    outcome = build_collection_outcome(
        is_finished=False,
        hit_recursion_limit=True,
        review={"decision": "revise"},
        persisted_count=1,
        resolved_count=1,
        rejected_count=1,
        target_count=2,
        scope_exhausted=False,
    )

    assert outcome.as_dict() == {
        "worker_status": "limit_reached",
        "review_status": "revision_required",
        "persistence_status": "partial",
        "target_status": "unmet",
        "completion_status": "partial",
    }


def test_scope_exhaustion_is_complete_without_claiming_target_met():
    outcome = build_collection_outcome(
        is_finished=False,
        hit_recursion_limit=True,
        review={"decision": "revise"},
        persisted_count=1,
        resolved_count=1,
        rejected_count=0,
        target_count=10,
        scope_exhausted=True,
    )

    assert outcome.target_status.value == "scope_exhausted"
    assert outcome.completion_status.value == "complete"


def test_collection_run_status_uses_structured_completion_fields():
    assert collection_run_status("complete") == RunStatus.COMPLETED
    assert collection_run_status("partial") == RunStatus.PARTIAL
    assert collection_run_status("rejected") == RunStatus.FAILED
    assert collection_run_status("unknown") == RunStatus.FAILED
    assert collection_run_status(
        "partial",
        needs_human_approval=True,
    ) == RunStatus.WAITING_APPROVAL


def test_rejected_review_cannot_be_reported_as_complete():
    outcome = build_collection_outcome(
        is_finished=True,
        hit_recursion_limit=False,
        review={"decision": "reject"},
        persisted_count=0,
        resolved_count=2,
        rejected_count=0,
        target_count=2,
        scope_exhausted=False,
    )

    assert outcome.review_status.value == "rejected"
    assert outcome.completion_status.value == "rejected"

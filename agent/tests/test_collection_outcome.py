from agent.application.collection_outcome import (
    build_collection_outcome,
)


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

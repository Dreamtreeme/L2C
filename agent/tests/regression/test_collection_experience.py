"""공고 저장과 분리된 실행 기록·레시피 후보 경계를 검증한다."""

import pytest

from agent.application import collection_experience as service
from shared.schema.collection_intent import CollectionIntent
from shared.schema.collection_run import CollectionBatch, PersistenceReport
from shared.schema.feedback_schema import WorkerSubmission


def _batch(*, finished: bool = True) -> CollectionBatch:
    return CollectionBatch(
        submission=WorkerSubmission(
            run_id="worker-1",
            run_status="finished" if finished else "recursion_limit",
            action_events=[
                {
                    "seq": 1,
                    "candidate_action": {
                        "source_seq": 1,
                        "action": "click_marker",
                    },
                    "before_checkpoint": {"observation_id": "observation:1"},
                    "transition": {
                        "seq": 1,
                        "before": {"observation_id": "observation:1"},
                        "actions": [
                            {"source_seq": 1, "action": "click_marker"}
                        ],
                        "after": {"observation_id": "observation:2"},
                        "evidence": {
                            "source": "autonomous",
                            "result_status": "success",
                            "status": "ready",
                        },
                    },
                }
            ],
            collection_intent=CollectionIntent(site="wanted"),
        ),
        site_name="Wanted",
    )


@pytest.fixture
def submission_commit(monkeypatch):
    commits = []
    monkeypatch.setattr(
        "agent.recipe.submission_store.SubmissionStore.commit_submission",
        lambda self, submission, **kwargs: (
            commits.append((submission, kwargs)) or "worker-1"
        ),
    )
    return commits


def test_experience_records_submission_and_reusable_candidate(
    monkeypatch,
    submission_commit,
    tmp_path,
):
    monkeypatch.setattr(
        "agent.recipe.candidate_store.RecipeCandidateStore.commit_candidate",
        lambda self, *args, **kwargs: "worker-1",
    )
    monkeypatch.setattr(
        "agent.recipe.candidate_store.RecipeCandidateStore.enqueue_review",
        lambda self, run_id: True,
    )
    persistence = PersistenceReport(
        persisted_items=[{"job_id": 1, "operation": "created"}]
    )

    result = service.record_collection_experience(
        _batch(),
        persistence,
        db_path=tmp_path / "jobs.db",
    )

    assert submission_commit[0][0].persisted_count == 1
    assert result.recipe_learning.status == "queued"


def test_incomplete_run_is_not_recipe_candidate(
    monkeypatch,
    submission_commit,
    tmp_path,
):
    monkeypatch.setattr(
        "agent.recipe.candidate_store.RecipeCandidateStore.commit_candidate",
        lambda *args, **kwargs: pytest.fail("후보가 생성되면 안 됩니다."),
    )

    result = service.record_collection_experience(
        _batch(finished=False),
        PersistenceReport(persisted_items=[{"job_id": 1}]),
        db_path=tmp_path / "jobs.db",
    )

    assert result.recipe_learning.status == "not_eligible"


def test_submission_failure_is_returned_without_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent.recipe.submission_store.SubmissionStore.commit_submission",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("db unavailable")),
    )

    result = service.record_collection_experience(
        _batch(),
        PersistenceReport(),
        db_path=tmp_path / "jobs.db",
    )

    assert result.recipe_learning.reason == "submission_registration_failed"


def test_candidate_storage_failure_does_not_fail_submission(
    monkeypatch,
    submission_commit,
    tmp_path,
):
    monkeypatch.setattr(
        "agent.recipe.candidate_store.RecipeCandidateStore.commit_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("db unavailable")),
    )

    result = service.record_collection_experience(
        _batch(),
        PersistenceReport(persisted_items=[{"job_id": 1}]),
        db_path=tmp_path / "jobs.db",
    )

    assert result.recipe_learning.status == "failed"
    assert result.recipe_learning.reason == "candidate_registration_failed"
    assert "RuntimeError" in result.recipe_learning.error

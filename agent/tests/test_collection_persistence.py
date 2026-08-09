import pytest

from agent.application import collection_persistence as service
from shared.schema.feedback_schema import WorkerSubmission
from shared.schema.jd_schema import CollectedJob, JobPosting
from shared.schema.collection_run import CollectionBatch


def _worker_result(*, finished: bool = True) -> CollectionBatch:
    return CollectionBatch(
        submission=WorkerSubmission(
            run_id="worker-1",
            run_status="finished" if finished else "recursion_limit",
            is_finished=finished,
            hit_recursion_limit=not finished,
            recorded_steps=[{"seq": 1, "action": "click_marker"}],
            collection_intent={
                "site": "wanted",
                "search_keyword": "개발자",
                "task_category": "검색",
            },
        ),
        collected_jobs=[
            CollectedJob(
                posting=JobPosting(
                    company_name="ABC",
                    position="개발자",
                    url="https://example.com/jobs/1",
                )
            )
        ],
        site_slug="wanted",
        site_name="Wanted",
    )


@pytest.fixture
def submission_dependencies(monkeypatch):
    commits = []
    monkeypatch.setattr(
        "agent.recipe.submission_store.SubmissionStore.commit_submission",
        lambda self, submission, **kwargs: (
            commits.append((submission.copy(), kwargs)) or "submission-1"
        ),
    )
    monkeypatch.setattr(
        service,
        "persist_collected_jobs_with_report",
        lambda *args, **kwargs: {
            "persisted_count": 1,
            "created_count": 1,
            "updated_count": 0,
            "rejected_count": 0,
            "persisted_items": [{"job_id": 1}],
        },
    )
    return commits


def test_finalize_stores_submission_once_without_persistence_copy(
    monkeypatch,
    submission_dependencies,
):
    monkeypatch.setattr(
        "agent.recipe.candidate_store.RecipeCandidateStore.commit_candidate",
        lambda self, *args, **kwargs: "candidate-1",
    )
    monkeypatch.setattr(
        service,
        "schedule_recipe_candidate_promotion",
        lambda _candidate_id: True,
    )

    result = service.persist_collection_batch(_worker_result())

    assert result.submission_id == "submission-1"
    assert result.persistence["persisted_count"] == 1
    assert result.submission.persisted_count == 1
    assert len(submission_dependencies) == 1
    assert "persistence_validation" not in submission_dependencies[0][0]
    assert result.recipe_learning == {
        "status": "queued",
        "candidate_id": "candidate-1",
        "reason": "",
        "error": "",
    }


@pytest.mark.parametrize(
    ("worker_result", "persistence", "expected"),
    [
        (
            _worker_result(finished=False),
            {"persisted_count": 1, "rejected_count": 0},
            "not_eligible",
        ),
        (_worker_result(), {"persisted_count": 0, "rejected_count": 1}, "not_eligible"),
    ],
)
def test_incomplete_or_rejected_run_does_not_create_candidate(
    monkeypatch,
    submission_dependencies,
    worker_result,
    persistence,
    expected,
):
    monkeypatch.setattr(
        service,
        "persist_collected_jobs_with_report",
        lambda *args, **kwargs: persistence,
    )
    monkeypatch.setattr(
        "agent.recipe.candidate_store.RecipeCandidateStore.commit_candidate",
        lambda *args, **kwargs: pytest.fail("후보가 생성되면 안 됩니다."),
    )

    result = service.persist_collection_batch(worker_result)

    assert result.recipe_learning["status"] == expected


def test_candidate_storage_failure_is_reported(monkeypatch, submission_dependencies):
    def fail_candidate(*_args, **_kwargs):
        raise RuntimeError("candidate storage unavailable")

    monkeypatch.setattr(
        "agent.recipe.candidate_store.RecipeCandidateStore.commit_candidate",
        fail_candidate,
    )

    result = service.persist_collection_batch(_worker_result())

    assert result.recipe_learning["status"] == "failed"
    assert result.recipe_learning["reason"] == "candidate_registration_failed"
    assert "RuntimeError" in result.recipe_learning["error"]


def test_submission_store_has_one_lookup_contract(tmp_path):
    from agent.observability.worker_trace_report import build_worker_trace
    from agent.recipe.submission_store import SubmissionStore

    store = SubmissionStore(tmp_path / "submissions.db")
    submission_id = store.commit_submission(
        {
            "run_id": "worker-1",
            "collection_intent": {"site": "wanted"},
            "recorded_steps": [],
        }
    )

    found = store.find_submission(run_id="worker-1")
    assert found == {
        "submission_id": submission_id,
        "run_id": "worker-1",
        "source": "vision_worker",
        "payload": {
            "collection_intent": {"site": "wanted"},
            "recorded_steps": [],
            "run_id": "worker-1",
        },
    }
    assert build_worker_trace(found)["site"] == "wanted"

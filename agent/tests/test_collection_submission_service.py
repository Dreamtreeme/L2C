from types import SimpleNamespace

from agent.application import collection_submission_service as submission_service


def _worker_result(*, finished: bool = True) -> dict:
    return {
        "submission": {
            "run_id": "worker-1",
            "run_status": "finished" if finished else "recursion_limit",
            "is_finished": finished,
            "hit_recursion_limit": not finished,
        },
        "extracted_jd": {"jobs": [{"company": "ABC", "position": "개발자"}]},
        "keyword": "개발자",
        "collection_intent": {"search_keyword": "개발자"},
    }


def _patch_submission_store(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.recipe.submission_store.SubmissionStore.commit_submission",
        lambda self, submission, **kwargs: "submission-1",
    )


def test_persist_accepted_worker_result_rejects_failed_validation(monkeypatch):
    _patch_submission_store(monkeypatch)
    monkeypatch.setattr(
        submission_service,
        "persist_collected_data_with_report",
        lambda *args, **kwargs: {
            "persisted_count": 0,
            "rejected_count": 1,
            "rejected_items": [{"reason": "missing identity"}],
        },
    )

    persisted, submission, review, submission_id = (
        submission_service.persist_accepted_worker_result(
            _worker_result(),
            {
                "decision": "accept",
                "recipe_candidate": True,
            },
        )
    )

    assert persisted == 0
    assert submission_id == "submission-1"
    assert submission["persistence_validation"]["rejected_count"] == 1
    assert review["decision"] == "reject"
    assert review["recipe_candidate"] is False


def test_persist_accepted_worker_result_promotes_only_complete_run(monkeypatch):
    _patch_submission_store(monkeypatch)
    promoted: list[str] = []
    monkeypatch.setattr(
        submission_service,
        "persist_collected_data_with_report",
        lambda *args, **kwargs: {
            "persisted_count": 1,
            "rejected_count": 0,
            "persisted_items": [{"job_id": 1}],
        },
    )
    monkeypatch.setattr(
        submission_service,
        "get_settings",
        lambda: SimpleNamespace(
            recipe=SimpleNamespace(learning_mode="record")
        ),
    )
    monkeypatch.setattr(
        "agent.recipe.candidate_store.RecipeCandidateStore.commit_candidate",
        lambda self, *args, **kwargs: "candidate-1",
    )
    monkeypatch.setattr(
        "agent.application.recipe_promotion_service."
        "schedule_recipe_candidate_promotion",
        lambda candidate_id: promoted.append(candidate_id) or True,
    )

    complete_result = _worker_result()
    persisted, submission, review, _ = (
        submission_service.persist_accepted_worker_result(
            complete_result,
            {
                "decision": "accept",
                "recipe_candidate": True,
            },
        )
    )

    assert persisted == 1
    assert review["decision"] == "accept"
    assert submission["recipe_candidate_id"] == "candidate-1"
    assert promoted == ["candidate-1"]

    promoted.clear()
    incomplete_result = _worker_result(finished=False)
    _, incomplete_submission, _, _ = (
        submission_service.persist_accepted_worker_result(
            incomplete_result,
            {
                "decision": "accept",
                "recipe_candidate": True,
            },
        )
    )

    assert "recipe_candidate_id" not in incomplete_submission
    assert promoted == []

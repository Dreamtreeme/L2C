"""후처리 결과의 SQLite 저장 경계를 검증한다."""

from agent.application.collection_storage import store_postprocessed_collection
from shared.db.database import Database
from shared.schema.collection_run import PostprocessedCollection
from shared.schema.feedback_schema import WorkerSubmission
from shared.schema.jd_schema import CollectedJob, JobCollectionEvidence, JobPosting


def _processed() -> PostprocessedCollection:
    return PostprocessedCollection(
        submission=WorkerSubmission(run_id="worker-1"),
        collected_jobs=[
            CollectedJob(
                posting=JobPosting(
                    company_name="예시회사",
                    position="AI 엔지니어",
                    url="https://example.com/jobs/1",
                    requirements=["Python"],
                    raw_ocr_text="원문",
                ),
                evidence=JobCollectionEvidence(
                    required_fields=["company_name", "position", "url"],
                    screenshot_path="detail.png",
                ),
            )
        ],
        site_name="Example",
    )


def test_storage_upserts_validated_job_and_preserves_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent.application.collection_storage.JobTaxonomyLinker.link_job",
        lambda self, job_id: None,
    )
    db_path = tmp_path / "jobs.db"

    first = store_postprocessed_collection(_processed(), db_path=db_path)
    second = store_postprocessed_collection(_processed(), db_path=db_path)
    saved = Database(db_path).get(first.persisted_items[0]["job_id"])

    assert first.persisted_items[0]["operation"] == "created"
    assert second.persisted_items[0]["operation"] == "updated"
    assert saved["raw_ocr_text"] == "원문"
    assert saved["screenshot_path"] == "detail.png"


def test_taxonomy_failure_excludes_job_from_answer_ready_report(monkeypatch, tmp_path):
    def fail_link(_self, _job_id):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(
        "agent.application.collection_storage.JobTaxonomyLinker.link_job",
        fail_link,
    )

    db_path = tmp_path / "jobs.db"
    result = store_postprocessed_collection(_processed(), db_path=db_path)

    assert result.persisted_count == 0
    assert result.rejected_count == 1
    assert result.rejected_items[0]["issues"] == [
        "taxonomy_index_failed:RuntimeError"
    ]
    assert Database(db_path).get_by_url("https://example.com/jobs/1") is not None


def test_storage_failure_isolated_per_job(monkeypatch, tmp_path):
    collection = _processed()
    original = collection.collected_jobs[0]
    collection.collected_jobs.append(
        original.model_copy(
            update={
                "posting": original.posting.model_copy(
                    update={"url": "https://example.com/jobs/2"}
                )
            }
        )
    )
    real_upsert = Database.upsert

    def fail_first(self, posting, *, evidence=None):
        if str(posting.url).endswith("/1"):
            raise RuntimeError("write failed")
        return real_upsert(self, posting, evidence=evidence)

    monkeypatch.setattr(Database, "upsert", fail_first)
    monkeypatch.setattr(
        "agent.application.collection_storage.JobTaxonomyLinker.link_job",
        lambda self, job_id: None,
    )

    result = store_postprocessed_collection(
        collection,
        db_path=tmp_path / "jobs.db",
    )

    assert result.persisted_count == 1
    assert result.persisted_items[0]["url"].endswith("/2")
    assert result.rejected_count == 1
    assert result.rejected_items[0]["index"] == 0
    assert result.rejected_items[0]["issues"] == ["persistence_error:RuntimeError"]

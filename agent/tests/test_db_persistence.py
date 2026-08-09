"""공고 스키마와 SQLite UPSERT·버전 저장 계약을 검증한다."""

import pytest
from pydantic import ValidationError

from agent.application.job_normalization_service import complete_extracted_job
from shared.db.database import Database
from shared.schema.jd_schema import CollectedJob, JobCollectionEvidence, JobPosting


def test_job_posting_rejects_internal_field_aliases():
    with pytest.raises(ValidationError):
        JobPosting.model_validate(
            {
                "회사명": "테스트컴퍼니",
                "직무명": "데이터 엔지니어",
                "url": "https://example.com/jobs/1",
            }
        )


def test_extracted_job_is_completed_once_with_source_and_hash():
    posting = complete_extracted_job(
        JobPosting(
            company_name="Acme",
            position="AI Engineer",
            requirements=["Python"],
        ),
        current_url="https://www.wanted.co.kr/wd/123",
        raw_ocr_text="실제 OCR 원문",
    )

    assert posting.url == "https://www.wanted.co.kr/wd/123"
    assert posting.source_platform == "Wanted"
    assert posting.raw_ocr_text == "실제 OCR 원문"
    assert posting.content_hash


def test_collected_job_rejects_unresolved_required_field():
    with pytest.raises(ValidationError, match="main_tasks"):
        CollectedJob(
            posting=JobPosting(
                company_name="예시회사",
                position="데이터 엔지니어",
                url="https://example.com/jobs/incomplete",
            ),
            evidence=JobCollectionEvidence(
                required_fields=[
                    "company_name",
                    "position",
                    "url",
                    "main_tasks",
                ]
            ),
        )


def test_collected_job_accepts_confirmed_unavailable_field_at_page_end():
    item = CollectedJob(
        posting=JobPosting(
            company_name="예시회사",
            position="데이터 엔지니어",
            url="https://example.com/jobs/no-benefits",
        ),
        evidence=JobCollectionEvidence(
            required_fields=["company_name", "position", "url", "benefits"],
            unavailable_fields=["benefits"],
            page_exhausted=True,
        ),
    )

    assert [field.value for field in item.evidence.unavailable_fields] == ["benefits"]


def test_database_preserves_same_content_at_different_urls(tmp_path):
    db = Database(tmp_path / "same-content.db")
    first = complete_extracted_job(
        JobPosting(
            company_name="Acme",
            position="iOS Engineer",
            requirements=["Swift"],
        ),
        current_url="https://example.com/jobs/1",
        raw_ocr_text="Swift",
    )
    second = first.model_copy(update={"url": "https://example.com/jobs/2"})

    first_id = db.upsert(first)
    second_id = db.upsert(second)

    assert first_id != second_id
    assert db.get(first_id)["url"] == first.url
    assert db.get(second_id)["url"] == second.url


def test_job_posting_keeps_only_iso_posted_date():
    posting = JobPosting(posted_at="3일 전", posted_at_text="3일 전")

    assert posting.posted_at is None
    assert posting.posted_at_text == "3일 전"


def test_database_records_only_meaningful_job_versions(tmp_path):
    db = Database(tmp_path / "versioned_jobs.db")
    first = JobPosting(
        url="https://example.com/jobs/versioned",
        company_name="Acme",
        position="Data Engineer",
        requirements=["Python"],
        benefits=["장비 지원"],
        raw_ocr_text="첫 번째 공고 원문",
        source_platform="Example",
    )

    job_id = db.upsert(first)
    db.upsert(first)
    db.upsert(
        first.model_copy(
            update={
                "benefits": ["장비 지원", "재택근무"],
                "raw_ocr_text": "변경된 공고 원문",
            }
        )
    )
    versions = db.list_versions(job_id)

    assert [item["version_number"] for item in versions] == [2, 1]
    assert {"benefits", "raw_ocr_text"} <= set(versions[0]["changed_fields"])


def test_database_preserves_collection_evidence_outside_posting(tmp_path):
    db = Database(tmp_path / "evidence_jobs.db")
    posting = JobPosting(
        url="https://example.com/jobs/evidence",
        company_name="증거회사",
        position="iOS 개발자",
        raw_ocr_text="실제 누적 OCR 원문",
    )
    evidence = JobCollectionEvidence(
        required_fields=["company_name", "position", "url"],
        screenshot_path="detail.png",
        field_evidence={"requirements": "Swift"},
    )

    job_id = db.upsert(posting, evidence=evidence)
    saved = db.get(job_id)

    assert saved["raw_ocr_text"] == "실제 누적 OCR 원문"
    assert saved["screenshot_path"] == "detail.png"
    assert saved["raw_json"].get("field_evidence") is None

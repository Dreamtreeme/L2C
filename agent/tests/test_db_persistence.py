"""정규 공고 계약과 SQLite 저장 경계를 검증한다."""

from __future__ import annotations

import sqlite3

import pytest
from pydantic import ValidationError

from agent.application.job_normalization_service import complete_extracted_job
from agent.application.job_persistence_service import persist_collected_jobs_with_report
from agent.config import get_settings
from agent.runtime.job_identity import source_card_key
from shared.db.database import Database
from shared.schema.collection_intent import CollectionIntent, JobSearchFilters
from shared.schema.jd_schema import CollectedJob, JobCollectionEvidence, JobPosting


def collected_job(
    *,
    url: str,
    company_name: str = "예시회사",
    position: str = "데이터 엔지니어",
    requirements: list[str] | None = None,
    evidence: JobCollectionEvidence | None = None,
    **fields,
) -> CollectedJob:
    return CollectedJob(
        posting=JobPosting(
            url=url,
            company_name=company_name,
            position=position,
            requirements=requirements or ["Python"],
            **fields,
        ),
        evidence=evidence
        or JobCollectionEvidence(
            required_fields=["company_name", "position", "url"]
        ),
    )


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
        collected_job(
            url="https://example.com/jobs/incomplete",
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
    item = collected_job(
        url="https://example.com/jobs/no-benefits",
        evidence=JobCollectionEvidence(
            required_fields=["company_name", "position", "url", "benefits"],
            unavailable_fields=["benefits"],
            page_exhausted=True,
        ),
    )

    assert [field.value for field in item.evidence.unavailable_fields] == [
        "benefits"
    ]


def test_database_preserves_same_content_at_different_urls(tmp_path):
    db = Database(tmp_path / "same-content.db")
    first = complete_extracted_job(
        JobPosting(company_name="Acme", position="iOS Engineer", requirements=["Swift"]),
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
    assert versions[0]["content"]["benefits"] == ["장비 지원", "재택근무"]


def test_persistence_keeps_collection_evidence_outside_posting(monkeypatch, tmp_path):
    db_path = tmp_path / "evidence_jobs.db"
    monkeypatch.setattr(get_settings().paths, "db_path", db_path)
    screenshot_path = str(tmp_path / "detail.png")
    item = collected_job(
        url="https://www.jobkorea.co.kr/Recruit/GI_Read/50000001",
        company_name="증거회사",
        position="iOS 개발자",
        raw_ocr_text="실제 누적 OCR 원문",
        source_platform="JobKorea",
        evidence=JobCollectionEvidence(
            required_fields=["company_name", "position", "url"],
            screenshot_path=screenshot_path,
            field_evidence={"requirements": "Swift"},
        ),
    )

    result = persist_collected_jobs_with_report(
        [item],
        collection_intent=CollectionIntent(),
    )
    saved = Database(db_path).get(result["persisted_items"][0]["job_id"])

    assert saved["raw_ocr_text"] == "실제 누적 OCR 원문"
    assert saved["screenshot_path"] == screenshot_path
    assert saved["raw_json"].get("field_evidence") is None


def test_persistence_separates_embedded_cards_sharing_search_url(monkeypatch, tmp_path):
    db_path = tmp_path / "embedded-details.db"
    monkeypatch.setattr(get_settings().paths, "db_path", db_path)
    search_url = "https://www.saramin.co.kr/zf_user/search?searchword=ml"
    first = collected_job(
        url=search_url,
        company_name="에너자이",
        position="ML Engineer/Researcher",
        evidence=JobCollectionEvidence(
            required_fields=["company_name", "position", "url"],
            source_card_key=source_card_key(
                search_url,
                "(주)에너자이",
                "ML Engineer/Researcher",
            ),
        ),
    )
    second = collected_job(
        url=search_url,
        company_name="로민",
        position="ML 머신러닝 엔지니어",
        evidence=JobCollectionEvidence(
            required_fields=["company_name", "position", "url"],
            source_card_key=source_card_key(
                search_url,
                "(주)로민",
                "ML 머신러닝 엔지니어",
            ),
        ),
    )

    first_result = persist_collected_jobs_with_report(
        [first],
        collection_intent=CollectionIntent(),
    )
    second_result = persist_collected_jobs_with_report(
        [second],
        collection_intent=CollectionIntent(),
    )

    assert first_result["created_count"] == second_result["created_count"] == 1
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("SELECT url FROM jobs ORDER BY id").fetchall()
    assert len(rows) == 2
    assert all("#l2c-card=" in row[0] for row in rows)


def test_persistence_applies_requested_date_range(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings().paths, "db_path", tmp_path / "dated.db")
    result = persist_collected_jobs_with_report(
        [
            collected_job(
                url="https://example.com/jobs/out-of-range",
                posted_at="2026-06-01",
            )
        ],
        collection_intent=CollectionIntent(
            filters=JobSearchFilters(
                posted_from="2026-07-01",
                posted_to="2026-07-31",
            )
        ),
    )

    assert result["persisted_count"] == 0
    assert result["rejected_items"][0]["issues"] == [
        "requested_filter_mismatch:posted_at_before_range"
    ]


def test_persistence_report_distinguishes_create_and_update(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings().paths, "db_path", tmp_path / "jobs.db")
    item = collected_job(url="https://example.com/jobs/scope-1")

    created = persist_collected_jobs_with_report(
        [item],
        collection_intent=CollectionIntent(),
    )
    updated = persist_collected_jobs_with_report(
        [item],
        collection_intent=CollectionIntent(),
    )

    assert created["persisted_items"][0]["operation"] == "created"
    assert updated["persisted_items"][0]["operation"] == "updated"
    assert created["persisted_items"][0]["job_id"] == updated["persisted_items"][0]["job_id"]

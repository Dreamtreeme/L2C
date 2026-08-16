"""공고 검토 서비스와 저장 전 전달 경계를 검증한다."""

import json

from agent.application import job_review_service as service
from agent.application.collection_postprocessing import postprocess_collection_batch
from shared.schema.collection_intent import CollectionIntent, JobSearchFilters
from shared.schema.collection_run import CollectionBatch
from shared.schema.feedback_schema import WorkerSubmission
from shared.schema.jd_schema import (
    CollectedJob,
    JobCollectionEvidence,
    JobDraft,
    JobField,
    JobOcrLine,
    JobPosting,
    JobReviewStatus,
)


REQUIRED_FIELDS = [
    "company_name",
    "position",
    "url",
    "main_tasks",
    "requirements",
]


def _draft() -> JobDraft:
    return JobDraft(
        url="https://www.wanted.co.kr/wd/1",
        detail_key="card-1",
        raw_ocr_text=(
            "예시회사 AI 엔지니어 주요 업무 모델 운영 자격 요건 Python"
        ),
        ocr_items=[
            JobOcrLine(
                id=1,
                text="예시회사 AI 엔지니어",
                bbox_ratio=[0.05, 0.1, 0.4, 0.15],
                marker_ids=[3, 5],
                screen="detail.png",
            ),
            JobOcrLine(
                id=2,
                text="주요 업무 모델 운영",
                bbox_ratio=[0.05, 0.4, 0.5, 0.45],
                marker_ids=[20],
                screen="detail.png",
            ),
            JobOcrLine(
                id=3,
                text="자격 요건 Python",
                bbox_ratio=[0.05, 0.6, 0.5, 0.65],
                marker_ids=[30],
                screen="detail.png",
            ),
        ],
        target_company_name="예시회사",
        target_position="AI 엔지니어",
        required_fields=REQUIRED_FIELDS,
        screenshot_path="detail.png",
        screen_count=2,
        last_action="scroll",
        transition_status="unknown",
        transition_reason="no_screen_change",
    )


def _extraction(*, source_exhausted: bool, complete: bool = True):
    return service.JobReviewExtraction(
        posting=JobPosting(
            company_name="예시회사",
            position="AI 엔지니어",
            main_tasks=["모델 운영"] if complete else [],
            requirements=["Python"],
        ),
        is_job_posting=True,
        source_exhausted=source_exhausted,
        field_evidence={
            "company_name": "예시회사",
            "position": "AI 엔지니어",
            **({"main_tasks": "주요 업무 모델 운영"} if complete else {}),
            "requirements": "자격 요건 Python",
        },
        field_evidence_line_ids={
            "company_name": [1],
            "position": [1],
            **({"main_tasks": [2]} if complete else {}),
            "requirements": [3],
        },
        identity_conflict=False,
        reason="누적 OCR 검토 결과",
    )


def _stub_model(monkeypatch, extraction):
    invocation = {}

    def fake_invoke(model, messages, component, *, stream=False):
        invocation.update(
            model=model,
            messages=messages,
            component=component,
            stream=stream,
        )
        return extraction

    model = object()
    monkeypatch.setattr(service, "get_job_review_llm", lambda _tier="lightweight": model)
    monkeypatch.setattr(service, "invoke_with_metrics", fake_invoke)
    monkeypatch.setattr(
        service,
        "job_review_model_spec",
        lambda _tier="lightweight": "test-model",
    )
    return model, invocation


def test_review_structures_ocr_and_preserves_source(monkeypatch):
    model, invocation = _stub_model(
        monkeypatch,
        _extraction(source_exhausted=False),
    )

    review = service.review_job_draft(_draft(), CollectionIntent(site="wanted"))
    payload = json.loads(invocation["messages"][1].content)

    assert invocation["model"] is model
    assert invocation["component"] == "detail_review"
    assert invocation["stream"] is True
    assert payload["required_fields"] == REQUIRED_FIELDS
    assert payload["target_context"] == {
        "company_name": "예시회사",
        "position": "AI 엔지니어",
    }
    assert payload["ocr_items"][0]["bbox_ratio"] == [0.05, 0.1, 0.4, 0.15]
    assert "ocr_text" not in payload
    assert payload["transition_reason"] == "no_screen_change"
    assert review.status == JobReviewStatus.COMPLETE
    assert review.posting.url == _draft().url
    assert review.posting.source_platform == "Wanted"
    assert review.posting.raw_ocr_text == _draft().raw_ocr_text
    assert review.field_evidence_line_ids[JobField.COMPANY_NAME] == [1]


def test_review_keeps_identity_conflict_out_of_completed_jobs(monkeypatch):
    extraction = _extraction(source_exhausted=False)
    extraction.identity_conflict = True
    extraction.identity_candidates = ["예시회사", "추천회사"]
    _stub_model(monkeypatch, extraction)

    review = service.review_job_draft(_draft(), CollectionIntent(site="wanted"))

    assert review.status == JobReviewStatus.NEEDS_MORE
    assert review.identity_conflict is True
    assert review.identity_candidates == ["예시회사", "추천회사"]
    assert "identity_conflict" in review.issues
    assert JobField.COMPANY_NAME in review.missing_fields
    assert JobField.POSITION in review.missing_fields


def test_review_requests_more_when_required_field_is_missing(monkeypatch):
    _stub_model(monkeypatch, _extraction(source_exhausted=True, complete=False))
    draft = _draft().model_copy(
        update={
            "transition_status": "ready",
            "transition_reason": "screen_change_pixels_matched",
        }
    )

    review = service.review_job_draft(draft, CollectionIntent(site="wanted"))

    assert review.status == JobReviewStatus.NEEDS_MORE
    assert [field.value for field in review.missing_fields] == ["main_tasks"]


def test_review_rejects_source_after_exhausted_missing_field(monkeypatch):
    _stub_model(monkeypatch, _extraction(source_exhausted=True, complete=False))

    review = service.review_job_draft(_draft(), CollectionIntent(site="wanted"))

    assert review.status == JobReviewStatus.SOURCE_INCOMPLETE


def test_review_rejects_requested_date_mismatch(monkeypatch):
    extraction = _extraction(source_exhausted=True)
    extraction.posting.posted_at = "2026-06-01"
    extraction.field_evidence["posted_at"] = "2026-06-01"
    draft = _draft().model_copy(
        update={
            "required_fields": [*_draft().required_fields, JobField.POSTED_AT]
        }
    )
    _stub_model(monkeypatch, extraction)
    intent = CollectionIntent(
        site="wanted",
        filters=JobSearchFilters(
            posted_from="2026-07-01",
            posted_to="2026-07-31",
        ),
    )

    review = service.review_job_draft(draft, intent)

    assert review.status == JobReviewStatus.INVALID_TARGET
    assert "posted_at_before_range" in review.issues[0]


def test_postprocessing_passes_reviewed_jobs_without_model_call():
    draft = _draft()
    evidence = JobCollectionEvidence(
        required_fields=draft.required_fields,
        field_evidence={
            "company_name": "예시회사",
            "position": "AI 엔지니어",
            "url": draft.url,
            "main_tasks": "모델 운영",
            "requirements": "Python",
        },
    )
    collected = CollectedJob(
        posting=JobPosting(
            company_name="예시회사",
            position="AI 엔지니어",
            url=draft.url,
            main_tasks=["모델 운영"],
            requirements=["Python"],
        ),
        evidence=evidence,
    )
    batch = CollectionBatch(
        submission=WorkerSubmission(run_id="worker-1"),
        collected_jobs=[collected],
        rejected_items=[{"url": "rejected", "issues": ["invalid_target"]}],
        site_name="Wanted",
    )

    result = postprocess_collection_batch(batch)

    assert result.collected_jobs == [collected]
    assert result.rejected_items == batch.rejected_items

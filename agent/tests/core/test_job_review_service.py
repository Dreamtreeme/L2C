"""공고 검토 서비스와 저장 전 전달 경계를 검증한다."""

import json

import pytest
from langchain_core.exceptions import OutputParserException

from agent.application import job_review_service as service
from shared.schema.collection_intent import CollectionIntent, JobSearchFilters
from shared.schema.jd_schema import (
    JobDraft,
    JobField,
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
    assert invocation["stream"] is False
    assert payload["required_fields"] == REQUIRED_FIELDS
    assert payload["ocr_text"] == _draft().raw_ocr_text
    assert payload["transition_reason"] == "no_screen_change"
    assert review.status == JobReviewStatus.COMPLETE
    assert review.posting.url == _draft().url
    assert review.posting.source_platform == "Wanted"
    assert review.posting.raw_ocr_text == _draft().raw_ocr_text


def test_review_includes_representative_detail_screen(monkeypatch, tmp_path):
    from PIL import Image

    screenshot = tmp_path / "detail.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)
    _model, invocation = _stub_model(
        monkeypatch,
        _extraction(source_exhausted=False),
    )
    draft = _draft().model_copy(update={"screenshot_path": str(screenshot)})

    review = service.review_job_draft(draft, CollectionIntent(site="wanted"))

    content = invocation["messages"][1].content
    assert review.status == JobReviewStatus.COMPLETE
    assert [item["type"] for item in content] == [
        "text",
        "text",
        "image_url",
    ]
    assert content[-1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_review_requests_more_when_required_field_is_missing(monkeypatch):
    _stub_model(monkeypatch, _extraction(source_exhausted=False, complete=False))
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
    draft = _draft().model_copy(
        update={
            "transition_status": "ready",
            "transition_reason": "screen_change_pixels_matched",
        }
    )

    review = service.review_job_draft(draft, CollectionIntent(site="wanted"))

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


def test_review_uses_primary_model_after_structured_output_failure(monkeypatch):
    calls = []
    models = {"lightweight": object(), "primary": object()}

    monkeypatch.setattr(
        service,
        "get_job_review_llm",
        lambda tier="lightweight": models[tier],
    )
    monkeypatch.setattr(
        service,
        "job_review_model_spec",
        lambda tier="lightweight": f"test-{tier}",
    )

    def fake_invoke(model, _messages, component, *, stream=False):
        calls.append((model, component, stream))
        if model is models["lightweight"]:
            raise OutputParserException("empty structured output")
        return _extraction(source_exhausted=False)

    monkeypatch.setattr(service, "invoke_with_metrics", fake_invoke)

    review = service.review_job_draft(_draft(), CollectionIntent(site="wanted"))

    assert review.status == JobReviewStatus.COMPLETE
    assert calls == [
        (models["lightweight"], "detail_review", False),
        (models["primary"], "detail_review_fallback", False),
    ]


def test_primary_review_does_not_retry_structured_output_failure(monkeypatch):
    draft = _draft().model_copy(update={"review_model_tier": "primary"})
    monkeypatch.setattr(service, "get_job_review_llm", lambda _tier: object())
    monkeypatch.setattr(
        service,
        "invoke_with_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OutputParserException("empty structured output")
        ),
    )

    with pytest.raises(OutputParserException):
        service.review_job_draft(draft, CollectionIntent(site="wanted"))

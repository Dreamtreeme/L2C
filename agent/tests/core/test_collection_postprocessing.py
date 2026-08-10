"""OCR 원문 구조화와 후처리의 의미 판정 경계를 검증한다."""

import json

import pytest

from agent.application import collection_postprocessing as service
from agent.observability.run_context import ModelRequestTimeout
from shared.schema.collection_intent import CollectionIntent, JobSearchFilters
from shared.schema.collection_run import CollectionBatch
from shared.schema.feedback_schema import WorkerSubmission
from shared.schema.jd_schema import JobCapture, JobCollectionEvidence, JobPosting


def _batch(
    capture: JobCapture,
    *,
    intent: CollectionIntent | None = None,
) -> CollectionBatch:
    return CollectionBatch(
        submission=WorkerSubmission(
            run_id="worker-1",
            collection_intent=intent or CollectionIntent(site="wanted"),
        ),
        job_captures=[capture],
        site_name="Wanted",
    )


def _capture(*, page_exhausted: bool = False) -> JobCapture:
    return JobCapture(
        url="https://www.wanted.co.kr/wd/1",
        raw_ocr_text="예시회사 AI 엔지니어 주요 업무 모델 운영 자격 요건 Python",
        evidence=JobCollectionEvidence(
            required_fields=[
                "company_name",
                "position",
                "url",
                "main_tasks",
                "requirements",
            ],
            field_evidence={
                "company_name": "예시회사",
                "position": "AI 엔지니어",
                "url": "https://www.wanted.co.kr/wd/1",
                "main_tasks": "모델 운영",
                "requirements": "Python",
            },
            page_exhausted=page_exhausted,
        ),
    )


def test_extract_job_from_capture_uses_screen_evidence_and_preserves_source(
    monkeypatch,
):
    capture = _capture()
    invocation = {}

    def fake_invoke(model, messages, component, *, stream=False):
        invocation.update(
            model=model,
            messages=messages,
            component=component,
            stream=stream,
        )
        return JobPosting(
            company_name="예시회사",
            position="AI 엔지니어",
            main_tasks=["모델 운영"],
            requirements=["Python"],
        )

    model = object()
    monkeypatch.setattr(service, "get_detail_extraction_llm", lambda: model)
    monkeypatch.setattr(service, "invoke_with_metrics", fake_invoke)
    monkeypatch.setattr(service, "detail_extraction_model_spec", lambda: "test-model")

    posting = service.extract_job_from_capture(capture)
    payload = json.loads(invocation["messages"][1].content)

    assert invocation["model"] is model
    assert invocation["component"] == "detail_extraction"
    assert invocation["stream"] is True
    assert payload == {
        "current_url": capture.url,
        "required_fields": [
            "company_name",
            "position",
            "url",
            "main_tasks",
            "requirements",
        ],
        "field_evidence": {
            "company_name": "예시회사",
            "position": "AI 엔지니어",
            "url": capture.url,
            "main_tasks": "모델 운영",
            "requirements": "Python",
        },
        "unavailable_fields": [],
        "ocr_text": capture.raw_ocr_text,
    }
    assert posting.url == capture.url
    assert posting.source_platform == "Wanted"
    assert posting.raw_ocr_text == capture.raw_ocr_text
    assert posting.content_hash


def test_postprocessing_structures_capture_without_worker_state(monkeypatch):
    monkeypatch.setattr(
        service,
        "extract_job_from_capture",
        lambda capture: JobPosting(
            company_name="예시회사",
            position="AI 엔지니어",
            url=capture.url,
            main_tasks=["모델 운영"],
            requirements=["Python"],
            raw_ocr_text=capture.raw_ocr_text,
        ),
    )

    result = service.postprocess_collection_batch(_batch(_capture()))

    assert result.rejected_items == []
    assert result.collected_jobs[0].posting.position == "AI 엔지니어"
    assert result.collected_jobs[0].posting.raw_ocr_text


def test_postprocessing_applies_requested_date_range(monkeypatch):
    monkeypatch.setattr(
        service,
        "extract_job_from_capture",
        lambda capture: JobPosting(
            company_name="예시회사",
            position="AI 엔지니어",
            url=capture.url,
            main_tasks=["모델 운영"],
            requirements=["Python"],
            posted_at="2026-06-01",
        ),
    )
    intent = CollectionIntent(
        site="wanted",
        filters=JobSearchFilters(
            posted_from="2026-07-01",
            posted_to="2026-07-31",
        ),
    )

    result = service.postprocess_collection_batch(_batch(_capture(), intent=intent))

    assert result.collected_jobs == []
    assert "posted_at_before_range" in result.rejected_items[0]["issues"][0]


def test_postprocessing_preserves_visible_required_field_evidence(monkeypatch):
    monkeypatch.setattr(
        service,
        "extract_job_from_capture",
        lambda capture: JobPosting(
            company_name="예시회사",
            position="AI 엔지니어",
            url=capture.url,
            requirements=["Python"],
        ),
    )

    capture = _capture(page_exhausted=True)
    result = service.postprocess_collection_batch(_batch(capture))

    assert result.rejected_items == []
    assert result.collected_jobs[0].posting.main_tasks == ["모델 운영"]

    missing_evidence = capture.model_copy(
        update={
            "evidence": capture.evidence.model_copy(
                update={
                    "field_evidence": {
                        key: value
                        for key, value in capture.evidence.field_evidence.items()
                        if key.value != "main_tasks"
                    }
                }
            )
        }
    )
    rejected = service.postprocess_collection_batch(_batch(missing_evidence))

    assert rejected.collected_jobs == []
    assert "required_field_extraction_incomplete:main_tasks" in (
        rejected.rejected_items[0]["issues"][0]
    )


def test_model_timeout_stops_batch_instead_of_becoming_rejection(monkeypatch):
    def fail_extraction(_capture):
        raise ModelRequestTimeout("detail timeout")

    monkeypatch.setattr(service, "extract_job_from_capture", fail_extraction)

    with pytest.raises(ModelRequestTimeout, match="detail timeout"):
        service.postprocess_collection_batch(_batch(_capture()))


def test_one_invalid_capture_does_not_discard_later_valid_capture(monkeypatch):
    valid_capture = _capture()
    first = valid_capture.model_copy(
        update={
            "evidence": valid_capture.evidence.model_copy(
                update={
                    "field_evidence": {
                        key: value
                        for key, value in valid_capture.evidence.field_evidence.items()
                        if key.value != "main_tasks"
                    }
                }
            )
        }
    )
    second = valid_capture.model_copy(
        update={"url": "https://www.wanted.co.kr/wd/2"}
    )
    batch = _batch(first)
    batch.job_captures.append(second)

    def extract(capture):
        return JobPosting(
            company_name="예시회사",
            position="AI 엔지니어",
            url=capture.url,
            main_tasks=[] if capture.url.endswith("/1") else ["모델 운영"],
            requirements=["Python"],
        )

    monkeypatch.setattr(service, "extract_job_from_capture", extract)

    result = service.postprocess_collection_batch(batch)

    assert [item.posting.url for item in result.collected_jobs] == [second.url]
    assert result.rejected_items[0]["index"] == 0

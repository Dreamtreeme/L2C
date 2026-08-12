"""상세 화면 원문 수집 계약을 검증한다."""

from agent.graph import worker_execution_dispatch
from agent.graph.worker_execution_policy import compact_action_args
from agent.runtime.job_field_contract import merge_job_detail_coverage
from agent.runtime.tool_schema import scroll
from agent.tests.worker_test_support import worker_data_services, worker_state
from shared.schema.collection_intent import CollectionIntent


def _detail_state(current_url: str, required_fields: list[str]):
    return worker_state(
        request={
            "collection_intent": CollectionIntent(required_fields=required_fields)
        },
        collection={
            "job_detail_buffer": {
                "url": current_url,
                "lines": [
                    {"text": "예시회사 백엔드 개발자"},
                    {"text": "주요 업무 API 개발"},
                    {"text": "자격 요건 Python"},
                ],
                "screens": ["detail.png"],
            }
        },
    )


def test_detail_observation_treats_null_as_no_evidence():
    action = scroll.model_validate({"direction": "down", "observed_fields": None})

    assert action.observed_fields == {}


def test_detail_finish_captures_raw_ocr_and_clears_buffer():
    current_url = "https://www.wanted.co.kr/wd/1"
    outcome = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {
            "observed_fields": {
                "company_name": "예시회사",
                "position": "백엔드 개발자",
                "main_tasks": "API 개발",
                "requirements": "Python",
            }
        },
        [],
        current_url=current_url,
        state=_detail_state(
            current_url,
            ["company_name", "position", "url", "main_tasks", "requirements"],
        ),
        data_services=worker_data_services(),
    )

    assert outcome.result["status"] == "success"
    collection_update = outcome.state_update["collection"]
    assert collection_update["job_detail_buffer"] == {}
    assert len(collection_update["job_captures"]) == 1
    capture = collection_update["job_captures"][0]
    assert capture.url == current_url
    assert "자격 요건 Python" in capture.raw_ocr_text
    assert capture.evidence.screenshot_path == "detail.png"


def test_detail_finish_waits_for_required_screen_evidence():
    current_url = "https://www.wanted.co.kr/wd/1"
    outcome = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {"observed_fields": {"company_name": "예시회사"}},
        [],
        current_url=current_url,
        state=_detail_state(
            current_url,
            ["company_name", "position", "url", "requirements"],
        ),
        data_services=worker_data_services(),
    )

    assert outcome.result["status"] == "skipped"
    assert outcome.result["reason"] == "required_field_evidence_incomplete"
    assert set(outcome.result["field_coverage"]["missing_fields"]) == {
        "position",
        "requirements",
    }
    assert "job_captures" not in outcome.state_update["collection"]


def test_detail_finish_preserves_confirmed_unavailable_field():
    current_url = "https://www.wanted.co.kr/wd/1"
    outcome = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {
            "observed_fields": {
                "company_name": "예시회사",
                "position": "백엔드 개발자",
                "benefits": "확인 필요",
            },
            "page_exhausted": True,
            "unavailable_fields": ["benefits"],
        },
        [],
        current_url=current_url,
        state=_detail_state(
            current_url,
            ["company_name", "position", "url", "benefits"],
        ),
        data_services=worker_data_services(),
    )

    assert outcome.result["status"] == "success"
    capture = outcome.state_update["collection"]["job_captures"][0]
    assert [field.value for field in capture.evidence.unavailable_fields] == [
        "benefits"
    ]
    assert "benefits" not in capture.evidence.field_evidence


def test_card_queue_identity_does_not_overwrite_detail_ocr_evidence():
    state = worker_state(
        collection={
            "job_card_queue": [
                {
                    "status": "active",
                    "company": "잘못 연결된 회사",
                    "title": "잘못 연결된 직무",
                }
            ]
        }
    )
    current_url = "https://www.wanted.co.kr/wd/365869"
    coverage = merge_job_detail_coverage(
        {},
        {
            "observed_fields": {
                "company_name": "백패커",
                "position": "[텀블벅] iOS 개발자(1~3년)",
            }
        },
        state=state,
        current_url=current_url,
    )

    assert coverage["field_evidence"]["company_name"] == "백패커"
    assert coverage["field_evidence"]["position"] == "[텀블벅] iOS 개발자(1~3년)"
    assert coverage["field_evidence"]["url"] == current_url


def test_detail_action_args_compaction_is_idempotent():
    args = {
        "page_role": "job_detail",
        "observed_fields": {
            "requirements": "Python",
            "main_tasks": "API 개발",
        },
        "page_exhausted": True,
    }

    compacted = compact_action_args("finish_detail_reading", args)

    assert compact_action_args("finish_detail_reading", compacted) == compacted

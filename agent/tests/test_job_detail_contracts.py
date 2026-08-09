import json

from agent.graph import worker_execution_dispatch
from agent.graph.worker_execution_policy import compact_action_args
from agent.runtime.job_field_contract import merge_job_detail_coverage
from agent.runtime.worker_contracts import build_action_request
from agent.tests.worker_test_support import worker_data_services, worker_state
from shared.schema.jd_schema import JobPosting


def posting(current_url: str, *, include_requirements: bool = True) -> JobPosting:
    return JobPosting(
        company_name="예시회사",
        position="백엔드 개발자",
        url=current_url,
        main_tasks=["API 개발"],
        requirements=["Python"] if include_requirements else [],
    )


def detail_state(current_url: str, required_fields: list[str]):
    return worker_state(
        request={
            "job_collection_contract": {"required_fields": required_fields}
        },
        collection={
            "job_detail_buffer": {
                "url": current_url,
                "lines": [{"text": "API 개발"}, {"text": "Python"}],
                "screens": ["detail.png"],
            }
        },
    )


def test_detail_finish_creates_one_collected_job_and_clears_buffer():
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
        state=detail_state(
            current_url,
            ["company_name", "position", "url", "main_tasks", "requirements"],
        ),
        data_services=worker_data_services(
            extract_job_detail=lambda _state, url: posting(url)
        ),
    )

    assert outcome.result["status"] == "success"
    assert outcome.state_update.job_detail_buffer == {}
    assert outcome.collected_jobs[0].posting.position == "백엔드 개발자"
    assert outcome.collected_jobs[0].evidence.screenshot_path == "detail.png"


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


def test_detail_extraction_does_not_use_card_identity_as_fallback(monkeypatch):
    from agent.application import detail_extraction_service

    class FakeDetailLLM:
        def __init__(self):
            self.messages = []

        def invoke(self, messages):
            self.messages = list(messages)
            return JobPosting(
                position="[텀블벅] iOS 개발자(1~3년)",
                main_tasks=["iOS 앱 개발"],
            )

    fake_llm = FakeDetailLLM()
    monkeypatch.setattr(
        detail_extraction_service,
        "get_detail_extraction_llm",
        lambda: fake_llm,
    )
    monkeypatch.setattr(
        detail_extraction_service,
        "detail_extraction_model_spec",
        lambda: "test-model",
    )
    current_url = "https://www.wanted.co.kr/wd/365869"
    result = detail_extraction_service.extract_job_from_job_detail_buffer(
        worker_state(
            collection={
                "job_card_queue": [
                    {
                        "status": "active",
                        "company": "글로벌머니익스프레스",
                        "title": "[텀블벅] iOS 개발자(1~3년)",
                    }
                ],
                "job_detail_coverage": {
                    "url": current_url,
                    "field_evidence": {
                        "company_name": "백패커",
                        "position": "[텀블벅] iOS 개발자(1~3년)",
                    },
                    "unavailable_fields": [],
                },
                "job_detail_buffer": {
                    "lines": [
                        {"text": "백패커 · 서울 서초구 · 경력 1~3년"},
                        {"text": "[텀블벅] iOS 개발자(1~3년)"},
                    ]
                },
            }
        ),
        current_url,
    )
    request_payload = json.loads(fake_llm.messages[1].content)

    assert "active_card" not in request_payload
    assert "글로벌머니익스프레스" not in fake_llm.messages[1].content
    assert result.company_name is None
    assert result.url == current_url


def test_detail_observation_accepts_multiple_evidence_lines():
    request = build_action_request(
        "llm",
        "",
        [
            {
                "id": "finish",
                "name": "finish_detail_reading",
                "args": {"observed_fields": {"main_tasks": ["API 개발", "성능 최적화"]}},
            }
        ],
    )

    assert request.tool_calls[0].args["observed_fields"] == {
        "main_tasks": "API 개발; 성능 최적화"
    }


def test_detail_finish_waits_for_required_screen_evidence():
    calls = []
    current_url = "https://www.wanted.co.kr/wd/2"

    def fail_if_called(_state, _current_url):
        calls.append(True)
        return None

    outcome = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {
            "observed_fields": {
                "company_name": "예시회사",
                "position": "백엔드 개발자",
            }
        },
        [],
        current_url=current_url,
        state=detail_state(
            current_url,
            ["company_name", "position", "url", "main_tasks", "requirements"],
        ),
        data_services=worker_data_services(extract_job_detail=fail_if_called),
    )

    assert outcome.result["reason"] == "required_field_evidence_incomplete"
    assert outcome.state_update.job_detail_followup["missing_fields"] == [
        "main_tasks",
        "requirements",
    ]
    assert calls == []


def test_detail_finish_preserves_confirmed_unavailable_field():
    current_url = "https://www.wanted.co.kr/wd/3"
    required_fields = [
        "company_name",
        "position",
        "url",
        "main_tasks",
        "requirements",
        "benefits",
    ]
    outcome = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {
            "observed_fields": {
                "company_name": "예시회사",
                "position": "백엔드 개발자",
                "main_tasks": "API 개발",
                "requirements": "Python",
            },
            "unavailable_fields": ["benefits"],
            "page_exhausted": True,
        },
        [],
        current_url=current_url,
        state=detail_state(current_url, required_fields),
        data_services=worker_data_services(
            extract_job_detail=lambda _state, url: posting(url)
        ),
    )

    evidence = outcome.collected_jobs[0].evidence
    assert [field.value for field in evidence.required_fields] == required_fields
    assert [field.value for field in evidence.unavailable_fields] == ["benefits"]


def test_page_end_converts_visible_but_unextracted_field_to_partial_evidence():
    current_url = "https://www.wanted.co.kr/wd/4"
    required_fields = [
        "company_name",
        "position",
        "url",
        "main_tasks",
        "requirements",
        "benefits",
    ]
    outcome = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {
            "observed_fields": {
                "company_name": "예시회사",
                "position": "백엔드 개발자",
                "main_tasks": "API 개발",
                "requirements": "Python",
                "benefits": "고용 조건",
            },
            "page_exhausted": True,
        },
        [],
        current_url=current_url,
        state=detail_state(current_url, required_fields),
        data_services=worker_data_services(
            extract_job_detail=lambda _state, url: posting(url)
        ),
    )

    evidence = outcome.collected_jobs[0].evidence
    assert [field.value for field in evidence.extraction_missing_fields] == [
        "benefits"
    ]
    assert [field.value for field in evidence.unavailable_fields] == ["benefits"]


def test_detail_finish_retries_missing_extraction_before_page_end():
    current_url = "https://www.wanted.co.kr/wd/5"
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
        state=detail_state(
            current_url,
            ["company_name", "position", "url", "main_tasks", "requirements"],
        ),
        data_services=worker_data_services(
            extract_job_detail=lambda _state, url: posting(
                url,
                include_requirements=False,
            )
        ),
    )

    assert outcome.result["reason"] == "required_field_extraction_incomplete"
    assert outcome.result["missing_fields"] == ["requirements"]

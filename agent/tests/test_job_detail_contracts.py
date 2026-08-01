import time

from agent.graph import (
    worker_execution_dispatch,
    worker_observation,
    worker_selection,
    worker_transition,
)
from agent.graph.worker_reflex import reflex_node
from agent.runtime.job_card_queue import replay_job_card_after_return


def test_detail_finish_extracts_once_and_clears_buffer(monkeypatch):
    monkeypatch.setattr(
        worker_execution_dispatch,
        "extract_job_from_job_detail_buffer",
        lambda _state, current_url: {
            "company_name": "보이저엑스",
            "position": "iOS 개발자",
            "url": current_url,
            "requirements": ["Swift"],
        },
    )

    result, extracted = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {
            "page_role": "job_detail",
            "observed_fields": {
                "company_name": "보이저엑스",
                "position": "iOS 개발자",
                "requirements": "자격요건 Swift",
            },
        },
        {},
        current_url="https://www.wanted.co.kr/wd/1",
        state={
            "job_collection_contract": {
                "required_fields": [
                    "company_name",
                    "position",
                    "url",
                    "requirements",
                ]
            },
            "job_detail_buffer": {
                "url": "https://www.wanted.co.kr/wd/1",
                "lines": [{"text": "자격요건 Swift"}],
            }
        },
    )

    assert result["status"] == "success"
    assert result["_job_detail_buffer"] == {}
    assert extracted["공고목록"][0]["position"] == "iOS 개발자"


def test_detail_action_args_compaction_is_idempotent():
    from agent.graph.worker_execution_policy import compact_action_args

    args = {
        "page_role": "job_detail",
        "observed_fields": {
            "requirements": "Python",
            "main_tasks": "API 개발",
        },
        "page_exhausted": True,
    }

    compacted = compact_action_args("finish_detail_reading", args)

    assert compact_action_args(
        "finish_detail_reading",
        compacted,
    ) == compacted


def test_card_queue_identity_does_not_overwrite_detail_ocr_evidence():
    from agent.runtime.job_field_contract import merge_job_detail_coverage

    state = {
        "active_job_card": {
            "company": "잘못 연결된 회사",
            "title": "잘못 연결된 직무",
        }
    }
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
    coverage = merge_job_detail_coverage(
        coverage,
        {"observed_fields": {"main_tasks": "iOS 앱 개발"}},
        state=state,
        current_url=current_url,
    )

    assert coverage["field_evidence"]["company_name"] == "백패커"
    assert coverage["field_evidence"]["position"] == (
        "[텀블벅] iOS 개발자(1~3년)"
    )
    assert coverage["field_evidence"]["url"] == current_url


def test_detail_extraction_does_not_use_card_identity_as_fallback(
    monkeypatch,
):
    import json

    from agent.application import detail_extraction_service

    class FakeDetailLLM:
        def __init__(self):
            self.messages = []

        def invoke(self, messages):
            self.messages = list(messages)
            return {
                "position": "[텀블벅] iOS 개발자(1~3년)",
                "main_tasks": ["iOS 앱 개발"],
            }

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
    result = (
        detail_extraction_service.extract_job_from_job_detail_buffer(
            {
                "active_job_card": {
                    "company": "글로벌머니익스프레스",
                    "title": "[텀블벅] iOS 개발자(1~3년)",
                },
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
            },
            current_url,
        )
    )
    request_payload = json.loads(fake_llm.messages[1].content)

    assert "active_card" not in request_payload
    assert "글로벌머니익스프레스" not in fake_llm.messages[1].content
    assert "company_name" not in result
    assert result["url"] == current_url


def test_detail_observation_accepts_multiple_evidence_lines():
    from agent.graph.action_request import build_action_request

    request = build_action_request(
        "llm",
        "",
        [
            {
                "id": "finish",
                "name": "finish_detail_reading",
                "args": {
                    "observed_fields": {
                        "main_tasks": [
                            "API 개발",
                            "성능 최적화",
                        ]
                    }
                },
            }
        ],
    )

    assert request.tool_calls[0].args["observed_fields"] == {
        "main_tasks": "API 개발; 성능 최적화"
    }


def test_detail_finish_skips_extraction_until_required_evidence_is_complete(
    monkeypatch,
):
    calls = []

    def fail_if_called(_state, _current_url):
        calls.append(True)
        return {}

    monkeypatch.setattr(
        worker_execution_dispatch,
        "extract_job_from_job_detail_buffer",
        fail_if_called,
    )

    result, _extracted = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {
            "page_role": "job_detail",
            "observed_fields": {
                "company_name": "예시회사",
                "position": "백엔드 개발자",
            },
        },
        {},
        current_url="https://www.wanted.co.kr/wd/2",
        state={
            "job_collection_contract": {
                "required_fields": [
                    "company_name",
                    "position",
                    "url",
                    "main_tasks",
                    "requirements",
                ]
            },
            "job_detail_buffer": {
                "url": "https://www.wanted.co.kr/wd/2",
                "lines": [{"text": "백엔드 개발자"}],
            },
        },
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "required_field_evidence_incomplete"
    assert result["_job_detail_followup"]["missing_fields"] == [
        "main_tasks",
        "requirements",
    ]
    assert calls == []


def test_detail_finish_allows_explicit_unavailable_field_at_page_end(
    monkeypatch,
):
    calls = []

    def extract_once(_state, current_url):
        calls.append(True)
        return {
            "company_name": "예시회사",
            "position": "백엔드 개발자",
            "url": current_url,
            "main_tasks": ["API 개발"],
            "requirements": ["Python"],
        }

    monkeypatch.setattr(
        worker_execution_dispatch,
        "extract_job_from_job_detail_buffer",
        extract_once,
    )
    required_fields = [
        "company_name",
        "position",
        "url",
        "main_tasks",
        "requirements",
        "benefits",
    ]

    result, extracted = worker_execution_dispatch.dispatch_state_action(
        "finish_detail_reading",
        {
            "page_role": "job_detail",
            "observed_fields": {
                "company_name": "예시회사",
                "position": "백엔드 개발자",
                "main_tasks": "API 개발",
                "requirements": "Python",
            },
            "unavailable_fields": ["benefits"],
            "page_exhausted": True,
        },
        {},
        current_url="https://www.wanted.co.kr/wd/3",
        state={
            "job_collection_contract": {
                "required_fields": required_fields,
            },
            "job_detail_buffer": {
                "url": "https://www.wanted.co.kr/wd/3",
                "lines": [{"text": "API 개발"}, {"text": "Python"}],
            },
        },
    )

    assert result["status"] == "success"
    assert calls == [True]
    job = extracted["공고목록"][0]
    assert job["_collection_required_fields"] == required_fields
    assert job["_collection_unavailable_fields"] == ["benefits"]

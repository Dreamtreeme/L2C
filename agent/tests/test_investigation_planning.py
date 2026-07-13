from __future__ import annotations

from agent.application.investigation_store import InvestigationStore
from agent.tools.request_clarification import request_clarification
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    ClarificationOption,
    ClarificationQuestion,
    InvestigationPurpose,
    InvestigationRequest,
    InvestigationStatus,
)


def test_clarification_question_requires_unique_option_ids():
    try:
        ClarificationQuestion(
            question_id="recent_period",
            field="recent_period",
            question="최근 기간을 선택해 주세요.",
            options=[
                ClarificationOption(option_id="three_months", label="3개월", value="P3M"),
                ClarificationOption(option_id="three_months", label="다른 3개월", value="P3M"),
            ],
        )
    except ValueError as exc:
        assert "중복" in str(exc)
    else:
        raise AssertionError("중복 선택지 식별자가 허용되었습니다.")


def test_request_clarification_returns_structured_options():
    payload = request_clarification.invoke(
        {
            "question_id": "recent_period",
            "field": "recent_period",
            "question": "최근의 기준을 선택해 주세요.",
            "missing_fields": ["최근 기간"],
            "options": [
                {
                    "option_id": "one_month",
                    "label": "최근 30일",
                    "value": "P30D",
                    "description": "단기 변화를 확인합니다.",
                },
                {
                    "option_id": "three_months",
                    "label": "최근 3개월",
                    "value": "P3M",
                    "description": "계절 변동을 일부 완화합니다.",
                },
            ],
        }
    )

    import json

    data = json.loads(payload)
    assert data["question_id"] == "recent_period"
    assert data["field"] == "recent_period"
    assert [item["value"] for item in data["options"]] == ["P30D", "P3M"]


def test_investigation_store_round_trip(tmp_path):
    store = InvestigationStore(tmp_path / "jobs.db")
    investigation = InvestigationRequest(
        investigation_id="investigation-1",
        conversation_id="conversation-1",
        original_query="최근 AI 개발자 채용 트렌드를 알려줘",
        objective="AI 개발자 채용 트렌드 분석",
        purpose=InvestigationPurpose.TREND,
        status=InvestigationStatus.AWAITING_CLARIFICATION,
        unresolved_fields=["recent_period"],
        clarification_answers=[
            ClarificationAnswer(
                question_id="site_scope",
                selected_option_id="all_sites",
                value="all_enabled",
            )
        ],
    )

    store.save(investigation)
    loaded = store.get("investigation-1")
    latest = store.latest_for_conversation("conversation-1")

    assert loaded == investigation
    assert latest == investigation

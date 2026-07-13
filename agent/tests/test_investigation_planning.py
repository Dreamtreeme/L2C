from __future__ import annotations

from agent.application.investigation_store import InvestigationStore
from agent.application.clarification_service import apply_clarification_answer
from agent.application.evidence_service import inspect_job_evidence
from agent.application.tool_capabilities import build_tool_capability_catalog
from shared.db.database import Database
from agent.tools.request_clarification import request_clarification
from agent.tools.evidence_inventory import inspect_job_evidence as inspect_job_evidence_tool
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    ClarificationOption,
    ClarificationQuestion,
    InvestigationPurpose,
    InvestigationRequest,
    InvestigationStatus,
    EvidenceRequirement,
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


def test_recent_three_months_resolves_analysis_and_comparison_periods():
    investigation = InvestigationRequest(
        investigation_id="investigation-period",
        original_query="최근 AI 개발자 채용 트렌드를 알려줘",
        purpose=InvestigationPurpose.TREND,
        status=InvestigationStatus.AWAITING_CLARIFICATION,
        unresolved_fields=["recent_period"],
        clarification_questions=[
            ClarificationQuestion(
                question_id="recent_period",
                field="recent_period",
                question="최근 기간을 선택해 주세요.",
                options=[
                    ClarificationOption(
                        option_id="three_months",
                        label="최근 3개월",
                        value="P3M",
                    )
                ],
            )
        ],
    )

    from datetime import date

    updated = apply_clarification_answer(
        investigation,
        ClarificationAnswer(
            question_id="recent_period",
            selected_option_id="three_months",
        ),
        today=date(2026, 7, 14),
    )

    assert updated.constraints.posted_from == "2026-04-14"
    assert updated.constraints.posted_to == "2026-07-14"
    assert updated.constraints.comparison_posted_from == "2026-01-14"
    assert updated.constraints.comparison_posted_to == "2026-04-13"
    assert updated.status == InvestigationStatus.CHECKING_EVIDENCE


def test_tool_catalog_exposes_limits_before_collection():
    catalog = build_tool_capability_catalog()
    by_name = {item.tool_name: item for item in catalog}

    assert "inspect_job_evidence" in by_name
    assert "sqlite_query" in by_name
    assert any(name.startswith("realtime_scraping:") for name in by_name)
    assert "created_at은 공고 게시일 근거로 사용할 수 없습니다." in by_name[
        "inspect_job_evidence"
    ].limitations


def test_evidence_inspection_rejects_rows_without_verified_posted_date(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    db.upsert(
        "https://example.com/jobs/1",
        {
            "company_name": "예시회사",
            "position": "AI 엔지니어",
            "source_platform": "wanted",
            "raw_ocr_text": "AI 모델 개발",
        },
    )
    requirement = EvidenceRequirement(
        requirement_id="recent_ai",
        description="최근 AI 공고",
        search_keywords=["AI 엔지니어"],
        posted_from="2026-04-14",
        posted_to="2026-07-14",
        required_fields=["posted_at", "position"],
        minimum_count=1,
    )

    report = inspect_job_evidence(
        db_path,
        [requirement],
        investigation_constraints := InvestigationRequest(
            investigation_id="evidence-test",
            original_query="최근 AI 공고",
        ).constraints,
    )

    assert investigation_constraints.posted_from == ""
    assert report["sufficient"] is False
    assert report["requirements"][0]["matching_count"] == 0
    assert any("게시일" in item for item in report["missing_evidence"])


def test_evidence_inventory_is_exposed_as_commander_tool():
    assert inspect_job_evidence_tool.name == "inspect_job_evidence"

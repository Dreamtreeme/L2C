from __future__ import annotations

import json

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
    EvidencePlan,
    EvidenceValidation,
    InvestigationActionPlan,
    InvestigationConstraints,
    InvestigationPlanStep,
    RequestAnalysis,
    ToolCapability,
    RequirementEvidenceDecision,
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


class _FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.result


class _FailingTool:
    def invoke(self, arguments):
        raise AssertionError("계획 전에 도구가 실행되었습니다.")


def _test_capabilities():
    return [
        ToolCapability(
            tool_name="inspect_job_evidence",
            purpose="DB 근거 확인",
        ),
        ToolCapability(
            tool_name="realtime_scraping:wanted",
            purpose="원티드 수집",
        ),
    ]


def test_workflow_stops_for_choice_before_db_or_collection(tmp_path):
    from agent.graph.investigation_workflow import InvestigationModels, InvestigationWorkflow

    analysis_model = _FakeModel(
        RequestAnalysis(
            objective="AI 개발자 채용 트렌드 분석",
            deliverable="최근 기간과 이전 기간 비교",
            purpose=InvestigationPurpose.TREND,
            unresolved_fields=["recent_period"],
            clarification_questions=[
                ClarificationQuestion(
                    question_id="recent_period",
                    field="recent_period",
                    question="최근의 기준을 선택해 주세요.",
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
    )
    evidence_model = _FakeModel(EvidencePlan())
    action_model = _FakeModel(InvestigationActionPlan())
    answer_model = _FakeModel("호출되면 안 됨")
    workflow = InvestigationWorkflow(
        db_path=tmp_path / "jobs.db",
        models=InvestigationModels(
            analysis_model=analysis_model,
            evidence_model=evidence_model,
            action_model=action_model,
            answer_model=answer_model,
        ),
        capabilities=_test_capabilities(),
        collection_tool=_FailingTool(),
        query_tool=_FailingTool(),
    )

    result = workflow.run("최근 AI 개발자 채용 트렌드를 알려줘")

    assert result["run_status"] == "waiting_input"
    assert result["clarification"]["field"] == "recent_period"
    assert analysis_model.calls == 1
    assert evidence_model.calls == 0
    assert action_model.calls == 0
    assert answer_model.calls == 0


def test_workflow_resumes_choice_then_builds_evidence_plan(tmp_path):
    from datetime import datetime, timezone

    from agent.graph.investigation_workflow import InvestigationModels, InvestigationWorkflow

    analysis_model = _FakeModel(
        RequestAnalysis(
            objective="AI 개발자 채용 트렌드 분석",
            deliverable="두 기간 비교",
            purpose=InvestigationPurpose.TREND,
            unresolved_fields=["recent_period"],
            clarification_questions=[
                ClarificationQuestion(
                    question_id="recent_period",
                    field="recent_period",
                    question="기간을 선택해 주세요.",
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
    )
    evidence_model = _FakeModel(
        EvidencePlan(
            requirements=[
                EvidenceRequirement(
                    requirement_id="current",
                    description="최근 3개월 AI 공고",
                    search_keywords=["AI 개발자"],
                    posted_from="2026-04-14",
                    posted_to="2026-07-14",
                    required_fields=["posted_at"],
                ),
                EvidenceRequirement(
                    requirement_id="previous",
                    description="이전 3개월 AI 공고",
                    search_keywords=["AI 개발자"],
                    posted_from="2026-01-14",
                    posted_to="2026-04-13",
                    required_fields=["posted_at"],
                ),
            ]
        )
    )
    workflow = InvestigationWorkflow(
        db_path=tmp_path / "jobs.db",
        models=InvestigationModels(
            analysis_model=analysis_model,
            evidence_model=evidence_model,
            action_model=_FakeModel(
                InvestigationActionPlan(cannot_proceed_reason="게시일 근거를 확인할 수 없음")
            ),
            answer_model=_FakeModel("게시일 근거가 없어 비교할 수 없습니다."),
        ),
        capabilities=_test_capabilities(),
        collection_tool=_FailingTool(),
        query_tool=_FailingTool(),
        now=lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    first = workflow.run("최근 AI 개발자 채용 트렌드를 알려줘")

    resumed = workflow.run(
        "",
        investigation_id=first["investigation"]["investigation_id"],
        clarification_answer={
            "question_id": "recent_period",
            "selected_option_id": "three_months",
        },
    )

    assert resumed["run_status"] == "completed"
    assert resumed["investigation"]["constraints"]["posted_from"] == "2026-04-14"
    assert resumed["investigation"]["constraints"]["comparison_posted_to"] == "2026-04-13"
    assert [
        item["requirement_id"]
        for item in resumed["investigation"]["evidence_requirements"]
    ] == ["current", "previous"]


def test_workflow_executes_only_registered_collection_plan(tmp_path):
    from langchain_core.messages import AIMessage

    from agent.graph.investigation_workflow import InvestigationModels, InvestigationWorkflow

    db_path = tmp_path / "jobs.db"
    db = Database(db_path)

    class CollectionTool:
        def __init__(self):
            self.calls = []

        def invoke(self, arguments):
            self.calls.append(arguments)
            job_id = db.upsert(
                "https://example.com/jobs/ai-1",
                {
                    "company_name": "예시회사",
                    "position": "AI 개발자",
                    "posted_at": "2026-07-01",
                    "source_platform": "wanted",
                    "raw_ocr_text": "AI 개발자 모델 운영",
                },
            )
            return json.dumps(
                {
                    "persisted_count": 1,
                    "persistence_validation": {
                        "persisted_items": [{"job_id": job_id, "operation": "created"}]
                    },
                },
                ensure_ascii=False,
            )

    class QueryTool:
        def invoke(self, arguments):
            return '<document id="1">AI 개발자 공고</document>'

    collection_tool = CollectionTool()
    workflow = InvestigationWorkflow(
        db_path=db_path,
        models=InvestigationModels(
            analysis_model=_FakeModel(
                RequestAnalysis(
                    objective="최근 AI 개발자 공고 조회",
                    deliverable="공고 목록",
                    purpose=InvestigationPurpose.COLLECT,
                    constraints=InvestigationConstraints(
                        search_keywords=["AI 개발자"],
                        posted_from="2026-06-01",
                        posted_to="2026-07-14",
                    ),
                )
            ),
            evidence_model=_FakeModel(
                EvidencePlan(
                    requirements=[
                        EvidenceRequirement(
                            requirement_id="recent_ai",
                            description="최근 AI 개발자 공고",
                            search_keywords=["AI 개발자"],
                            posted_from="2026-06-01",
                            posted_to="2026-07-14",
                            required_fields=["position", "posted_at"],
                        )
                    ]
                )
            ),
            action_model=_FakeModel(
                InvestigationActionPlan(
                    steps=[
                        InvestigationPlanStep(
                            step_id="collect_recent_ai",
                            action="최근 AI 공고 수집",
                            tool_name="realtime_scraping",
                            arguments={
                                "query": "AI 개발자",
                                "site": "wanted",
                                "posted_from": "2026-06-01",
                                "posted_to": "2026-07-14",
                            },
                            purpose="부족한 최근 공고 확보",
                        )
                    ]
                )
            ),
            answer_model=_FakeModel(AIMessage(content="공고를 확인했습니다 [job_id:1]")),
        ),
        capabilities=_test_capabilities(),
        collection_tool=collection_tool,
        query_tool=QueryTool(),
    )

    result = workflow.run("최근 AI 개발자 공고 찾아줘")

    assert result["run_status"] == "completed"
    assert len(collection_tool.calls) == 1
    assert collection_tool.calls[0]["query"] == "AI 개발자"
    assert collection_tool.calls[0]["site"] == "wanted"
    assert collection_tool.calls[0]["posted_from"] == "2026-06-01"
    assert collection_tool.calls[0]["posted_to"] == "2026-07-14"
    assert collection_tool.calls[0]["original_query"] == "최근 AI 개발자 공고 찾아줘"
    assert collection_tool.calls[0]["freshness_required"] is True
    assert result["valid_ids"] == [1]
    assert result["final_answer"] == "공고를 확인했습니다 [job_id:1]"
    assert result["investigation"]["collection_document_ids"] == [1]


def test_chat_service_uses_investigation_workflow_for_production_path():
    from agent.application.chat_service import ChatService

    class FakeWorkflow:
        def run(self, query, **kwargs):
            assert query == "최근 AI 채용 트렌드"
            assert kwargs["conversation_id"] == "conversation-1"
            return {
                "investigation": {"investigation_id": "investigation-1"},
                "run_status": "waiting_input",
                "final_answer": "최근의 기준을 선택해 주세요.",
                "valid_ids": [],
                "clarification": {
                    "question_id": "recent_period",
                    "field": "recent_period",
                    "question": "최근의 기준을 선택해 주세요.",
                    "options": [],
                },
            }

    result = ChatService(investigation_workflow=FakeWorkflow()).run(
        "최근 AI 채용 트렌드",
        run_id="planned-chat",
        conversation_id="conversation-1",
    )

    assert result["run_status"] == "waiting_input"
    assert result["investigation_id"] == "investigation-1"
    assert result["clarification"]["field"] == "recent_period"


def test_collection_plan_inherits_confirmed_request_and_cohort_constraints():
    from agent.graph.investigation_workflow import _normalized_collection_steps

    investigation = InvestigationRequest(
        investigation_id="plan-normalization",
        original_query="최근 3개월 서울 AI 개발자 트렌드",
        objective="최근과 이전 기간의 요구 기술 비교",
        purpose=InvestigationPurpose.TREND,
        constraints=InvestigationConstraints(
            search_keywords=["AI 개발자"],
            sites=["wanted"],
            count_mode="visible_all",
            location="서울",
        ),
        evidence_requirements=[
            EvidenceRequirement(
                requirement_id="current",
                description="최근 기간",
                search_keywords=["AI 개발자"],
                posted_from="2026-04-14",
                posted_to="2026-07-14",
            )
        ],
    )
    plan = InvestigationActionPlan(
        steps=[
            InvestigationPlanStep(
                step_id="collect-current",
                action="최근 공고 수집",
                tool_name="realtime_scraping",
                arguments={"site": "wanted"},
                expected_evidence=["current"],
            )
        ]
    )

    steps = _normalized_collection_steps(
        plan,
        investigation,
        [
            {
                "tool_name": "realtime_scraping:wanted",
                "purpose": "원티드 수집",
            }
        ],
    )

    assert len(steps) == 1
    assert steps[0].arguments == {
        "query": "AI 개발자",
        "site": "wanted",
        "original_query": "최근 3개월 서울 AI 개발자 트렌드",
        "count_mode": "visible_all",
        "target_count": 0,
        "posted_from": "2026-04-14",
        "posted_to": "2026-07-14",
        "experience": "",
        "location": "서울",
        "employment_type": "",
        "freshness_required": True,
        "purpose": "trend",
        "analysis_goal": "최근과 이전 기간의 요구 기술 비교",
        "task_category": "검색",
    }


def test_semantic_evidence_validation_keeps_only_matching_candidate():
    from agent.graph.investigation_workflow import _apply_evidence_validation

    investigation = InvestigationRequest(
        investigation_id="semantic-validation",
        original_query="서울 AI 개발자 공고",
        constraints=InvestigationConstraints(location="서울"),
        evidence_requirements=[
            EvidenceRequirement(
                requirement_id="seoul_ai",
                description="서울 AI 개발자 공고",
                minimum_count=1,
            )
        ],
    )
    report = {
        "total_db_rows": 2,
        "requirements": [
            {
                "requirement_id": "seoul_ai",
                "description": "서울 AI 개발자 공고",
                "matching_count": 2,
                "verified_posted_at_count": 0,
                "field_coverage": {},
                "document_ids": [1, 2],
                "candidates": [
                    {"document_id": 1, "position": "AI 개발자", "location": "서울"},
                    {"document_id": 2, "position": "영업 담당자", "location": "부산"},
                ],
                "sufficient": True,
                "missing": [],
            }
        ],
    }

    validated = _apply_evidence_validation(
        report,
        investigation,
        EvidenceValidation(
            decisions=[
                RequirementEvidenceDecision(
                    requirement_id="seoul_ai",
                    matching_document_ids=[1, 999],
                )
            ]
        ),
    )

    assert validated["sufficient"] is True
    assert validated["document_ids"] == [1]
    assert validated["requirements"][0]["document_ids"] == [1]

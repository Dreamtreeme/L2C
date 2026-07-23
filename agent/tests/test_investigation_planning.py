from __future__ import annotations

import json
import sqlite3

import pytest

from agent.application.clarification_service import apply_clarification_answer
from agent.application.evidence_service import inspect_job_evidence
from agent.application.search_taxonomy_service import SearchTaxonomyService
from agent.application.tool_capabilities import build_tool_capability_catalog
from agent.runtime.investigation_checkpoint import InvestigationCheckpointRuntime
from shared.db.database import Database
from agent.tools.evidence_inventory import inspect_job_evidence as inspect_job_evidence_tool
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    ClarificationOption,
    ClarificationQuestion,
    EvidencePolicy,
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


def test_visible_all_evidence_does_not_keep_model_invented_fixed_count(tmp_path):
    from agent.graph.investigation_workflow import _normalized_evidence_requirements

    investigation = InvestigationRequest(
        investigation_id="visible-all-evidence",
        original_query="데이터 엔지니어 공고를 전부 수집해줘",
        objective="첫 화면의 데이터 엔지니어 공고 전체 수집",
        purpose=InvestigationPurpose.COLLECT,
        constraints=InvestigationConstraints(
            occupation_query="데이터 엔지니어",
            sites=["wanted"],
            count_mode="visible_all",
        ),
    )
    plan = EvidencePlan(
        requirements=[
            EvidenceRequirement(
                requirement_id="data-engineer",
                description="데이터 엔지니어 공고",
                occupation_query="데이터 엔지니어",
                minimum_count=50,
            )
        ]
    )

    normalized = _normalized_evidence_requirements(
        plan,
        investigation,
        SearchTaxonomyService(tmp_path / "jobs.db"),
    )

    assert normalized[0].minimum_count == 1
    assert plan.requirements[0].minimum_count == 50


def test_explicit_count_is_required_for_single_evidence_group(tmp_path):
    from agent.graph.investigation_workflow import _normalized_evidence_requirements

    investigation = InvestigationRequest(
        investigation_id="explicit-evidence",
        original_query="AI 엔지니어 공고 2개",
        constraints=InvestigationConstraints(
            occupation_query="AI 엔지니어",
            count_mode="explicit",
            target_count=2,
        ),
    )
    plan = EvidencePlan(
        requirements=[
            EvidenceRequirement(
                requirement_id="ai-engineer",
                description="AI 엔지니어 공고",
                occupation_query="AI 엔지니어",
                minimum_count=1,
            )
        ]
    )

    normalized = _normalized_evidence_requirements(
        plan,
        investigation,
        SearchTaxonomyService(tmp_path / "jobs.db"),
    )

    assert normalized[0].minimum_count == 2


def test_single_evidence_requirement_resolves_one_dictionary_scope(tmp_path):
    from agent.graph.investigation_workflow import _normalized_evidence_requirements

    investigation = InvestigationRequest(
        investigation_id="dictionary-scope",
        original_query="DB의 AI 에이전트 공고를 정리해줘",
        constraints=InvestigationConstraints(occupation_query="AI 에이전트"),
    )
    plan = EvidencePlan(
        requirements=[
            EvidenceRequirement(
                requirement_id="ai-agent",
                description="AI 에이전트 관련 공고",
                occupation_query="AI 에이전트",
            )
        ]
    )

    normalized = _normalized_evidence_requirements(
        plan,
        investigation,
        SearchTaxonomyService(tmp_path / "jobs.db"),
    )

    assert normalized[0].occupation_concept_keys == [
        "l2c:occupation:ai_agent_engineer"
    ]
    assert normalized[0].collection_search_term == "AI 에이전트"


def test_evidence_inspection_requires_all_explicit_text_groups(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    matching_id = db.upsert(
        "https://example.com/jobs/ai-agent",
        {
            "company_name": "에이전트회사",
            "position": "AI Agent Engineer",
            "source_platform": "wanted",
        },
    )
    db.upsert(
        "https://example.com/jobs/general-ai",
        {
            "company_name": "AI회사",
            "position": "AI 모델 엔지니어",
            "source_platform": "wanted",
        },
    )
    db.upsert(
        "https://example.com/jobs/general-agent",
        {
            "company_name": "플랫폼회사",
            "position": "고객 지원 Agent 플랫폼 개발자",
            "source_platform": "wanted",
        },
    )
    requirement = EvidenceRequirement(
        requirement_id="ai-agent",
        description="AI 에이전트 공고",
        exact_text_groups=[
            ["AI", "인공지능"],
            ["에이전트", "agent"],
        ],
    )

    report = inspect_job_evidence(
        db_path,
        [requirement],
        InvestigationConstraints(),
    )

    assert report["requirements"][0]["document_ids"] == [matching_id]


def test_unresolved_occupation_builds_candidates_for_semantic_review(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    expected_ids = {
        db.upsert(
            "https://example.com/jobs/ai",
            {
                "company_name": "AI회사",
                "position": "AI 모델 엔지니어",
                "posted_at": "2026-07-01",
                "source_platform": "wanted",
            },
        ),
        db.upsert(
            "https://example.com/jobs/llm",
            {
                "company_name": "LLM회사",
                "position": "LLM 서비스 아키텍트",
                "posted_at": "2026-07-02",
                "source_platform": "wanted",
            },
        ),
        db.upsert(
            "https://example.com/jobs/qa",
            {
                "company_name": "QA회사",
                "position": "QA 엔지니어",
                "posted_at": "2026-07-03",
                "source_platform": "wanted",
            },
        ),
    }
    db.upsert(
        "https://example.com/jobs/other-site",
        {
            "company_name": "다른회사",
            "position": "AI 엔지니어",
            "posted_at": "2026-07-04",
            "source_platform": "jobkorea",
        },
    )
    requirement = EvidenceRequirement(
        requirement_id="ai-engineer",
        description="AI 엔지니어 공고",
        occupation_query="알려지지 않은 신직무",
        posted_from="2026-07-01",
        posted_to="2026-07-31",
        required_sites=["wanted"],
    )

    report = inspect_job_evidence(
        db_path,
        [requirement],
        InvestigationConstraints(),
    )

    requirement_report = report["requirements"][0]
    assert set(requirement_report["document_ids"]) == expected_ids
    assert requirement_report["semantic_review_required"] is True
    assert requirement_report["occupation_query"] == "알려지지 않은 신직무"


def test_investigation_checkpoint_is_separate_from_business_database(tmp_path):
    db_path = tmp_path / "jobs.db"
    Database(db_path)
    checkpoint_runtime = InvestigationCheckpointRuntime(db_path)
    checkpoint_path = checkpoint_runtime.checkpoint_path
    checkpoint_runtime.close()

    with sqlite3.connect(db_path) as connection:
        business_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    with sqlite3.connect(checkpoint_path) as connection:
        checkpoint_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert "investigation_sessions" not in business_tables
    assert "checkpoints" not in business_tables
    assert {"checkpoints", "writes"} <= checkpoint_tables


def test_recent_three_months_resolves_analysis_and_comparison_periods():
    investigation = InvestigationRequest(
        investigation_id="investigation-period",
        original_query="최근 AI 개발자 채용 트렌드를 알려줘",
        purpose=InvestigationPurpose.TREND,
        status=InvestigationStatus.AWAITING_CLARIFICATION,
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


def test_custom_two_month_option_uses_same_deterministic_date_resolution():
    from datetime import date

    investigation = InvestigationRequest(
        investigation_id="investigation-custom-period",
        original_query="최근 채용 트렌드",
        purpose=InvestigationPurpose.TREND,
        status=InvestigationStatus.AWAITING_CLARIFICATION,
        clarification_questions=[
            ClarificationQuestion(
                question_id="recent_period",
                field="recent_period",
                question="기간을 입력해 주세요.",
                options=[],
                allow_custom=True,
            )
        ],
    )

    updated = apply_clarification_answer(
        investigation,
        ClarificationAnswer(question_id="recent_period", custom_value="P2M"),
        today=date(2026, 7, 14),
    )

    assert updated.constraints.posted_from == "2026-05-14"
    assert updated.constraints.comparison_posted_from == "2026-03-14"
    assert updated.constraints.comparison_posted_to == "2026-05-13"


def test_comparison_period_answer_resolves_all_related_date_fields():
    investigation = InvestigationRequest(
        investigation_id="investigation-comparison-period",
        original_query="지난달보다 AI 채용이 늘었는지 알려줘",
        purpose=InvestigationPurpose.TREND,
        status=InvestigationStatus.AWAITING_CLARIFICATION,
        clarification_questions=[
            ClarificationQuestion(
                question_id="comparison_period",
                field="comparison_period",
                question="어떤 기간을 비교할까요?",
                options=[
                    ClarificationOption(
                        option_id="same_days",
                        label="동일 일수 비교",
                        value=(
                            "current=2026-07-01/2026-07-14;"
                            "comparison=2026-06-01/2026-06-14"
                        ),
                    )
                ],
            )
        ],
    )

    updated = apply_clarification_answer(
        investigation,
        ClarificationAnswer(
            question_id="comparison_period",
            selected_option_id="same_days",
        ),
    )

    assert updated.constraints.posted_from == "2026-07-01"
    assert updated.constraints.posted_to == "2026-07-14"
    assert updated.constraints.comparison_posted_from == "2026-06-01"
    assert updated.constraints.comparison_posted_to == "2026-06-14"
    assert updated.clarification_questions == []


def test_tool_catalog_exposes_limits_before_collection():
    catalog = build_tool_capability_catalog()
    by_name = {item.tool_name: item for item in catalog}

    assert "inspect_job_evidence" in by_name
    assert "sqlite_query" not in by_name
    assert any(name.startswith("realtime_scraping:") for name in by_name)
    assert "created_at은 공고 게시일 근거로 사용할 수 없습니다." in by_name[
        "inspect_job_evidence"
    ].limitations


def test_capability_catalog_only_exposes_confirmed_collection_site():
    from agent.graph.investigation_workflow import _capabilities_for_investigation

    investigation = InvestigationRequest(
        investigation_id="selected-capability",
        original_query="원티드 공고",
        constraints=InvestigationConstraints(sites=["wanted"]),
    )
    catalog = [
        {"tool_name": "inspect_job_evidence"},
        {"tool_name": "realtime_scraping:wanted"},
        {"tool_name": "realtime_scraping:saramin"},
    ]

    selected = _capabilities_for_investigation(catalog, investigation)

    assert [item["tool_name"] for item in selected] == [
        "inspect_job_evidence",
        "realtime_scraping:wanted",
    ]


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
        occupation_query="AI 엔지니어",
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


def test_evidence_inspection_matches_site_slug_case_insensitively(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    db.upsert(
        "https://example.com/jobs/ai",
        {
            "company_name": "예시회사",
            "position": "AI 엔지니어",
            "source_platform": "Wanted",
            "tech_stack": '["Python"]',
        },
    )
    requirement = EvidenceRequirement(
        requirement_id="wanted_ai",
        description="원티드 AI 공고",
        occupation_query="AI 엔지니어",
        required_sites=["wanted"],
        required_fields=["tech_stack"],
        minimum_count=1,
    )

    report = inspect_job_evidence(
        db_path,
        [requirement],
        InvestigationRequest(
            investigation_id="site-case",
            original_query="원티드 AI 공고",
        ).constraints,
    )

    assert report["sufficient"] is True
    assert report["requirements"][0]["matching_count"] == 1


def test_evidence_inventory_is_exposed_as_commander_tool():
    assert inspect_job_evidence_tool.name == "inspect_job_evidence"


def test_evidence_documents_are_loaded_as_structured_json(tmp_path):
    from agent.application.evidence_service import load_job_evidence_documents

    db_path = tmp_path / "documents.db"
    db = Database(db_path)
    job_id = db.upsert(
        "https://example.com/jobs/structured",
        {
            "company_name": "예시회사",
            "position": "AI 엔지니어",
            "source_platform": "Wanted",
            "tech_stack": '["Python", "PyTorch"]',
        },
    )

    documents = load_job_evidence_documents(db_path, [job_id])

    assert len(documents) == 1
    assert documents[0].id == job_id
    assert documents[0].company_name == "예시회사"
    assert "Python" in documents[0].tech_stack


def test_answer_document_projection_uses_raw_ocr_only_as_fallback():
    from agent.graph.investigation_workflow import _answer_evidence_documents

    complete = {
        "id": 1,
        "tech_stack": "Python",
        "main_tasks": "서비스 개발",
        "requirements": "경력 3년",
        "preferred": "Agent 경험",
        "benefits": "교육비",
        "raw_ocr_text": "중복된 전체 OCR 본문",
    }
    incomplete = {
        **complete,
        "id": 2,
        "preferred": "",
        "raw_ocr_text": "누락 필드를 보완할 OCR 본문",
    }

    projected = _answer_evidence_documents([complete, incomplete])

    assert "raw_ocr_text" not in projected[0]
    assert projected[1]["raw_ocr_text"] == "누락 필드를 보완할 OCR 본문"


def test_llm_prompt_context_excludes_accumulated_investigation_snapshot():
    from agent.graph.investigation_workflow import (
        _compact_db_report,
        _request_prompt_context,
    )

    investigation = InvestigationRequest(
        investigation_id="compact-context",
        original_query="AI 에이전트 공고를 정리해줘",
        objective="AI 에이전트 공고 정리",
        constraints=InvestigationConstraints(occupation_query="AI 에이전트"),
        evidence_snapshot={"large_debug_value": "반복 상태" * 1000},
        final_answer="이전 답변",
    )
    report = {
        "total_db_rows": 2,
        "sufficient": True,
        "document_ids": [1],
        "missing_evidence": [],
        "requirements": [
            {
                "requirement_id": "ai-agent",
                "document_ids": [1],
                "candidates": [{"document_id": 1, "debug": "후보 원본" * 1000}],
            }
        ],
    }

    request_context = _request_prompt_context(investigation)
    compact_report = _compact_db_report(report)

    assert "evidence_snapshot" not in request_context
    assert "final_answer" not in request_context
    assert "candidates" not in compact_report["requirements"][0]
    assert request_context["constraints"]["occupation_query"] == "AI 에이전트"


def test_evidence_requirements_reject_fields_outside_database_contract():
    with pytest.raises(ValueError, match="job_body"):
        EvidencePlan(
            requirements=[
                EvidenceRequirement(
                    requirement_id="skills",
                    description="AI 엔지니어 주요 기술",
                    required_fields=["position", "job_body", "tech_stack"],
                )
            ]
        )


def test_evidence_requirements_deduplicate_supported_fields(tmp_path):
    from agent.graph.investigation_workflow import _normalized_evidence_requirements

    investigation = InvestigationRequest(
        investigation_id="field-contract",
        original_query="AI 엔지니어 주요 기술",
    )
    plan = EvidencePlan(
        requirements=[
            EvidenceRequirement(
                requirement_id="skills",
                description="AI 엔지니어 주요 기술",
                required_fields=["position", "tech_stack", "position"],
            )
        ]
    )

    normalized = _normalized_evidence_requirements(
        plan,
        investigation,
        SearchTaxonomyService(tmp_path / "jobs.db"),
    )

    assert normalized[0].required_fields == ["position", "tech_stack"]


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


def test_workflow_answers_general_knowledge_without_evidence_or_tools(tmp_path):
    from agent.graph.investigation_workflow import InvestigationModels, InvestigationWorkflow

    analysis_model = _FakeModel(
        RequestAnalysis(
            objective="iOS 개발자의 일반적인 업무와 필요 기술 설명",
            deliverable="업무와 기술 요약",
            purpose=InvestigationPurpose.LOOKUP,
            evidence_policy=EvidencePolicy.MODEL_KNOWLEDGE,
        )
    )
    evidence_model = _FakeModel(EvidencePlan())
    action_model = _FakeModel(InvestigationActionPlan())
    answer_model = _FakeModel("iOS 개발자는 Apple 플랫폼용 앱을 설계하고 개발합니다.")
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
    )

    result = workflow.run("iOS 개발자는 보통 어떤 일을 하고 어떤 기술이 필요해?")

    assert result["run_status"] == "completed"
    assert result["final_answer"].startswith("iOS 개발자는")
    assert result["investigation"]["evidence_policy"] == "model_knowledge"
    assert result["investigation"]["evidence_requirements"] == []
    assert analysis_model.calls == 1
    assert evidence_model.calls == 0
    assert action_model.calls == 0
    assert answer_model.calls == 1


def test_workflow_stops_for_choice_before_db_or_collection(tmp_path):
    from agent.graph.investigation_workflow import InvestigationModels, InvestigationWorkflow

    analysis_model = _FakeModel(
        RequestAnalysis(
            objective="AI 개발자 채용 트렌드 분석",
            deliverable="최근 기간과 이전 기간 비교",
            purpose=InvestigationPurpose.TREND,
            constraints=InvestigationConstraints(
                posted_from="2026-04-14",
                posted_to="2026-07-14",
                comparison_posted_from="2026-01-14",
                comparison_posted_to="2026-04-13",
            ),
            assumptions=["최근을 3개월로 해석했다."],
            clarification_questions=[
                ClarificationQuestion(
                    question_id="analysis_dimensions",
                    field="analysis_dimensions",
                    question="어떤 변화를 분석할까요?",
                    options=[
                        ClarificationOption(
                            option_id="job_count",
                            label="공고 수",
                            value="공고 수",
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
    )

    result = workflow.run("최근 AI 개발자 채용 트렌드를 알려줘")

    assert result["run_status"] == "waiting_input"
    assert result["clarification"]["field"] == "analysis_dimensions"
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
            constraints=InvestigationConstraints(
                posted_from="2026-04-14",
                posted_to="2026-07-14",
                comparison_posted_from="2026-01-14",
                comparison_posted_to="2026-04-13",
            ),
            assumptions=["최근을 3개월로 해석했다."],
            clarification_questions=[
                ClarificationQuestion(
                    question_id="analysis_dimensions",
                    field="analysis_dimensions",
                    question="어떤 변화를 분석할까요?",
                    options=[
                        ClarificationOption(
                            option_id="job_count",
                            label="공고 수",
                            value="공고 수",
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
                    occupation_query="AI 개발자",
                    posted_from="2026-04-14",
                    posted_to="2026-07-14",
                    required_fields=["posted_at"],
                ),
                EvidenceRequirement(
                    requirement_id="previous",
                    description="이전 3개월 AI 공고",
                    occupation_query="AI 개발자",
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
        now=lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    first = workflow.run("최근 AI 개발자 채용 트렌드를 알려줘")
    workflow.close()

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
        now=lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError, match="식별자가 다릅니다"):
        workflow.run(
            "",
            investigation_id=first["investigation"]["investigation_id"],
            clarification_answer={
                "question_id": "different-question",
                "selected_option_id": "job_count",
            },
        )

    resumed = workflow.run(
        "",
        investigation_id=first["investigation"]["investigation_id"],
        clarification_answer={
            "question_id": "analysis_dimensions",
            "selected_option_id": "job_count",
        },
    )

    assert resumed["run_status"] == "completed"
    assert resumed["investigation"]["constraints"]["posted_from"] == "2026-04-14"
    assert resumed["investigation"]["constraints"]["comparison_posted_to"] == "2026-04-13"
    assert resumed["investigation"]["constraints"]["analysis_dimensions"] == ["공고 수"]
    assert [
        item["requirement_id"]
        for item in resumed["investigation"]["evidence_requirements"]
    ] == ["current", "previous"]
    assert analysis_model.calls == 1
    assert evidence_model.calls == 1
    workflow.close()


def test_workflow_executes_only_registered_collection_plan(tmp_path):
    from langchain_core.messages import AIMessage

    from agent.graph.investigation_workflow import InvestigationModels, InvestigationWorkflow

    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = SearchTaxonomyService(db_path)

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
            taxonomy.link_job(job_id)
            return json.dumps(
                {
                    "persisted_count": 1,
                    "persistence_validation": {
                        "persisted_items": [{"job_id": job_id, "operation": "created"}]
                    },
                },
                ensure_ascii=False,
            )

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
                        occupation_query="AI 개발자",
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
                            occupation_query="AI 개발자",
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
        taxonomy_service=taxonomy,
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


def test_workflow_loads_dictionary_indexed_collection_documents_without_role_recheck(tmp_path):
    from langchain_core.messages import AIMessage

    from agent.graph.investigation_workflow import InvestigationModels, InvestigationWorkflow

    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = SearchTaxonomyService(db_path)

    class CollectionTool:
        def invoke(self, _arguments):
            job_id = db.upsert(
                "https://example.com/jobs/llm-1",
                {
                    "company_name": "예시회사",
                    "position": "AI 엔지니어",
                    "job_category": "AI/ML",
                    "tech_stack": ["RAG"],
                    "source_platform": "wanted",
                    "raw_ocr_text": "Vector DB와 RAG 서비스 설계",
                },
            )
            taxonomy.link_job(job_id)
            return json.dumps(
                {
                    "persisted_count": 0,
                    "observed_job_ids": [job_id],
                    "persistence_validation": {"persisted_items": []},
                },
                ensure_ascii=False,
            )

    validation_model = _FakeModel(
        EvidenceValidation(
            decisions=[
                RequirementEvidenceDecision(
                    requirement_id="ai_skills",
                    matching_document_ids=[1],
                )
            ]
        )
    )
    workflow = InvestigationWorkflow(
        db_path=db_path,
        models=InvestigationModels(
            analysis_model=_FakeModel(
                RequestAnalysis(
                    objective="AI 엔지니어 기술 조사",
                    deliverable="주요 기술 요약",
                    purpose=InvestigationPurpose.COLLECT,
                    constraints=InvestigationConstraints(
                        occupation_query="AI 엔지니어",
                        sites=["wanted"],
                    ),
                )
            ),
            evidence_model=_FakeModel(
                EvidencePlan(
                    requirements=[
                        EvidenceRequirement(
                            requirement_id="ai_skills",
                            description="AI 엔지니어 기술 근거",
                            occupation_query="AI 엔지니어",
                            required_sites=["wanted"],
                            required_fields=["position", "tech_stack"],
                        )
                    ]
                )
            ),
            validation_model=validation_model,
            action_model=_FakeModel(
                InvestigationActionPlan(
                    steps=[
                        InvestigationPlanStep(
                            step_id="collect_ai",
                            action="AI 공고 수집",
                            tool_name="realtime_scraping",
                            arguments={"query": "AI 엔지니어", "site": "wanted"},
                            purpose="현재 검색 결과 확인",
                        )
                    ]
                )
            ),
            answer_model=_FakeModel(AIMessage(content="RAG가 확인됩니다 [job_id:1]")),
        ),
        capabilities=_test_capabilities(),
        collection_tool=CollectionTool(),
        taxonomy_service=taxonomy,
    )

    result = workflow.run("원티드 AI 엔지니어 공고를 수집하고 기술을 정리해줘")

    assert result["valid_ids"] == [1]
    assert result["investigation"]["collection_document_ids"] == [1]
    assert result["documents"][0]["position"] == "AI 엔지니어"
    assert validation_model.calls == 0


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
            occupation_query="AI 개발자",
            sites=["wanted"],
            count_mode="visible_all",
            location="서울",
        ),
        evidence_requirements=[
            EvidenceRequirement(
                requirement_id="current",
                description="최근 기간",
                occupation_query="AI 개발자",
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
    assert steps[0].arguments.model_dump(mode="json") == {
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


def test_site_display_name_is_normalized_to_registry_slug():
    from agent.graph.investigation_workflow import _normalize_site_slugs

    constraints = _normalize_site_slugs(
        InvestigationConstraints(sites=["원티드", "jobkorea"])
    )

    assert constraints.sites == ["wanted", "jobkorea"]


def test_site_specific_collection_tool_name_is_normalized_for_execution():
    from agent.graph.investigation_workflow import _normalized_collection_steps

    investigation = InvestigationRequest(
        investigation_id="site-tool-name",
        original_query="원티드 백엔드 공고",
        purpose=InvestigationPurpose.COLLECT,
        constraints=InvestigationConstraints(occupation_query="백엔드 개발자"),
        evidence_requirements=[
            EvidenceRequirement(
                requirement_id="backend",
                description="백엔드 공고",
            )
        ],
    )
    plan = InvestigationActionPlan(
        steps=[
            InvestigationPlanStep(
                step_id="wanted-backend",
                action="원티드 수집",
                tool_name="realtime_scraping:wanted",
                arguments={},
            )
        ]
    )

    steps = _normalized_collection_steps(
        plan,
        investigation,
        [{"tool_name": "realtime_scraping:wanted"}],
    )

    assert len(steps) == 1
    assert steps[0].tool_name == "realtime_scraping"
    assert steps[0].arguments.site == "wanted"


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
                occupation_query="AI 개발자",
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
                "semantic_review_required": True,
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


def test_semantic_validation_selects_explicit_count_after_ordered_classification():
    from agent.graph.investigation_workflow import _apply_evidence_validation

    investigation = InvestigationRequest(
        investigation_id="semantic-order",
        original_query="AI 엔지니어 공고 2개",
        constraints=InvestigationConstraints(count_mode="explicit", target_count=2),
        evidence_requirements=[
            EvidenceRequirement(
                requirement_id="ai",
                description="AI 엔지니어 공고",
                occupation_query="AI 엔지니어",
            )
        ],
    )
    candidates = [
        {"document_id": document_id, "position": f"후보 {document_id}"}
        for document_id in (1, 2, 3)
    ]
    report = {
        "total_db_rows": 3,
        "requirements": [
            {
                "requirement_id": "ai",
                "description": "AI 엔지니어 공고",
                "semantic_review_required": True,
                "candidates": candidates,
            }
        ],
    }

    validated = _apply_evidence_validation(
        report,
        investigation,
        EvidenceValidation(
            decisions=[
                RequirementEvidenceDecision(
                    requirement_id="ai",
                    matching_document_ids=[3, 1, 2],
                )
            ]
        ),
    )

    assert validated["document_ids"] == [3, 1]


def test_evidence_validation_payload_excludes_full_document_text():
    from agent.graph.investigation_workflow import _evidence_validation_payload

    investigation = InvestigationRequest(
        investigation_id="compact-validation",
        original_query="AI 엔지니어 공고",
        evidence_requirements=[
            EvidenceRequirement(
                requirement_id="ai",
                description="AI 엔지니어 공고",
                occupation_query="AI 엔지니어",
            )
        ],
    )
    payload = _evidence_validation_payload(
        {
            "requirements": [
                {
                    "requirement_id": "ai",
                    "semantic_review_required": True,
                    "candidates": [
                        {
                            "document_id": 1,
                            "position": "LLM 서비스 아키텍트",
                            "job_category": "AI",
                            "raw_ocr_text": "매우 긴 공고 본문",
                            "company_name": "예시회사",
                        }
                    ],
                }
            ]
        },
        investigation,
    )

    candidate = payload[0]["candidates"][0]
    assert candidate == {
        "document_id": 1,
        "position": "LLM 서비스 아키텍트",
        "job_category": "AI",
    }

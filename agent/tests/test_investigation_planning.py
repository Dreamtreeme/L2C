from __future__ import annotations

import pytest

from agent.application.clarification_service import apply_clarification_answer
from agent.application.evidence_service import inspect_job_evidence
from agent.application.search_taxonomy_service import SearchTaxonomyService
from agent.graph.investigation_context import (
    InvestigationModels,
)
from agent.graph.investigation_evidence_policy import (
    apply_evidence_validation,
    build_evidence_validation_payload,
    normalize_collection_steps,
    normalize_evidence_requirements,
)
from agent.graph.investigation_workflow import InvestigationWorkflow
from shared.db.database import Database
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


def test_visible_all_evidence_does_not_keep_model_invented_fixed_count(tmp_path):

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

    normalized = normalize_evidence_requirements(
        plan,
        investigation,
        SearchTaxonomyService(tmp_path / "jobs.db"),
    )

    assert normalized[0].minimum_count == 1
    assert plan.requirements[0].minimum_count == 50


def test_explicit_count_is_required_for_single_evidence_group(tmp_path):

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

    normalized = normalize_evidence_requirements(
        plan,
        investigation,
        SearchTaxonomyService(tmp_path / "jobs.db"),
    )

    assert normalized[0].minimum_count == 2

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

def test_evidence_inspection_limits_candidates_to_document_scope(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    excluded_id = db.upsert(
        "https://example.com/jobs/old",
        {
            "company_name": "기존회사",
            "position": "기존 공고",
            "source_platform": "wanted",
        },
    )
    included_id = db.upsert(
        "https://example.com/jobs/current",
        {
            "company_name": "현재회사",
            "position": "현재 수집 공고",
            "source_platform": "wanted",
        },
    )

    report = inspect_job_evidence(
        db_path,
        [
            EvidenceRequirement(
                requirement_id="current_collection",
                description="이번 수집 공고",
                required_sites=["wanted"],
            )
        ],
        InvestigationConstraints(),
        document_scope_ids=[included_id],
    )

    assert report["document_scope_ids"] == [included_id]
    assert report["document_ids"] == [included_id]
    assert excluded_id not in report["requirements"][0]["document_ids"]

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

class _FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.result

class _FailingTool:
    def __call__(self, arguments):
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
        collect_jobs=_FailingTool(),
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

def test_workflow_resumes_choice_then_builds_evidence_plan(tmp_path):
    from datetime import datetime, timezone


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
        collect_jobs=_FailingTool(),
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
        collect_jobs=_FailingTool(),
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


    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = SearchTaxonomyService(db_path)

    class CollectionTool:
        def __init__(self):
            self.calls = []

        def __call__(self, arguments):
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
            return {
                "status": "completed",
                "persisted_count": 1,
                "resolved_count": 1,
                "document_ids": [job_id],
                "persisted_items": [{"job_id": job_id, "operation": "created"}],
            }

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
                            tool_name="realtime_scraping",
                            arguments={
                                "search_keyword": "AI 개발자",
                                "site": "wanted",
                                "filters": {
                                    "posted_from": "2026-06-01",
                                    "posted_to": "2026-07-14",
                                },
                            },
                            purpose="부족한 최근 공고 확보",
                        )
                    ]
                )
            ),
            answer_model=_FakeModel(AIMessage(content="공고를 확인했습니다 [job_id:1]")),
        ),
        capabilities=_test_capabilities(),
        collect_jobs=collection_tool,
        taxonomy_service=taxonomy,
    )

    result = workflow.run("최근 AI 개발자 공고 찾아줘")

    assert result["run_status"] == "completed"
    assert len(collection_tool.calls) == 1
    intent = collection_tool.calls[0]
    assert intent.search_keyword == "AI 개발자"
    assert intent.site == "wanted"
    assert intent.filters.posted_from == "2026-06-01"
    assert intent.filters.posted_to == "2026-07-14"
    assert intent.original_query == "최근 AI 개발자 공고 찾아줘"
    assert intent.freshness_required is True
    assert result["valid_ids"] == [1]
    assert result["final_answer"] == "공고를 확인했습니다 [job_id:1]"
    assert result["investigation"]["collection_document_ids"] == [1]

def test_workflow_semantically_rechecks_dictionary_indexed_web_collection(tmp_path):
    from langchain_core.messages import AIMessage


    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = SearchTaxonomyService(db_path)

    class CollectionTool:
        def __call__(self, _arguments):
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
            return {
                "status": "completed",
                "resolved_count": 1,
                "observed_job_ids": [job_id],
                "document_ids": [job_id],
            }

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
                        evidence_policy=EvidencePolicy.WEB_REQUIRED,
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
                            tool_name="realtime_scraping",
                            arguments={"search_keyword": "AI 엔지니어", "site": "wanted"},
                            purpose="현재 검색 결과 확인",
                        )
                    ]
                )
            ),
            answer_model=_FakeModel(AIMessage(content="RAG가 확인됩니다 [job_id:1]")),
        ),
        capabilities=_test_capabilities(),
        collect_jobs=CollectionTool(),
        taxonomy_service=taxonomy,
    )

    result = workflow.run("원티드 AI 엔지니어 공고를 수집하고 기술을 정리해줘")

    assert result["valid_ids"] == [1]
    assert result["investigation"]["collection_document_ids"] == [1]
    assert result["documents"][0]["position"] == "AI 엔지니어"
    assert validation_model.calls == 1

def test_workflow_does_not_answer_web_request_from_stale_database_evidence(tmp_path):
    from langchain_core.messages import AIMessage


    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    stale_id = db.upsert(
        "https://example.com/jobs/stale-qa-automation",
        {
            "company_name": "기존회사",
            "position": "QA 자동화 엔지니어",
            "source_platform": "wanted",
        },
    )

    class CollectionTool:
        def __call__(self, _arguments):
            collected_id = db.upsert(
                "https://example.com/jobs/current-qa-lead",
                {
                    "company_name": "현재회사",
                    "position": "QA 팀장",
                    "source_platform": "wanted",
                },
            )
            return {
                "status": "completed",
                "persisted_count": 1,
                "resolved_count": 1,
                "document_ids": [collected_id],
                "persisted_items": [
                    {"job_id": collected_id, "operation": "created"}
                ],
            }

    validation_model = _FakeModel(
        EvidenceValidation(
            decisions=[
                RequirementEvidenceDecision(
                    requirement_id="current_qa_automation",
                    matching_document_ids=[],
                    reason="QA 팀장은 QA 자동화 엔지니어 공고가 아니다.",
                )
            ]
        )
    )
    workflow = InvestigationWorkflow(
        db_path=db_path,
        models=InvestigationModels(
            analysis_model=_FakeModel(
                RequestAnalysis(
                    objective="현재 QA 자동화 엔지니어 공고 확인",
                    deliverable="현재 웹에서 확인한 공고",
                    purpose=InvestigationPurpose.COLLECT,
                    evidence_policy=EvidencePolicy.WEB_REQUIRED,
                    constraints=InvestigationConstraints(sites=["wanted"]),
                )
            ),
            evidence_model=_FakeModel(
                EvidencePlan(
                    requirements=[
                        EvidenceRequirement(
                            requirement_id="current_qa_automation",
                            description="현재 QA 자동화 엔지니어 공고",
                            required_sites=["wanted"],
                            required_fields=["position"],
                        )
                    ]
                )
            ),
            validation_model=validation_model,
            action_model=_FakeModel(
                InvestigationActionPlan(
                    steps=[
                        InvestigationPlanStep(
                            step_id="collect_current_qa",
                            tool_name="realtime_scraping",
                            arguments={
                                "search_keyword": "QA 자동화 엔지니어",
                                "site": "wanted",
                            },
                            purpose="현재 웹 결과 확인",
                        )
                    ]
                )
            ),
            answer_model=_FakeModel(
                AIMessage(content="현재 수집 결과에서 정확히 일치하는 공고를 찾지 못했습니다.")
            ),
        ),
        capabilities=_test_capabilities(),
        collect_jobs=CollectionTool(),
    )

    result = workflow.run(
        "원티드에서 QA 자동화 엔지니어 공고를 직접 확인해줘"
    )

    assert validation_model.calls == 1
    assert result["valid_ids"] == []
    assert result["documents"] == []
    assert stale_id not in result["investigation"]["evidence_document_ids"]
    assert result["investigation"]["collection_document_ids"] != [stale_id]

def test_collection_plan_inherits_confirmed_request_and_cohort_constraints():

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
                    required_fields=["posted_at", "requirements"],
                )
        ],
    )
    plan = InvestigationActionPlan(
        steps=[
            InvestigationPlanStep(
                step_id="collect-current",
                tool_name="realtime_scraping",
                arguments={"site": "wanted"},
                expected_evidence=["current"],
            )
        ]
    )

    steps = normalize_collection_steps(
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
        "original_query": "최근 3개월 서울 AI 개발자 트렌드",
        "site": "wanted",
        "search_keyword": "AI 개발자",
        "count_mode": "visible_all",
        "target_count": 0,
        "filters": {
            "posted_from": "2026-04-14",
            "posted_to": "2026-07-14",
            "experience": "",
            "location": "서울",
            "employment_type": "",
        },
        "freshness_required": True,
        "purpose": "trend",
        "analysis_goal": "최근과 이전 기간의 요구 기술 비교",
        "task_category": "검색",
        "required_fields": ["posted_at", "requirements"],
    }

def test_semantic_evidence_validation_keeps_only_matching_candidate():

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

    validated = apply_evidence_validation(
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

def test_evidence_validation_payload_excludes_full_document_text():

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
    payload = build_evidence_validation_payload(
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

from __future__ import annotations

import pytest

from benchmark.investigation_quality_eval import evaluate_investigation_analysis
from benchmark.investigation_scenarios import INVESTIGATION_SCENARIOS
from shared.schema.investigation_schema import (
    ClarificationOption,
    ClarificationQuestion,
    EvidencePolicy,
    EvidencePlan,
    EvidenceRequirement,
    InvestigationConstraints,
    InvestigationPurpose,
    RequestAnalysis,
)


def test_twelve_regression_scenarios_are_unique_and_cover_service_boundaries():
    assert len(INVESTIGATION_SCENARIOS) == 12
    assert len({item.scenario_id for item in INVESTIGATION_SCENARIOS}) == 12
    assert {item.expected_purpose for item in INVESTIGATION_SCENARIOS} == {
        "lookup",
        "collect",
        "compare",
        "trend",
    }
    assert sum(item.requires_verified_posted_at for item in INVESTIGATION_SCENARIOS) >= 4


def test_request_analysis_does_not_require_question_for_assumed_period():
    analysis = RequestAnalysis(
        objective="최근 공고 분석",
        deliverable="기술 요약",
        purpose=InvestigationPurpose.TREND,
        constraints=InvestigationConstraints(
            posted_from="2026-04-14",
            posted_to="2026-07-14",
            comparison_posted_from="2026-01-14",
            comparison_posted_to="2026-04-13",
        ),
        assumptions=["요즘을 최근 3개월로 해석했다."],
    )

    assert analysis.clarification_questions == []
    assert analysis.constraints.posted_from == "2026-04-14"


def test_investigation_constraints_reject_unknown_count_mode():
    with pytest.raises(ValueError):
        InvestigationConstraints(count_mode="limit", target_count=0)


def test_occupation_expression_clears_missing_scope_flag():
    constraints = InvestigationConstraints(
        occupation_scope_required=True,
        occupation_query="개발자",
    )

    assert constraints.occupation_scope_required is False


def test_ambiguous_trend_quality_keeps_collection_goal_until_metric_is_selected():
    scenario = INVESTIGATION_SCENARIOS[0]
    questions = [
        ClarificationQuestion(
            question_id="trend_metric",
            field="analysis_dimensions",
            question="어떤 변화를 볼까요?",
            options=[
                ClarificationOption(
                    option_id="job_count",
                    label="공고 수",
                    value="job_count",
                )
            ],
        ),
    ]
    analysis = RequestAnalysis(
        objective="개발자 채용 트렌드 확인",
        deliverable="기간별 변화 요약",
        purpose=InvestigationPurpose.COLLECT,
        evidence_policy=EvidencePolicy.WEB_REQUIRED,
        constraints=InvestigationConstraints(
            occupation_query="개발자",
            collection_search_term="개발자",
            count_mode="visible_all",
            posted_from="2026-04-14",
            posted_to="2026-07-14",
            comparison_posted_from="2026-01-14",
            comparison_posted_to="2026-04-13",
        ),
        assumptions=["요즘을 최근 3개월로 해석했다."],
        clarification_questions=questions,
    )

    result = evaluate_investigation_analysis(scenario, analysis)

    assert result["passed"] is True


def test_monthly_growth_quality_uses_model_assumption_without_period_question():
    scenario = next(
        item for item in INVESTIGATION_SCENARIOS if item.scenario_id == "ai_monthly_growth"
    )
    analysis = RequestAnalysis(
        objective="AI 개발자 채용 공고 수 변화 확인",
        deliverable="동일 길이 기간의 공고 수 비교",
        purpose=InvestigationPurpose.TREND,
        evidence_policy=EvidencePolicy.WEB_REQUIRED,
        constraints=InvestigationConstraints(
            occupation_query="AI 개발자",
            collection_search_term="AI 개발자",
            sites=["wanted"],
            posted_from="2026-07-01",
            posted_to="2026-07-14",
            comparison_posted_from="2026-06-01",
            comparison_posted_to="2026-06-14",
            count_mode="visible_all",
            analysis_dimensions=["공고 수"],
        ),
        assumptions=["진행 중인 이번 달과 지난달의 같은 일수 구간을 비교한다."],
    )
    evidence_plan = EvidencePlan(
        requirements=[
            EvidenceRequirement(
                requirement_id="current",
                description="이번 달 AI 개발자 공고",
                occupation_query="AI 개발자",
                collection_search_term="AI 개발자",
                posted_from="2026-07-01",
                posted_to="2026-07-14",
                required_fields=["posted_at", "position"],
            ),
            EvidenceRequirement(
                requirement_id="comparison",
                description="지난달 같은 기간 AI 개발자 공고",
                occupation_query="AI 개발자",
                collection_search_term="AI 개발자",
                posted_from="2026-06-01",
                posted_to="2026-06-14",
                required_fields=["posted_at", "position"],
            ),
        ]
    )

    result = evaluate_investigation_analysis(scenario, analysis, evidence_plan)
    incomplete_result = evaluate_investigation_analysis(scenario, analysis)

    assert analysis.clarification_questions == []
    assert result["passed"] is True
    assert incomplete_result["checks"]["evidence_plan"] is False
    assert incomplete_result["passed"] is False


@pytest.mark.parametrize(
    "scenario",
    [item for item in INVESTIGATION_SCENARIOS if item.requires_verified_posted_at],
)
def test_date_scenarios_fail_quality_without_posted_at_requirement(scenario):
    analysis = RequestAnalysis(
        objective="날짜 조건 공고 조사",
        deliverable="공고 결과",
        purpose=InvestigationPurpose(scenario.expected_purpose),
        constraints=InvestigationConstraints(count_mode=scenario.expected_count_mode),
    )
    evidence_plan = EvidencePlan(
        requirements=[
            EvidenceRequirement(
                requirement_id=f"group-{index}",
                description="기간별 공고",
                required_fields=["position"],
            )
            for index in range(scenario.expected_evidence_groups)
        ]
    )

    result = evaluate_investigation_analysis(scenario, analysis, evidence_plan)

    assert result["checks"]["posted_at"] is False
    assert result["passed"] is False

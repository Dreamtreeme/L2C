from __future__ import annotations

import pytest

from benchmark.investigation_quality_eval import evaluate_investigation_analysis
from benchmark.investigation_scenarios import INVESTIGATION_SCENARIOS
from shared.schema.investigation_schema import (
    ClarificationOption,
    ClarificationQuestion,
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


def test_request_analysis_rejects_unresolved_field_without_question():
    with pytest.raises(ValueError, match="대응하는 확인 질문"):
        RequestAnalysis(
            objective="최근 채용 트렌드",
            deliverable="트렌드 요약",
            purpose=InvestigationPurpose.TREND,
            unresolved_fields=["recent_period"],
        )


def test_request_analysis_accepts_relative_period_question_alias():
    analysis = RequestAnalysis(
        objective="최근 공고 분석",
        deliverable="기술 요약",
        purpose=InvestigationPurpose.TREND,
        unresolved_fields=["posted_from"],
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

    assert analysis.unresolved_fields == ["posted_from"]


def test_unspecified_count_mode_alias_does_not_create_arbitrary_limit():
    constraints = InvestigationConstraints(count_mode="limit", target_count=0)

    assert constraints.count_mode == "unspecified"


def test_ambiguous_trend_quality_requires_questions_before_evidence():
    scenario = INVESTIGATION_SCENARIOS[0]
    questions = [
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
        ),
        ClarificationQuestion(
            question_id="job_scope",
            field="search_keywords",
            question="어떤 개발자 직군을 볼까요?",
            options=[
                ClarificationOption(
                    option_id="all_developers",
                    label="개발자 전체",
                    value="개발자",
                )
            ],
        ),
        ClarificationQuestion(
            question_id="trend_metric",
            field="trend_metric",
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
        purpose=InvestigationPurpose.TREND,
        constraints=InvestigationConstraints(count_mode="visible_all"),
        unresolved_fields=[
            "recent_period",
            "search_keywords",
            "trend_metric",
        ],
        clarification_questions=questions,
    )

    result = evaluate_investigation_analysis(scenario, analysis)

    assert result["passed"] is True


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

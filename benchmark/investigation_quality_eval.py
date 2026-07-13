"""대표 요청에 대한 지휘자 구조화 판단을 평가한다."""

from __future__ import annotations

from typing import Any

from benchmark.investigation_scenarios import InvestigationScenario
from shared.schema.investigation_schema import EvidencePlan, RequestAnalysis


def evaluate_investigation_analysis(
    scenario: InvestigationScenario,
    analysis: RequestAnalysis,
    evidence_plan: EvidencePlan | None = None,
) -> dict[str, Any]:
    aliases = {
        "recent_period": {"recent_period", "posted_from", "posted_to"},
        "trend_metric": {"trend_metric", "analysis_dimensions"},
        "analysis_dimensions": {"trend_metric", "analysis_dimensions"},
        "comparison_period": {"comparison_period", "recent_period"},
        "site_scope": {"site_scope", "sites"},
    }
    question_fields = {item.field for item in analysis.clarification_questions}
    required_question_fields = set(scenario.expected_clarification_fields)
    required_groups = [aliases.get(field, {field}) for field in required_question_fields]
    optional_groups = [
        aliases.get(field, {field})
        for field in scenario.optional_clarification_fields
    ]
    allowed_groups = [*required_groups, *optional_groups]
    checks = {
        "purpose": analysis.purpose.value == scenario.expected_purpose,
        "clarification": (
            all(bool(question_fields & group) for group in required_groups)
            and all(any(field in group for group in allowed_groups) for field in question_fields)
            and bool(analysis.unresolved_fields) == bool(required_question_fields)
        ),
        "count_mode": analysis.constraints.count_mode == scenario.expected_count_mode,
    }
    if evidence_plan is not None:
        checks["evidence_groups"] = (
            len(evidence_plan.requirements) >= scenario.expected_evidence_groups
        )
        if scenario.requires_verified_posted_at:
            checks["posted_at"] = all(
                "posted_at" in requirement.required_fields
                for requirement in evidence_plan.requirements
            )
    return {
        "scenario_id": scenario.scenario_id,
        "passed": all(checks.values()),
        "checks": checks,
    }


__all__ = ["evaluate_investigation_analysis"]

"""대표 요청에 대한 지휘자 구조화 판단을 평가한다."""

from __future__ import annotations

from collections import Counter
from typing import Any

from benchmark.investigation_scenarios import InvestigationScenario
from shared.schema.investigation_schema import EvidencePlan, RequestAnalysis


def _normalized_term(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def evaluate_investigation_analysis(
    scenario: InvestigationScenario,
    analysis: RequestAnalysis,
    evidence_plan: EvidencePlan | None = None,
) -> dict[str, Any]:
    question_fields = {item.field for item in analysis.clarification_questions}
    required_question_fields = set(scenario.expected_clarification_fields)
    allowed_question_fields = required_question_fields | set(
        scenario.optional_clarification_fields
    )
    checks = {
        "purpose": analysis.purpose.value == scenario.expected_purpose,
        "evidence_policy": (
            analysis.evidence_policy.value == scenario.expected_evidence_policy
        ),
        "clarification": (
            required_question_fields <= question_fields
            and question_fields <= allowed_question_fields
        ),
        "count_mode": analysis.constraints.count_mode == scenario.expected_count_mode,
        "target_count": (
            analysis.constraints.target_count == scenario.expected_target_count
        ),
        "occupation_scope_required": (
            analysis.constraints.occupation_scope_required
            == scenario.expected_scope_required
        ),
        "occupation_domain_query": not bool(
            analysis.constraints.occupation_domain_query
        ),
        "sites": set(analysis.constraints.sites) == set(scenario.expected_sites),
    }
    if scenario.expected_occupation_query:
        checks["occupation_query"] = _normalized_term(
            analysis.constraints.occupation_query
        ) == _normalized_term(scenario.expected_occupation_query)
    if scenario.expected_collection_search_term:
        checks["collection_search_term"] = _normalized_term(
            analysis.constraints.collection_search_term
        ) == _normalized_term(scenario.expected_collection_search_term)
    if scenario.requires_comparison_dates:
        checks["comparison_dates"] = all(
            (
                analysis.constraints.posted_from,
                analysis.constraints.posted_to,
                analysis.constraints.comparison_posted_from,
                analysis.constraints.comparison_posted_to,
            )
        )
    if scenario.requires_assumption:
        checks["assumption"] = bool(analysis.assumptions)
    if not analysis.clarification_questions:
        checks["evidence_plan"] = evidence_plan is not None
    if evidence_plan is not None:
        checks["evidence_groups"] = (
            len(evidence_plan.requirements) >= scenario.expected_evidence_groups
        )
        if scenario.requires_verified_posted_at:
            checks["posted_at"] = all(
                "posted_at" in requirement.required_fields
                for requirement in evidence_plan.requirements
            )
        if scenario.expected_evidence_terms:
            expected_terms = Counter(
                _normalized_term(term)
                for term in scenario.expected_evidence_terms
            )
            occupation_terms = Counter(
                _normalized_term(requirement.scope.occupation_query)
                for requirement in evidence_plan.requirements
                if requirement.scope.occupation_query
            )
            collection_terms = Counter(
                _normalized_term(requirement.scope.collection_search_term)
                for requirement in evidence_plan.requirements
                if requirement.scope.collection_search_term
            )
            checks["evidence_occupation_terms"] = not (
                expected_terms - occupation_terms
            )
            checks["evidence_collection_terms"] = not (
                expected_terms - collection_terms
            )
    return {
        "scenario_id": scenario.scenario_id,
        "passed": all(checks.values()),
        "checks": checks,
    }


__all__ = ["evaluate_investigation_analysis"]

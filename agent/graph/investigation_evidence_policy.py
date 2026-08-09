"""조사 근거 판정과 수집 단계 정규화를 수행하는 순수 정책."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agent.graph.investigation_ports import TaxonomyRequirementPort
from shared.schema.collection_intent import CollectionIntent
from shared.schema.investigation_schema import (
    EvidencePlan,
    EvidenceValidation,
    InvestigationActionPlan,
    InvestigationPlanStep,
    InvestigationRequest,
)


def compact_db_report(report: dict[str, Any]) -> dict[str, Any]:
    """후보 판정이 끝난 뒤 계획과 답변에 필요한 근거 요약만 반환한다."""

    requirements = []
    for item in report.get("requirements", []) or []:
        if not isinstance(item, dict):
            continue
        requirements.append(
            {
                key: value
                for key, value in item.items()
                if key != "candidates"
            }
        )
    return {
        "total_db_rows": int(report.get("total_db_rows") or 0),
        "requirements": requirements,
        "sufficient": bool(report.get("sufficient", False)),
        "document_ids": list(report.get("document_ids") or []),
        "missing_evidence": list(report.get("missing_evidence") or []),
    }

def build_evidence_validation_payload(
    report: dict[str, Any],
    investigation: InvestigationRequest,
) -> list[dict[str, Any]]:
    """직무 판정에 필요한 후보 메타데이터만 모델에 전달한다."""

    requirements = {
        item.requirement_id: item for item in investigation.evidence_requirements
    }
    candidate_fields = (
        "document_id",
        "position",
        "job_category",
        "experience",
        "employment_type",
        "location",
        "posted_at",
        "source_platform",
        "tech_stack",
        "requirements",
        "preferred",
        "field_presence",
    )
    groups = []
    for item in report.get("requirements", []) or []:
        if not isinstance(item, dict):
            continue
        requirement = requirements.get(str(item.get("requirement_id") or ""))
        if requirement is None or not item.get("semantic_review_required"):
            continue
        groups.append(
            {
                "requirement_id": requirement.requirement_id,
                "description": requirement.description,
                "occupation_query": requirement.occupation_query,
                "skill_queries": list(requirement.skill_queries),
                "exact_text_groups": list(requirement.exact_text_groups),
                "minimum_count": requirement.minimum_count,
                "required_fields": list(requirement.required_fields),
                "required_sites": list(requirement.required_sites),
                "posted_from": requirement.posted_from,
                "posted_to": requirement.posted_to,
                "candidates": [
                    {
                        key: candidate.get(key)
                        for key in candidate_fields
                        if key in candidate
                    }
                    for candidate in item.get("candidates", [])
                    if isinstance(candidate, dict)
                ],
            }
        )
    return groups

def normalize_collection_steps(
    plan: InvestigationActionPlan,
    investigation: InvestigationRequest,
    capability_catalog: list[dict[str, Any]],
) -> list[InvestigationPlanStep]:
    """LLM이 선택한 단계에 이미 확정된 요청 조건을 빠짐없이 전달한다."""

    allowed_sites = {
        str(item.get("tool_name") or "").split(":", 1)[1]
        for item in capability_catalog
        if str(item.get("tool_name") or "").startswith("realtime_scraping:")
    }
    requirements = {
        item.requirement_id: item for item in investigation.evidence_requirements
    }
    normalized: list[InvestigationPlanStep] = []
    signatures: set[str] = set()
    maximum_steps = len(allowed_sites) * max(1, len(requirements))
    for step in plan.steps:
        tool_name = str(step.tool_name or "")
        if tool_name != "realtime_scraping" and not tool_name.startswith(
            "realtime_scraping:"
        ):
            continue
        intent = step.arguments
        site_from_tool = tool_name.split(":", 1)[1] if ":" in tool_name else ""
        site = str(intent.site or site_from_tool).strip()
        if not site and len(investigation.constraints.sites) == 1:
            site = investigation.constraints.sites[0]
        if not site or site not in allowed_sites:
            continue

        requirement = next(
            (
                requirements[requirement_id]
                for requirement_id in step.expected_evidence
                if requirement_id in requirements
            ),
            None,
        )
        search_keyword = str(
            intent.search_keyword
            or (requirement.collection_search_term if requirement else "")
            or (requirement.occupation_query if requirement else "")
            or investigation.constraints.collection_search_term
            or investigation.constraints.occupation_query
        ).strip()
        if not search_keyword:
            continue
        posted_from = str(
            intent.filters.posted_from
            or (requirement.posted_from if requirement else "")
            or investigation.constraints.posted_from
        )
        posted_to = str(
            intent.filters.posted_to
            or (requirement.posted_to if requirement else "")
            or investigation.constraints.posted_to
        )
        normalized_intent = CollectionIntent.model_validate(
            {
                **intent.model_dump(mode="json"),
                "site": site,
                "search_keyword": search_keyword,
                "original_query": investigation.original_query,
                "count_mode": investigation.constraints.count_mode,
                "target_count": investigation.constraints.target_count,
                "filters": {
                    "posted_from": posted_from,
                    "posted_to": posted_to,
                    "experience": investigation.constraints.experience,
                    "location": investigation.constraints.location,
                    "employment_type": investigation.constraints.employment_type,
                },
                "freshness_required": bool(
                    intent.freshness_required or posted_from or posted_to
                ),
                "purpose": investigation.purpose.value,
                "analysis_goal": investigation.objective,
                "task_category": "검색",
                "required_fields": (
                    list(requirement.required_fields)
                    if requirement is not None
                    else list(intent.required_fields)
                ),
            }
        )
        signature = json.dumps(
            normalized_intent.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        normalized.append(
            step.model_copy(
                update={
                    "tool_name": "realtime_scraping",
                    "arguments": normalized_intent,
                }
            )
        )
        if len(normalized) >= maximum_steps:
            break
    return normalized

def normalize_evidence_requirements(
    plan: EvidencePlan,
    investigation: InvestigationRequest,
    taxonomy_service: TaxonomyRequirementPort,
) -> list:
    """근거 필드를 DB 계약에 맞추고 화면 전체 수집의 표본 수를 정규화한다."""

    normalized = []
    single_requirement = len(plan.requirements) == 1
    for requirement in plan.requirements:
        updates = {
            "required_fields": list(dict.fromkeys(requirement.required_fields))
        }
        if investigation.constraints.count_mode == "visible_all":
            updates["minimum_count"] = 1
        elif (
            single_requirement
            and investigation.constraints.count_mode == "explicit"
        ):
            updates["minimum_count"] = investigation.constraints.target_count
        normalized.append(
            taxonomy_service.enrich_requirement(
                requirement.model_copy(update=updates),
                investigation.constraints,
            )
        )
    return normalized

def needs_semantic_evidence_validation(report: dict[str, Any]) -> bool:
    return any(
        item.get("semantic_review_required") and item.get("candidates")
        for item in report.get("requirements", [])
        if isinstance(item, dict)
    )


def _valid_candidate_ids(
    values: list[int],
    candidates: dict[int, dict[str, Any]],
) -> list[int]:
    return list(
        dict.fromkeys(
            int(document_id)
            for document_id in values
            if int(document_id) in candidates
        )
    )


def _validated_requirement_report(
    item: dict[str, Any],
    requirement: Any,
    decision: Any,
    investigation: InvestigationRequest,
) -> dict[str, Any]:
    """요구사항 하나의 모델 판정을 후보 집합에 한정해 다시 계산한다."""

    candidates = {
        int(candidate["document_id"]): candidate
        for candidate in item.get("candidates", [])
    }
    candidate_ids = (
        decision.matching_document_ids if decision is not None else []
    )
    if not item.get("semantic_review_required"):
        candidate_ids = list(candidates)
    matching_ids = _valid_candidate_ids(candidate_ids, candidates)
    selected_ids = matching_ids
    if (
        len(investigation.evidence_requirements) == 1
        and investigation.constraints.count_mode == "explicit"
    ):
        selected_ids = matching_ids[: investigation.constraints.target_count]
    selected = [candidates[document_id] for document_id in selected_ids]

    missing: list[str] = []
    if len(selected) < requirement.minimum_count:
        missing.append(
            "의미 조건을 만족하는 표본 "
            f"{requirement.minimum_count - len(selected)}건 부족"
        )
    field_coverage = {
        field: sum(
            1
            for candidate in selected
            if candidate.get("field_presence", {}).get(field)
        )
        for field in requirement.required_fields
    }
    missing.extend(
        f"{field} 근거 부족"
        for field in requirement.required_fields
        if field_coverage.get(field, 0) < requirement.minimum_count
    )
    posted_dates = sorted(
        str(candidate.get("posted_at") or "")[:10]
        for candidate in selected
        if str(candidate.get("posted_at") or "").strip()
    )
    if (
        requirement.posted_from or requirement.posted_to
    ) and len(posted_dates) < requirement.minimum_count:
        missing.append("검증된 게시일 근거 부족")
    return {
        **item,
        "matching_count": len(selected),
        "verified_posted_at_count": len(posted_dates),
        "oldest_posted_at": posted_dates[0] if posted_dates else "",
        "newest_posted_at": posted_dates[-1] if posted_dates else "",
        "field_coverage": field_coverage,
        "document_ids": selected_ids,
        "site_counts": dict(
            Counter(
                str(candidate.get("source_platform") or "unknown")
                for candidate in selected
            )
        ),
        "candidates": selected,
        "sufficient": not missing,
        "missing": list(dict.fromkeys(missing)),
    }

def apply_evidence_validation(
    report: dict[str, Any],
    investigation: InvestigationRequest,
    validation: EvidenceValidation,
) -> dict[str, Any]:
    """모델 판단을 후보 집합 안에서만 허용하고 충분성을 다시 계산한다."""

    requirements = {
        item.requirement_id: item for item in investigation.evidence_requirements
    }
    decisions = {
        item.requirement_id: item
        for item in validation.decisions
    }
    all_document_ids: list[int] = []
    seen_document_ids: set[int] = set()
    reports: list[dict[str, Any]] = []
    for item in report.get("requirements", []):
        requirement = requirements.get(str(item.get("requirement_id") or ""))
        if requirement is None:
            continue
        decision = decisions.get(requirement.requirement_id)
        validated_report = _validated_requirement_report(
            item,
            requirement,
            decision,
            investigation,
        )
        for document_id in validated_report["document_ids"]:
            if document_id not in seen_document_ids:
                seen_document_ids.add(document_id)
                all_document_ids.append(document_id)
        reports.append(validated_report)
    missing_evidence = [
        f"{item['description']}: {reason}"
        for item in reports
        for reason in item["missing"]
    ]
    return {
        **report,
        "requirements": reports,
        "sufficient": bool(reports) and all(item["sufficient"] for item in reports),
        "document_ids": all_document_ids,
        "missing_evidence": missing_evidence,
    }

__all__ = [
    "apply_evidence_validation",
    "build_evidence_validation_payload",
    "compact_db_report",
    "needs_semantic_evidence_validation",
    "normalize_collection_steps",
    "normalize_evidence_requirements",
]

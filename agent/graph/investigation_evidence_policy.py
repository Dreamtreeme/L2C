"""조사 근거 판정과 수집 단계 정규화를 수행하는 순수 정책."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from shared.schema.agent_contract import DEFAULT_JOB_COLLECTION_FIELDS
from shared.schema.collection_intent import CollectionIntent, JobSearchFilters
from shared.schema.investigation_schema import (
    EvidencePlan,
    EvidenceRequirement,
    EvidenceValidation,
    InvestigationActionPlan,
    InvestigationPlanStep,
    InvestigationRequest,
)


def _realtime_sites(collection_capabilities: list[dict[str, Any]]) -> set[str]:
    return {
        str(item.get("tool_name") or "").split(":", 1)[1]
        for item in collection_capabilities
        if str(item.get("tool_name") or "").startswith("realtime_scraping:")
    }


def build_database_lookup_evidence_plan(
    investigation: InvestigationRequest,
) -> EvidencePlan:
    """확정된 DB 조회 조건을 하나의 근거 집단으로 옮긴다."""

    constraints = investigation.constraints
    minimum_count = (
        constraints.target_count if constraints.count_mode == "explicit" else 1
    )
    return EvidencePlan(
        requirements=[
            EvidenceRequirement(
                requirement_id="database_lookup",
                description=investigation.objective or investigation.original_query,
                scope=constraints,
                required_fields=list(DEFAULT_JOB_COLLECTION_FIELDS),
                minimum_count=minimum_count,
                reason=investigation.objective,
            )
        ]
    )


def compact_db_report(report: dict[str, Any]) -> dict[str, Any]:
    """후보 판정이 끝난 뒤 계획과 답변에 필요한 근거 요약만 반환한다."""

    requirements = []
    for item in report.get("requirements", []) or []:
        if not isinstance(item, dict):
            continue
        requirements.append(
            {key: value for key, value in item.items() if key != "candidates"}
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
    evidence_requirements: list[EvidenceRequirement],
) -> list[dict[str, Any]]:
    """의미 판정에 필요한 조건과 구조화 공고 본문을 모델에 전달한다."""

    requirements = {item.requirement_id: item for item in evidence_requirements}
    candidate_fields = (
        "document_id",
        "company_name",
        "position",
        "url",
        "job_category",
        "experience_min",
        "experience_max",
        "experience_text",
        "education",
        "employment_type",
        "location",
        "posted_at",
        "deadline",
        "source_platform",
        "tech_stack",
        "main_tasks",
        "requirements",
        "preferred",
        "benefits",
        "salary",
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
                "scope": requirement.scope.model_dump(mode="json"),
                "minimum_count": requirement.minimum_count,
                "required_fields": list(requirement.required_fields),
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


def select_collection_steps(
    plan: InvestigationActionPlan,
    evidence_requirements: list[EvidenceRequirement],
    collection_capabilities: list[dict[str, Any]],
) -> list[InvestigationPlanStep]:
    """LLM 계획에서 실제 제공되는 사이트와 근거를 가리키는 단계만 남긴다."""

    allowed_sites = _realtime_sites(collection_capabilities)
    requirement_ids = {item.requirement_id for item in evidence_requirements}
    normalized: list[InvestigationPlanStep] = []
    signatures: set[str] = set()
    maximum_steps = len(allowed_sites) * max(1, len(requirement_ids))
    for step in plan.steps:
        tool_name = str(step.tool_name or "")
        if tool_name != "realtime_scraping" and not tool_name.startswith(
            "realtime_scraping:"
        ):
            continue
        intent = step.arguments
        site = str(intent.site or "").strip()
        if not site or site not in allowed_sites:
            continue
        expected_tool = f"realtime_scraping:{site}"
        if tool_name not in {"realtime_scraping", expected_tool}:
            continue
        if not intent.search_keyword.strip():
            continue
        expected_evidence = list(
            dict.fromkeys(
                item for item in step.expected_evidence if item in requirement_ids
            )
        )
        if not expected_evidence:
            continue
        signature = json.dumps(
            {
                "arguments": intent.model_dump(mode="json"),
                "expected_evidence": expected_evidence,
            },
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
                    "expected_evidence": expected_evidence,
                }
            )
        )
        if len(normalized) >= maximum_steps:
            break
    return normalized


def build_single_site_collection_step(
    investigation: InvestigationRequest,
    evidence_requirements: list[EvidenceRequirement],
    collection_capabilities: list[dict[str, Any]],
) -> InvestigationPlanStep | None:
    """사이트와 검색어가 하나로 확정된 근거를 실행 단계로 옮긴다."""

    if len(evidence_requirements) != 1:
        return None
    requirement = evidence_requirements[0]
    scope = requirement.scope
    allowed_sites = _realtime_sites(collection_capabilities)
    requested_sites = list(
        dict.fromkeys(scope.sites or investigation.constraints.sites)
    )
    candidate_sites = (
        [site for site in requested_sites if site in allowed_sites]
        if requested_sites
        else sorted(allowed_sites)
    )
    if len(candidate_sites) != 1:
        return None
    search_keyword = scope.collection_search_term.strip()
    if not search_keyword:
        return None

    intent = CollectionIntent(
        original_query=investigation.original_query,
        site=candidate_sites[0],
        search_keyword=search_keyword,
        count_mode=scope.count_mode,
        target_count=scope.target_count,
        filters=JobSearchFilters(
            posted_from=scope.posted_from,
            posted_to=scope.posted_to,
            experience=scope.experience,
            location=scope.location,
            employment_type=scope.employment_type,
        ),
        freshness_required=bool(
            scope.posted_from
            or scope.posted_to
            or any(field.value == "posted_at" for field in requirement.required_fields)
        ),
        purpose=investigation.purpose.value,
        required_fields=requirement.required_fields,
    )
    return InvestigationPlanStep(
        step_id=f"collect:{requirement.requirement_id}:{candidate_sites[0]}",
        tool_name="realtime_scraping",
        arguments=intent,
        purpose=requirement.reason or requirement.description,
        expected_evidence=[requirement.requirement_id],
    )


def collection_step_signature(step: InvestigationPlanStep) -> str:
    """도구 입력과 기대 근거가 같은 수집 단계를 식별한다."""

    return json.dumps(
        {
            "arguments": step.arguments.model_dump(mode="json"),
            "expected_evidence": sorted(set(step.expected_evidence)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def new_collection_steps(
    existing_steps: list[InvestigationPlanStep],
    proposed_steps: list[InvestigationPlanStep],
) -> list[InvestigationPlanStep]:
    """기존 계획과 ID 또는 실행 의미가 겹치지 않는 단계만 반환한다."""

    known_ids = {step.step_id for step in existing_steps}
    known_signatures = {collection_step_signature(step) for step in existing_steps}
    accepted: list[InvestigationPlanStep] = []
    for step in proposed_steps:
        signature = collection_step_signature(step)
        if step.step_id in known_ids or signature in known_signatures:
            continue
        known_ids.add(step.step_id)
        known_signatures.add(signature)
        accepted.append(step)
    return accepted


def normalize_evidence_requirements(
    plan: EvidencePlan,
    taxonomy_service: Any,
) -> list[EvidenceRequirement]:
    """근거 필드를 DB 계약에 맞추고 화면 전체 수집의 표본 수를 정규화한다."""

    normalized = []
    for requirement in plan.requirements:
        updates: dict[str, Any] = {
            "required_fields": list(dict.fromkeys(requirement.required_fields))
        }
        if requirement.scope.count_mode == "visible_all":
            updates["minimum_count"] = 1
        elif requirement.scope.count_mode == "explicit":
            updates["minimum_count"] = requirement.scope.target_count
        normalized.append(
            taxonomy_service.enrich_requirement(
                requirement.model_copy(update=updates),
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
            int(document_id) for document_id in values if int(document_id) in candidates
        )
    )


def _validated_requirement_report(
    item: dict[str, Any],
    requirement: EvidenceRequirement,
    decision: Any,
) -> dict[str, Any]:
    """요구사항 하나의 모델 판정을 후보 집합에 한정해 다시 계산한다."""

    candidates = {
        int(candidate["document_id"]): candidate
        for candidate in item.get("candidates", [])
    }
    candidate_ids = decision.matching_document_ids if decision is not None else []
    if not item.get("semantic_review_required"):
        candidate_ids = list(candidates)
    matching_ids = _valid_candidate_ids(candidate_ids, candidates)
    selected_ids = matching_ids
    if requirement.scope.count_mode == "explicit":
        selected_ids = matching_ids[: requirement.scope.target_count]
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
    if (requirement.scope.posted_from or requirement.scope.posted_to) and len(
        posted_dates
    ) < requirement.minimum_count:
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
    evidence_requirements: list[EvidenceRequirement],
    validation: EvidenceValidation,
) -> dict[str, Any]:
    """모델 판단을 후보 집합 안에서만 허용하고 충분성을 다시 계산한다."""

    requirements = {item.requirement_id: item for item in evidence_requirements}
    decisions = {item.requirement_id: item for item in validation.decisions}
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
    "build_database_lookup_evidence_plan",
    "build_evidence_validation_payload",
    "build_single_site_collection_step",
    "collection_step_signature",
    "compact_db_report",
    "needs_semantic_evidence_validation",
    "new_collection_steps",
    "select_collection_steps",
    "normalize_evidence_requirements",
]

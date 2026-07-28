"""DB 자료가 조사에 필요한 근거를 충족하는지 구조적으로 검사한다."""

from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from shared.schema.agent_contract import EVIDENCE_FIELDS, EvidenceDocument
from shared.schema.investigation_schema import EvidenceRequirement, InvestigationConstraints


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _matches_exact_text_groups(
    row: sqlite3.Row,
    text_groups: list[list[str]],
) -> bool:
    normalized_groups = [
        [str(item).casefold().strip() for item in group if str(item).strip()]
        for group in text_groups
        if isinstance(group, list)
    ]
    normalized_groups = [group for group in normalized_groups if group]
    if not normalized_groups:
        return True
    haystack = " ".join(
        str(row[field] or "")
        for field in ("position", "job_category", "raw_ocr_text")
    ).casefold()
    return all(
        any(term in haystack for term in group)
        for group in normalized_groups
    )


def _rows_for_requirement(
    rows: list[sqlite3.Row],
    requirement: EvidenceRequirement,
    constraints: InvestigationConstraints,
    allowed_job_ids: set[int] | None = None,
) -> list[sqlite3.Row]:
    exact_text_groups = requirement.exact_text_groups or constraints.exact_text_groups
    sites = {
        str(site).strip().casefold()
        for site in (requirement.required_sites or constraints.sites)
        if str(site).strip()
    }
    start = _parse_date(requirement.posted_from)
    end = _parse_date(requirement.posted_to)
    matched: list[sqlite3.Row] = []
    for row in rows:
        if allowed_job_ids is not None and int(row["id"]) not in allowed_job_ids:
            continue
        source_platform = str(row["source_platform"] or "").strip().casefold()
        if sites and source_platform not in sites:
            continue
        if exact_text_groups and not _matches_exact_text_groups(row, exact_text_groups):
            continue
        posted_at = _parse_date(row["posted_at"])
        if start and (posted_at is None or posted_at < start):
            continue
        if end and (posted_at is None or posted_at > end):
            continue
        matched.append(row)
    return matched


def inspect_job_evidence(
    db_path: str | Path,
    requirements: list[EvidenceRequirement],
    constraints: InvestigationConstraints,
    *,
    document_scope_ids: list[int] | set[int] | None = None,
    force_semantic_review: bool = False,
) -> dict[str, Any]:
    """요구사항별 표본 수, 날짜 근거, 필드 충족률을 반환한다."""

    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    try:
        columns = ", ".join(("id", "source_platform", *EVIDENCE_FIELDS))
        rows = conn.execute(f"SELECT {columns} FROM jobs").fetchall()
    finally:
        conn.close()

    from agent.application.search_taxonomy_service import SearchTaxonomyService

    taxonomy = SearchTaxonomyService(db_path)

    reports: list[dict[str, Any]] = []
    all_document_ids: set[int] = set()
    document_scope = (
        None
        if document_scope_ids is None
        else {
            int(document_id)
            for document_id in document_scope_ids
            if int(document_id) > 0
        }
    )
    taxonomy_constraints = constraints.model_copy(
        update={"location": "", "experience": "", "employment_type": ""}
    )
    for requirement in requirements:
        candidate_sets: list[set[int]] = []
        resolved_occupation_keys = (
            requirement.occupation_concept_keys
            or constraints.occupation_concept_keys
        )
        domain_keys = (
            requirement.occupation_domain_concept_keys
            or constraints.occupation_domain_concept_keys
        )
        occupation_filter_keys = (
            resolved_occupation_keys
            or requirement.occupation_domain_concept_keys
            or constraints.occupation_domain_concept_keys
        )
        if occupation_filter_keys:
            candidate_sets.append(
                taxonomy.matching_occupation_job_ids(
                    occupation_filter_keys,
                    taxonomy_constraints,
                )
            )
        skill_keys = requirement.skill_concept_keys or constraints.skill_concept_keys
        if skill_keys:
            candidate_sets.append(
                taxonomy.matching_skill_job_ids(
                    skill_keys,
                    taxonomy_constraints,
                    match_mode=requirement.skill_match_mode,
                    requirement_type=requirement.skill_requirement_type,
                )
            )
        allowed_job_ids = (
            set.intersection(*candidate_sets)
            if candidate_sets
            else None
        )
        if document_scope is not None:
            allowed_job_ids = (
                set(document_scope)
                if allowed_job_ids is None
                else allowed_job_ids & document_scope
            )
        matched = _rows_for_requirement(
            rows,
            requirement,
            constraints,
            allowed_job_ids,
        )
        all_document_ids.update(int(row["id"]) for row in matched)
        field_coverage = {
            field: sum(1 for row in matched if str(row[field] or "").strip())
            for field in requirement.required_fields
            if field in EVIDENCE_FIELDS
        }
        posted_dates = [
            parsed
            for parsed in (_parse_date(row["posted_at"]) for row in matched)
            if parsed is not None
        ]
        missing: list[str] = []
        if len(matched) < requirement.minimum_count:
            missing.append(
                f"표본 {requirement.minimum_count - len(matched)}건 부족"
            )
        for field in requirement.required_fields:
            if field_coverage.get(field, 0) < requirement.minimum_count:
                missing.append(f"{field} 근거 부족")
        if (requirement.posted_from or requirement.posted_to) and len(posted_dates) < requirement.minimum_count:
            missing.append("검증된 게시일 근거 부족")
        reports.append(
            {
                "requirement_id": requirement.requirement_id,
                "description": requirement.description,
                "occupation_domain_query": requirement.occupation_domain_query,
                "occupation_domain_concept_keys": list(domain_keys),
                "occupation_query": requirement.occupation_query,
                "occupation_concept_keys": list(resolved_occupation_keys),
                "skill_queries": list(requirement.skill_queries),
                "skill_concept_keys": list(skill_keys),
                "semantic_review_required": bool(
                    force_semantic_review
                    or (requirement.occupation_query and not resolved_occupation_keys)
                    or (
                        requirement.occupation_domain_query
                        and not domain_keys
                    )
                    or (requirement.skill_queries and not skill_keys)
                    or constraints.location
                    or constraints.experience
                    or constraints.employment_type
                ),
                "matching_count": len(matched),
                "verified_posted_at_count": len(posted_dates),
                "oldest_posted_at": min(posted_dates).isoformat() if posted_dates else "",
                "newest_posted_at": max(posted_dates).isoformat() if posted_dates else "",
                "field_coverage": field_coverage,
                "site_counts": dict(Counter(str(row["source_platform"] or "unknown") for row in matched)),
                "document_ids": [int(row["id"]) for row in matched],
                "candidates": [
                    {
                        "document_id": int(row["id"]),
                        "company_name": str(row["company_name"] or ""),
                        "position": str(row["position"] or ""),
                        "job_category": str(row["job_category"] or ""),
                        "experience": str(row["experience_text"] or ""),
                        "employment_type": str(row["employment_type"] or ""),
                        "location": str(row["location"] or ""),
                        "posted_at": str(row["posted_at"] or ""),
                        "source_platform": str(row["source_platform"] or ""),
                        "tech_stack": str(row["tech_stack"] or ""),
                        "requirements": str(row["requirements"] or ""),
                        "preferred": str(row["preferred"] or ""),
                        "field_presence": {
                            field: bool(str(row[field] or "").strip())
                            for field in requirement.required_fields
                            if field in EVIDENCE_FIELDS
                        },
                    }
                    for row in matched
                ],
                "sufficient": not missing,
                "missing": missing,
            }
        )
    return {
        "total_db_rows": len(rows),
        "document_scope_ids": (
            sorted(document_scope)
            if document_scope is not None
            else None
        ),
        "requirements": reports,
        "sufficient": bool(reports) and all(item["sufficient"] for item in reports),
        "document_ids": sorted(all_document_ids),
        "missing_evidence": [
            f"{item['description']}: {reason}"
            for item in reports
            for reason in item["missing"]
        ],
    }


def load_job_evidence_documents(
    db_path: str | Path,
    document_ids: list[int],
) -> list[EvidenceDocument]:
    """검증된 ID의 공고를 답변용 구조화 문서로 조회한다."""

    ids = sorted({int(document_id) for document_id in document_ids if int(document_id) > 0})
    if not ids:
        return []
    selected_fields = (
        "id",
        "url",
        "company_name",
        "position",
        "job_category",
        "experience_text",
        "employment_type",
        "location",
        "posted_at",
        "posted_at_text",
        "tech_stack",
        "main_tasks",
        "requirements",
        "preferred",
        "benefits",
        "raw_ocr_text",
    )
    placeholders = ",".join("?" for _ in ids)
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT {', '.join(selected_fields)} FROM jobs WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        conn.close()
    return [
        EvidenceDocument.model_validate(
            {
                key: int(row[key]) if key == "id" else str(row[key] or "")
                for key in selected_fields
            }
        )
        for row in rows
    ]


__all__ = ["EVIDENCE_FIELDS", "inspect_job_evidence", "load_job_evidence_documents"]

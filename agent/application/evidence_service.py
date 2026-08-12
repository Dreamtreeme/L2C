"""DB 자료가 조사에 필요한 근거를 충족하는지 구조적으로 검사한다."""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from agent.application.search_taxonomy_service import SearchTaxonomyService
from shared.db.database import Database
from shared.schema.agent_contract import ANSWER_EVIDENCE_FIELDS
from shared.schema.jd_schema import StoredJob
from shared.schema.investigation_schema import (
    EvidenceRequirement,
)


@dataclass(frozen=True)
class EvidenceRows:
    """DB 전체 건수와 검색 가능한 구조화 공고."""

    total_db_rows: int
    indexed_rows: list[sqlite3.Row]


@dataclass(frozen=True)
class RequirementEvidence:
    """요구사항 하나에 대응하는 후보 공고와 분류 해석 결과."""

    requirement: EvidenceRequirement
    matched_rows: list[sqlite3.Row]
    occupation_keys: list[str]
    skill_keys: list[str]
    semantic_review_required: bool


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
    allowed_job_ids: set[int] | None = None,
) -> list[sqlite3.Row]:
    scope = requirement.scope
    exact_text_groups = scope.exact_text_groups
    sites = {
        str(site).strip().casefold()
        for site in scope.sites
        if str(site).strip()
    }
    start = _parse_date(scope.posted_from)
    end = _parse_date(scope.posted_to)
    maximum_required_experience = scope.maximum_required_experience_years
    matched: list[sqlite3.Row] = []
    for row in rows:
        if allowed_job_ids is not None and int(row["id"]) not in allowed_job_ids:
            continue
        source_platform = str(row["source_platform"] or "").strip().casefold()
        if sites and source_platform not in sites:
            continue
        if exact_text_groups and not _matches_exact_text_groups(row, exact_text_groups):
            continue
        if maximum_required_experience is not None:
            experience_min = row["experience_min"]
            if (
                experience_min is None
                or int(experience_min) > maximum_required_experience
            ):
                continue
        posted_at = _parse_date(row["posted_at"])
        if start and (posted_at is None or posted_at < start):
            continue
        if end and (posted_at is None or posted_at > end):
            continue
        matched.append(row)
    return matched


def _load_evidence_rows(db_path: str | Path) -> EvidenceRows:
    """검색 인덱싱이 끝난 공고만 증거 판정 대상으로 읽는다."""

    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    try:
        columns = ", ".join(
            (
                "id",
                "source_platform",
                "experience_min",
                "experience_max",
                *ANSWER_EVIDENCE_FIELDS,
            )
        )
        total_db_rows = int(
            conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        )
        rows = conn.execute(
            f"SELECT {columns} FROM jobs "
            "WHERE taxonomy_index_status = 'indexed'"
        ).fetchall()
    finally:
        conn.close()
    return EvidenceRows(total_db_rows=total_db_rows, indexed_rows=list(rows))


def _document_scope_ids(
    document_scope_ids: list[int] | set[int] | None,
) -> set[int] | None:
    if document_scope_ids is None:
        return None
    return {
        int(document_id)
        for document_id in document_scope_ids
        if int(document_id) > 0
    }


def _match_requirement_evidence(
    rows: list[sqlite3.Row],
    requirement: EvidenceRequirement,
    *,
    taxonomy: SearchTaxonomyService,
    document_scope: set[int] | None,
    force_semantic_review: bool,
) -> RequirementEvidence:
    """분류 사전과 명시 조건을 적용해 요구사항 후보를 고른다."""

    scope = requirement.scope
    taxonomy_scope = scope.model_copy(
        update={"location": "", "experience": "", "employment_type": ""}
    )
    candidate_sets: list[set[int]] = []
    semantic_occupation_ids: set[int] = set()
    occupation_keys = list(scope.occupation_concept_keys)
    if occupation_keys:
        occupation_job_ids = taxonomy.matching_occupation_job_ids(
            occupation_keys,
            taxonomy_scope,
        )
        candidate_sets.append(occupation_job_ids)
        semantic_occupation_ids = (
            taxonomy.matching_occupation_job_ids(
                occupation_keys,
                taxonomy_scope,
                evidence_fields=("main_tasks",),
            )
            & occupation_job_ids
        )
    skill_keys = list(scope.skill_concept_keys)
    if skill_keys:
        candidate_sets.append(
            taxonomy.matching_skill_job_ids(
                skill_keys,
                taxonomy_scope,
                match_mode=scope.skill_match_mode,
                requirement_type=scope.skill_requirement_type,
            )
        )
    match_mode = scope.occupation_skill_match_mode
    if not candidate_sets:
        allowed_job_ids = None
    elif match_mode == "any":
        allowed_job_ids = set.union(*candidate_sets)
    else:
        allowed_job_ids = set.intersection(*candidate_sets)
    if document_scope is not None:
        allowed_job_ids = (
            set(document_scope)
            if allowed_job_ids is None
            else allowed_job_ids & document_scope
        )
    matched_rows = _rows_for_requirement(
        rows,
        requirement,
        allowed_job_ids,
    )
    matched_row_ids = {int(row["id"]) for row in matched_rows}
    semantic_review_required = bool(
        force_semantic_review
        or semantic_occupation_ids & matched_row_ids
        or (scope.occupation_query and not occupation_keys)
        or (scope.skill_queries and not skill_keys)
        or scope.location
        or scope.experience
        or scope.employment_type
        or scope.semantic_filters
    )
    return RequirementEvidence(
        requirement=requirement,
        matched_rows=matched_rows,
        occupation_keys=occupation_keys,
        skill_keys=skill_keys,
        semantic_review_required=semantic_review_required,
    )


def _requirement_report(evidence: RequirementEvidence) -> dict[str, Any]:
    """후보 공고의 표본 수, 날짜와 필드 충족률을 보고서로 만든다."""

    requirement = evidence.requirement
    scope = requirement.scope
    matched = evidence.matched_rows
    field_coverage = {
        field: sum(1 for row in matched if str(row[field] or "").strip())
        for field in requirement.required_fields
        if field in ANSWER_EVIDENCE_FIELDS
    }
    posted_dates = [
        parsed
        for parsed in (_parse_date(row["posted_at"]) for row in matched)
        if parsed is not None
    ]
    missing: list[str] = []
    if len(matched) < requirement.minimum_count:
        missing.append(f"표본 {requirement.minimum_count - len(matched)}건 부족")
    for field in requirement.required_fields:
        if field_coverage.get(field, 0) < requirement.minimum_count:
            missing.append(f"{field} 근거 부족")
    if (
        scope.posted_from or scope.posted_to
    ) and len(posted_dates) < requirement.minimum_count:
        missing.append("검증된 게시일 근거 부족")

    return {
        "requirement_id": requirement.requirement_id,
        "description": requirement.description,
        "occupation_query": scope.occupation_query,
        "occupation_concept_keys": evidence.occupation_keys,
        "skill_queries": list(scope.skill_queries),
        "skill_concept_keys": evidence.skill_keys,
        "semantic_review_required": evidence.semantic_review_required,
        "matching_count": len(matched),
        "verified_posted_at_count": len(posted_dates),
        "oldest_posted_at": min(posted_dates).isoformat() if posted_dates else "",
        "newest_posted_at": max(posted_dates).isoformat() if posted_dates else "",
        "field_coverage": field_coverage,
        "site_counts": dict(
            Counter(str(row["source_platform"] or "unknown") for row in matched)
        ),
        "document_ids": [int(row["id"]) for row in matched],
        "candidates": [
            {
                "document_id": int(row["id"]),
                "company_name": str(row["company_name"] or ""),
                "position": str(row["position"] or ""),
                "url": str(row["url"] or ""),
                "job_category": str(row["job_category"] or ""),
                "experience_min": row["experience_min"],
                "experience_max": row["experience_max"],
                "experience_text": str(row["experience_text"] or ""),
                "education": str(row["education"] or ""),
                "employment_type": str(row["employment_type"] or ""),
                "location": str(row["location"] or ""),
                "posted_at": str(row["posted_at"] or ""),
                "deadline": str(row["deadline"] or ""),
                "source_platform": str(row["source_platform"] or ""),
                "tech_stack": str(row["tech_stack"] or ""),
                "main_tasks": str(row["main_tasks"] or ""),
                "requirements": str(row["requirements"] or ""),
                "preferred": str(row["preferred"] or ""),
                "benefits": str(row["benefits"] or ""),
                "salary": str(row["salary"] or ""),
                "field_presence": {
                    field: bool(str(row[field] or "").strip())
                    for field in requirement.required_fields
                    if field in ANSWER_EVIDENCE_FIELDS
                },
            }
            for row in matched
        ],
        "sufficient": not missing,
        "missing": missing,
    }


def inspect_job_evidence(
    db_path: str | Path,
    requirements: list[EvidenceRequirement],
    *,
    document_scope_ids: list[int] | set[int] | None = None,
    force_semantic_review: bool = False,
    taxonomy_service: SearchTaxonomyService | None = None,
) -> dict[str, Any]:
    """요구사항별 표본 수, 날짜 근거, 필드 충족률을 반환한다."""

    taxonomy = taxonomy_service or SearchTaxonomyService(db_path)
    evidence_rows = _load_evidence_rows(db_path)
    document_scope = _document_scope_ids(document_scope_ids)
    reports = [
        _requirement_report(
            _match_requirement_evidence(
                evidence_rows.indexed_rows,
                requirement,
                taxonomy=taxonomy,
                document_scope=document_scope,
                force_semantic_review=force_semantic_review,
            )
        )
        for requirement in requirements
    ]
    all_document_ids = {
        int(document_id)
        for report in reports
        for document_id in report["document_ids"]
    }

    return {
        "total_db_rows": evidence_rows.total_db_rows,
        "search_ready_db_rows": len(evidence_rows.indexed_rows),
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


def load_stored_jobs(
    db_path: str | Path,
    document_ids: list[int],
) -> list[StoredJob]:
    """검증된 ID의 공고를 SQLite 저장소에서 정규 타입으로 조회한다."""

    return Database(db_path).load_jobs(document_ids)


__all__ = ["inspect_job_evidence", "load_stored_jobs"]

"""DB 자료가 조사에 필요한 근거를 충족하는지 구조적으로 검사한다."""

from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from shared.schema.investigation_schema import EvidenceRequirement, InvestigationConstraints


EVIDENCE_FIELDS = (
    "company_name",
    "position",
    "job_category",
    "experience_text",
    "employment_type",
    "location",
    "posted_at",
    "tech_stack",
    "main_tasks",
    "requirements",
    "preferred",
    "benefits",
    "raw_ocr_text",
)


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _matches_keyword(row: sqlite3.Row, keywords: list[str]) -> bool:
    normalized = [item.casefold().strip() for item in keywords if item.strip()]
    if not normalized:
        return True
    haystack = " ".join(
        str(row[field] or "")
        for field in ("position", "job_category", "raw_ocr_text")
    ).casefold()
    return any(keyword in haystack for keyword in normalized)


def _rows_for_requirement(
    rows: list[sqlite3.Row],
    requirement: EvidenceRequirement,
    constraints: InvestigationConstraints,
) -> list[sqlite3.Row]:
    keywords = requirement.search_keywords or constraints.search_keywords
    sites = set(requirement.required_sites or constraints.sites)
    start = _parse_date(requirement.posted_from)
    end = _parse_date(requirement.posted_to)
    matched: list[sqlite3.Row] = []
    for row in rows:
        if sites and str(row["source_platform"] or "") not in sites:
            continue
        if not _matches_keyword(row, keywords):
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
) -> dict[str, Any]:
    """요구사항별 표본 수, 날짜 근거, 필드 충족률을 반환한다."""

    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    try:
        columns = ", ".join(("id", "source_platform", *EVIDENCE_FIELDS))
        rows = conn.execute(f"SELECT {columns} FROM jobs").fetchall()
    finally:
        conn.close()

    reports: list[dict[str, Any]] = []
    all_document_ids: set[int] = set()
    for requirement in requirements:
        matched = _rows_for_requirement(rows, requirement, constraints)
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
                "matching_count": len(matched),
                "verified_posted_at_count": len(posted_dates),
                "oldest_posted_at": min(posted_dates).isoformat() if posted_dates else "",
                "newest_posted_at": max(posted_dates).isoformat() if posted_dates else "",
                "field_coverage": field_coverage,
                "site_counts": dict(Counter(str(row["source_platform"] or "unknown") for row in matched)),
                "document_ids": [int(row["id"]) for row in matched],
                "sufficient": not missing,
                "missing": missing,
            }
        )
    return {
        "total_db_rows": len(rows),
        "requirements": reports,
        "sufficient": bool(reports) and all(item["sufficient"] for item in reports),
        "document_ids": sorted(all_document_ids),
        "missing_evidence": [
            f"{item['description']}: {reason}"
            for item in reports
            for reason in item["missing"]
        ],
    }


__all__ = ["inspect_job_evidence"]

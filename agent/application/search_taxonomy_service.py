"""검색 의미 사전 연결, 범위 해석, 카디널리티 계산을 제공한다."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from agent.application.search_taxonomy_import_service import normalize_term
from agent.application.search_taxonomy_utils import (
    CORE_SOURCE_KEY,
    contains_taxonomy_alias,
)
from shared.schema.investigation_schema import (
    EvidenceRequirement,
    InvestigationConstraints,
)

DEFAULT_LOCAL_SEED = (
    Path(__file__).resolve().parents[2] / "data" / "samples" / "search_taxonomy_ko.json"
)


class SearchTaxonomyService:
    """SQLite 사전을 기준으로 공고와 사용자 검색 범위를 연결한다."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _specific_alias_concept_keys(
        matches: Iterable[tuple[str, str]],
    ) -> list[str]:
        """더 긴 별칭에 포함된 일반 별칭을 제외하고 독립 개념을 남긴다."""

        candidates = list(matches)
        return list(
            dict.fromkeys(
                concept_key
                for alias, concept_key in candidates
                if not any(
                    alias != other_alias
                    and contains_taxonomy_alias(other_alias, alias)
                    for other_alias, _other_key in candidates
                )
            )
        )

    def resolve_occupation_concepts(self, occupation_query: str) -> list[str]:
        """사용자 표현에서 서로 독립적인 구체 직무를 모두 찾는다."""

        text = normalize_term(occupation_query)
        if not text:
            return []
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT c.concept_key, a.normalized_alias
                FROM search_concepts AS c
                JOIN search_aliases AS a ON a.concept_id = c.id
                WHERE c.concept_type = 'occupation'
                  AND c.status = 'active'
                  AND c.source_key = ?
                  AND a.source_key = ?
                  AND a.active = 1
                """,
                (CORE_SOURCE_KEY, CORE_SOURCE_KEY),
            ).fetchall()
        finally:
            connection.close()

        exact_matches = [row for row in rows if text == str(row["normalized_alias"])]
        if exact_matches:
            return list(
                dict.fromkeys(str(row["concept_key"]) for row in exact_matches)
            )

        matches: list[tuple[str, str]] = []
        for row in rows:
            alias = str(row["normalized_alias"])
            if contains_taxonomy_alias(text, alias):
                matches.append((alias, str(row["concept_key"])))
        if not matches:
            return []
        return self._specific_alias_concept_keys(matches)

    def resolve_skill_concepts(self, skill_queries: Iterable[str]) -> list[str]:
        """기술 표현 안의 구체적인 활성 별칭을 해석한다."""

        terms = {normalize_term(item) for item in skill_queries if normalize_term(item)}
        if not terms:
            return []
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT c.concept_key, a.normalized_alias
                FROM search_concepts AS c
                JOIN search_aliases AS a ON a.concept_id = c.id
                WHERE c.concept_type = 'skill'
                  AND c.status = 'active'
                  AND c.source_key = ?
                  AND a.source_key = ?
                  AND a.active = 1
                """,
                (CORE_SOURCE_KEY, CORE_SOURCE_KEY),
            ).fetchall()
        finally:
            connection.close()
        resolved: list[str] = []
        for term in sorted(terms):
            candidates = [
                row for row in rows if str(row["normalized_alias"]) == term
            ]
            if candidates:
                resolved.extend(str(row["concept_key"]) for row in candidates)
                continue
            contained = [
                (str(row["normalized_alias"]), str(row["concept_key"]))
                for row in rows
                if contains_taxonomy_alias(term, str(row["normalized_alias"]))
            ]
            resolved.extend(self._specific_alias_concept_keys(contained))
        return list(dict.fromkeys(resolved))

    def _concept_ids(
        self, connection: sqlite3.Connection, concept_keys: Iterable[str]
    ) -> list[int]:
        keys = list(dict.fromkeys(str(key) for key in concept_keys if str(key)))
        if not keys:
            return []
        placeholders = ",".join("?" for _ in keys)
        return [
            int(row["id"])
            for row in connection.execute(
                f"SELECT id FROM search_concepts WHERE concept_key IN ({placeholders})",
                keys,
            ).fetchall()
        ]

    @staticmethod
    def _job_filter_clauses(
        constraints: InvestigationConstraints,
    ) -> tuple[list[str], list[Any]]:
        where: list[str] = ["jobs.taxonomy_index_status = 'indexed'"]
        params: list[Any] = []
        if constraints.sites:
            placeholders = ",".join("?" for _ in constraints.sites)
            where.append(
                f"LOWER(COALESCE(jobs.source_platform, '')) IN ({placeholders})"
            )
            params.extend(str(site).casefold() for site in constraints.sites)
        if constraints.posted_from:
            where.append("date(jobs.posted_at) >= date(?)")
            params.append(constraints.posted_from)
        if constraints.posted_to:
            where.append("date(jobs.posted_at) <= date(?)")
            params.append(constraints.posted_to)
        if constraints.location:
            where.append("LOWER(COALESCE(jobs.location, '')) LIKE ?")
            params.append(f"%{constraints.location.casefold()}%")
        if constraints.employment_type:
            where.append("LOWER(COALESCE(jobs.employment_type, '')) LIKE ?")
            params.append(f"%{constraints.employment_type.casefold()}%")
        return where, params

    def matching_occupation_job_ids(
        self,
        concept_keys: Iterable[str],
        constraints: InvestigationConstraints | None = None,
        *,
        evidence_fields: Iterable[str] | None = None,
    ) -> set[int]:
        """선택한 직무 별칭에 연결된 공고 ID를 반환한다."""

        connection = self._connect()
        try:
            concept_ids = self._concept_ids(connection, concept_keys)
            if not concept_ids:
                return set()
            placeholders = ",".join("?" for _ in concept_ids)
            constraints = constraints or InvestigationConstraints()
            filters, filter_params = self._job_filter_clauses(constraints)
            where = ["links.link_type = 'occupation'"]
            params: list[Any] = [*concept_ids]
            fields = list(
                dict.fromkeys(
                    str(field).strip()
                    for field in (evidence_fields or [])
                    if str(field).strip()
                )
            )
            if fields:
                field_placeholders = ",".join("?" for _ in fields)
                where.append(f"links.evidence_field IN ({field_placeholders})")
                params.extend(fields)
            where.extend(filters)
            params.extend(filter_params)
            rows = connection.execute(
                f"""
                SELECT DISTINCT jobs.id
                FROM jobs
                JOIN job_concept_links AS links ON links.job_id = jobs.id
                WHERE links.concept_id IN ({placeholders})
                  AND {" AND ".join(where)}
                """,
                params,
            ).fetchall()
            return {int(row["id"]) for row in rows}
        finally:
            connection.close()

    def matching_skill_job_ids(
        self,
        concept_keys: Iterable[str],
        constraints: InvestigationConstraints | None = None,
        *,
        match_mode: str = "all",
        requirement_type: str = "any",
    ) -> set[int]:
        """구조화된 기술 근거를 기준으로 공고 ID를 반환한다."""

        connection = self._connect()
        try:
            concept_ids = self._concept_ids(connection, concept_keys)
            if not concept_ids:
                return set()
            placeholders = ",".join("?" for _ in concept_ids)
            constraints = constraints or InvestigationConstraints()
            filters, filter_params = self._job_filter_clauses(constraints)
            where = [
                "links.link_type = 'skill'",
                f"links.concept_id IN ({placeholders})",
            ]
            params: list[Any] = [*concept_ids]
            if requirement_type != "any":
                where.append("links.requirement_type = ?")
                params.append(requirement_type)
            where.extend(filters)
            params.extend(filter_params)
            having = ""
            if match_mode == "all":
                having = "HAVING COUNT(DISTINCT links.concept_id) = ?"
                params.append(len(concept_ids))
            rows = connection.execute(
                f"""
                SELECT jobs.id
                FROM jobs
                JOIN job_concept_links AS links ON links.job_id = jobs.id
                WHERE {" AND ".join(where)}
                GROUP BY jobs.id
                {having}
                """,
                params,
            ).fetchall()
            return {int(row["id"]) for row in rows}
        finally:
            connection.close()

    def enrich_constraints(
        self, constraints: InvestigationConstraints
    ) -> InvestigationConstraints:
        """사용자 표현을 직무·기술 개념으로 해석하고 수집 검색어를 확정한다."""

        updates: dict[str, Any] = {}
        if not constraints.collection_search_term:
            collection_term = (
                constraints.occupation_query
                or next(iter(constraints.skill_queries), "")
            )
            if collection_term:
                updates["collection_search_term"] = collection_term
        if not constraints.occupation_concept_keys and constraints.occupation_query:
            occupation_keys = self.resolve_occupation_concepts(
                constraints.occupation_query
            )
            if occupation_keys:
                updates["occupation_concept_keys"] = occupation_keys
        if not constraints.skill_concept_keys and constraints.skill_queries:
            updates["skill_concept_keys"] = self.resolve_skill_concepts(
                constraints.skill_queries
            )
        return constraints.model_copy(update=updates) if updates else constraints

    def enrich_requirement(
        self,
        requirement: EvidenceRequirement,
    ) -> EvidenceRequirement:
        """근거 집단의 단일 scope를 검색 사전 기준으로 정규화한다."""

        return requirement.model_copy(
            update={"scope": self.enrich_constraints(requirement.scope)}
        )


__all__ = [
    "DEFAULT_LOCAL_SEED",
    "SearchTaxonomyService",
]

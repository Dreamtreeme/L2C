"""검색 의미 사전 연결, 범위 해석, 카디널리티 계산을 제공한다."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from agent.application.search_taxonomy_import_service import normalize_term
from agent.application.search_taxonomy_utils import (
    CORE_SOURCE_KEY,
    CURATED_SOURCE_KEY,
    contains_taxonomy_alias,
    taxonomy_timestamp,
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

    @staticmethod
    def _concept_label(row: sqlite3.Row) -> str:
        return str(
            row["preferred_label_ko"] or row["preferred_label_en"] or row["concept_key"]
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
                SELECT c.concept_key, c.source_key, a.normalized_alias
                FROM search_concepts AS c
                JOIN search_aliases AS a ON a.concept_id = c.id
                WHERE c.concept_type = 'occupation'
                  AND c.status = 'active'
                  AND a.active = 1
                """
            ).fetchall()
        finally:
            connection.close()

        exact_matches = [row for row in rows if text == str(row["normalized_alias"])]
        if exact_matches:
            local_matches = [
                row
                for row in exact_matches
                if str(row["source_key"]) in {CORE_SOURCE_KEY, CURATED_SOURCE_KEY}
            ]
            selected = local_matches or exact_matches
            return list(dict.fromkeys(str(row["concept_key"]) for row in selected))

        matches: list[tuple[str, str]] = []
        for row in rows:
            if str(row["source_key"]) not in {CORE_SOURCE_KEY, CURATED_SOURCE_KEY}:
                continue
            alias = str(row["normalized_alias"])
            if contains_taxonomy_alias(text, alias):
                matches.append((alias, str(row["concept_key"])))
        if not matches:
            return []
        return self._specific_alias_concept_keys(matches)

    def resolve_domain_concepts(self, domain_query: str) -> list[str]:
        """명시된 업무 영역 표현을 검토된 정확 별칭으로 해석한다."""

        normalized = normalize_term(domain_query)
        if not normalized:
            return []
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT DISTINCT concepts.concept_key
                FROM search_concepts AS concepts
                JOIN search_aliases AS aliases ON aliases.concept_id = concepts.id
                WHERE concepts.concept_type = 'domain'
                  AND concepts.status = 'active'
                  AND concepts.source_key IN (?, ?)
                  AND aliases.active = 1
                  AND aliases.normalized_alias = ?
                """,
                (CORE_SOURCE_KEY, CURATED_SOURCE_KEY, normalized),
            ).fetchall()
            return [str(row["concept_key"]) for row in rows]
        finally:
            connection.close()

    def resolve_skill_concepts(self, skill_queries: Iterable[str]) -> list[str]:
        """기술 표현 안의 구체적인 활성 별칭을 해석한다."""

        terms = {normalize_term(item) for item in skill_queries if normalize_term(item)}
        if not terms:
            return []
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT c.concept_key, c.source_key, a.normalized_alias
                FROM search_concepts AS c
                JOIN search_aliases AS a ON a.concept_id = c.id
                WHERE c.concept_type = 'skill'
                  AND c.status = 'active'
                  AND a.active = 1
                """
            ).fetchall()
        finally:
            connection.close()
        resolved: list[str] = []
        for term in sorted(terms):
            candidates = [
                row for row in rows if str(row["normalized_alias"]) == term
            ]
            local = [
                row
                for row in candidates
                if row["source_key"] in {CORE_SOURCE_KEY, CURATED_SOURCE_KEY}
            ]
            selected = local or candidates
            if selected:
                resolved.extend(str(row["concept_key"]) for row in selected)
                continue
            contained = [
                (str(row["normalized_alias"]), str(row["concept_key"]))
                for row in rows
                if row["source_key"] in {CORE_SOURCE_KEY, CURATED_SOURCE_KEY}
                and contains_taxonomy_alias(term, str(row["normalized_alias"]))
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
        """선택한 직무와 모든 하위 직무에 연결된 공고 ID를 반환한다."""

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
                WITH RECURSIVE selected_concepts(id) AS (
                    SELECT id FROM search_concepts WHERE id IN ({placeholders})
                    UNION
                    SELECT relations.source_concept_id
                    FROM search_concept_relations AS relations
                    JOIN selected_concepts
                      ON relations.target_concept_id = selected_concepts.id
                    WHERE relations.relation_type = 'broader'
                )
                SELECT DISTINCT jobs.id
                FROM jobs
                JOIN job_concept_links AS links ON links.job_id = jobs.id
                JOIN selected_concepts ON selected_concepts.id = links.concept_id
                WHERE {" AND ".join(where)}
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

    def list_direct_children(self, concept_key: str) -> list[dict[str, str]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT child.concept_key, child.preferred_label_ko,
                       child.preferred_label_en, child.concept_type,
                       child.definition, child.source_key
                FROM search_concepts AS parent
                JOIN search_concept_relations AS relations
                  ON relations.target_concept_id = parent.id
                 AND relations.relation_type = 'broader'
                JOIN search_concepts AS child
                  ON child.id = relations.source_concept_id
                WHERE parent.concept_key = ?
                  AND child.status = 'active'
                  AND child.source_key IN (?, ?)
                ORDER BY COALESCE(child.preferred_label_ko, child.preferred_label_en)
                """,
                (concept_key, CORE_SOURCE_KEY, CURATED_SOURCE_KEY),
            ).fetchall()
            return [
                {
                    "concept_key": str(row["concept_key"]),
                    "label": self._concept_label(row),
                    "concept_type": str(row["concept_type"]),
                    "definition": str(row["definition"] or ""),
                    "source": str(row["source_key"]),
                }
                for row in rows
            ]
        finally:
            connection.close()

    def occupation_descendant_count(self, concept_key: str) -> int:
        """선택한 노드 아래의 활성 직무 개념 수를 반환한다."""

        connection = self._connect()
        try:
            row = connection.execute(
                """
                WITH RECURSIVE descendants(id) AS (
                    SELECT id FROM search_concepts WHERE concept_key = ?
                    UNION
                    SELECT relations.source_concept_id
                    FROM search_concept_relations AS relations
                    JOIN descendants
                      ON relations.target_concept_id = descendants.id
                    WHERE relations.relation_type = 'broader'
                )
                SELECT COUNT(*) AS concept_count
                FROM search_concepts AS concepts
                JOIN descendants ON descendants.id = concepts.id
                WHERE concepts.concept_type = 'occupation'
                  AND concepts.status = 'active'
                """,
                (concept_key,),
            ).fetchone()
            return int(row["concept_count"] if row is not None else 0)
        finally:
            connection.close()

    def occupation_resolution_candidates(
        self,
        domain_concept_keys: Iterable[str],
    ) -> list[dict[str, Any]]:
        """선택된 업무 영역 아래의 활성 직무만 의미 판정 후보로 반환한다."""

        keys = list(dict.fromkeys(str(key) for key in domain_concept_keys if str(key)))
        if not keys:
            return []
        connection = self._connect()
        try:
            concept_ids = self._concept_ids(connection, keys)
            if not concept_ids:
                return []
            placeholders = ",".join("?" for _ in concept_ids)
            rows = connection.execute(
                f"""
                WITH RECURSIVE branch(id) AS (
                    SELECT id FROM search_concepts WHERE id IN ({placeholders})
                    UNION
                    SELECT relations.source_concept_id
                    FROM search_concept_relations AS relations
                    JOIN branch ON relations.target_concept_id = branch.id
                    WHERE relations.relation_type = 'broader'
                )
                SELECT concepts.concept_key, concepts.preferred_label_ko,
                       concepts.preferred_label_en, concepts.definition,
                       concepts.source_key,
                       EXISTS (
                           SELECT 1
                           FROM search_concept_relations AS child_relations
                           JOIN search_concepts AS child
                             ON child.id = child_relations.source_concept_id
                           WHERE child_relations.target_concept_id = concepts.id
                             AND child_relations.relation_type = 'broader'
                             AND child.status = 'active'
                       ) AS is_group
                FROM search_concepts AS concepts
                JOIN branch ON branch.id = concepts.id
                WHERE concepts.concept_type = 'occupation'
                  AND concepts.status = 'active'
                ORDER BY COALESCE(
                    concepts.preferred_label_ko,
                    concepts.preferred_label_en,
                    concepts.concept_key
                )
                """,
                concept_ids,
            ).fetchall()
            return [
                {
                    "concept_key": str(row["concept_key"]),
                    "label": self._concept_label(row),
                    "definition": str(row["definition"] or ""),
                    "source": str(row["source_key"]),
                    "is_group": bool(row["is_group"]),
                }
                for row in rows
            ]
        finally:
            connection.close()

    def record_occupation_candidate(
        self,
        term: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """선택된 영역에서도 확정하지 못한 직무 표현을 검토 후보로 남긴다."""

        normalized = normalize_term(term)
        if not normalized:
            return
        now = taxonomy_timestamp()
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    INSERT INTO search_term_candidates (
                        normalized_term, display_term, proposed_type, status,
                        observation_count, first_seen_at, last_seen_at,
                        sample_job_id, metadata_json
                    ) VALUES (?, ?, 'occupation', 'candidate', 1, ?, ?, NULL, ?)
                    ON CONFLICT(normalized_term, proposed_type) DO UPDATE SET
                        display_term = excluded.display_term,
                        observation_count = search_term_candidates.observation_count + 1,
                        last_seen_at = excluded.last_seen_at,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        normalized,
                        term.strip(),
                        now,
                        now,
                        json.dumps(metadata or {}, ensure_ascii=False),
                    ),
                )
        finally:
            connection.close()

    def concept_label(self, concept_key: str) -> str:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT concept_key, preferred_label_ko, preferred_label_en
                FROM search_concepts WHERE concept_key = ?
                """,
                (concept_key,),
            ).fetchone()
            return self._concept_label(row) if row is not None else concept_key
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
                or constraints.occupation_domain_query
                or next(iter(constraints.skill_queries), "")
            )
            if collection_term:
                updates["collection_search_term"] = collection_term
        if (
            not constraints.occupation_domain_concept_keys
            and constraints.occupation_domain_query
        ):
            updates["occupation_domain_concept_keys"] = self.resolve_domain_concepts(
                constraints.occupation_domain_query
            )
        if not constraints.occupation_concept_keys and constraints.occupation_query:
            occupation_keys = self.resolve_occupation_concepts(
                constraints.occupation_query
            )
            if occupation_keys:
                updates["occupation_concept_keys"] = occupation_keys
                updates["occupation_resolution"] = "exact_alias"
        if not constraints.skill_concept_keys and constraints.skill_queries:
            updates["skill_concept_keys"] = self.resolve_skill_concepts(
                constraints.skill_queries
            )
        return constraints.model_copy(update=updates) if updates else constraints

    def enrich_requirement(
        self,
        requirement: EvidenceRequirement,
        constraints: InvestigationConstraints,
    ) -> EvidenceRequirement:
        """근거 집단을 확정된 조사 조건과 같은 사전 기준으로 정규화한다."""

        domain_query = (
            requirement.occupation_domain_query or constraints.occupation_domain_query
        )
        domain_keys = list(requirement.occupation_domain_concept_keys)
        if not domain_keys:
            domain_keys = list(constraints.occupation_domain_concept_keys)
        if not domain_keys and domain_query:
            domain_keys = self.resolve_domain_concepts(domain_query)
        occupation_query = requirement.occupation_query or constraints.occupation_query
        occupation_keys = list(requirement.occupation_concept_keys)
        if not occupation_keys and occupation_query:
            occupation_keys = self.resolve_occupation_concepts(occupation_query)
        skill_queries = requirement.skill_queries or constraints.skill_queries
        skill_keys = list(requirement.skill_concept_keys)
        if not skill_keys and skill_queries:
            skill_keys = self.resolve_skill_concepts(skill_queries)
        return requirement.model_copy(
            update={
                "occupation_domain_query": domain_query,
                "occupation_domain_concept_keys": domain_keys,
                "occupation_query": occupation_query,
                "occupation_concept_keys": occupation_keys,
                "collection_search_term": (
                    requirement.collection_search_term
                    or occupation_query
                    or domain_query
                    or constraints.collection_search_term
                    or next(iter(skill_queries), "")
                ),
                "skill_queries": list(skill_queries),
                "skill_concept_keys": skill_keys,
                "skill_match_mode": (
                    requirement.skill_match_mode
                    if requirement.skill_queries or requirement.skill_concept_keys
                    else constraints.skill_match_mode
                ),
                "skill_requirement_type": (
                    requirement.skill_requirement_type
                    if requirement.skill_queries or requirement.skill_concept_keys
                    else constraints.skill_requirement_type
                ),
                "exact_text_groups": (
                    requirement.exact_text_groups or constraints.exact_text_groups
                ),
            }
        )


__all__ = [
    "DEFAULT_LOCAL_SEED",
    "SearchTaxonomyService",
]

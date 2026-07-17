"""검색 의미 사전 연결, 범위 해석, 카디널리티 계산을 제공한다."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from agent.application.search_taxonomy_import_service import import_local_seed, normalize_term
from agent.application.search_taxonomy_constants import (
    CORE_SOURCE_KEY,
    CURATED_SOURCE_KEY,
)
from shared.db.database import Database
from shared.schema.investigation_schema import (
    ClarificationOption,
    ClarificationQuestion,
    EvidenceRequirement,
    InvestigationConstraints,
)


LOCAL_SOURCE_KEY = CORE_SOURCE_KEY
OCCUPATION_DOMAIN_ROOT_KEY = "l2c:domain:occupation"
DEFAULT_LOCAL_SEED = (
    Path(__file__).resolve().parents[2] / "data" / "samples" / "search_taxonomy_ko.json"
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _contains_alias(text: str, alias: str) -> bool:
    """영문 식별자가 더 긴 토큰 일부로 잘못 일치하지 않도록 경계를 확인한다."""

    if not text or not alias:
        return False
    start = 0
    while True:
        index = text.find(alias, start)
        if index < 0:
            return False
        end = index + len(alias)
        left_ok = index == 0 or not (text[index - 1].isalnum() and alias[0].isalnum())
        right_ok = end == len(text) or not (text[end].isalnum() and alias[-1].isalnum())
        if left_ok and right_ok:
            return True
        start = index + 1


class SearchTaxonomyService:
    """SQLite 사전을 기준으로 공고와 사용자 검색 범위를 연결한다."""

    def __init__(self, db_path: str | Path, *, ensure_local_seed: bool = True):
        self.db_path = Path(db_path)
        self._occupation_alias_cache: list[sqlite3.Row] | None = None
        self._skill_alias_cache: list[sqlite3.Row] | None = None
        Database(self.db_path)
        if ensure_local_seed and DEFAULT_LOCAL_SEED.exists():
            payload = json.loads(DEFAULT_LOCAL_SEED.read_text(encoding="utf-8"))
            expected_version = str(payload.get("source", {}).get("version") or "")
            connection = self._connect()
            try:
                source = connection.execute(
                    "SELECT version FROM taxonomy_sources WHERE source_key = ?",
                    (LOCAL_SOURCE_KEY,),
                ).fetchone()
            finally:
                connection.close()
            if source is None or str(source["version"]) != expected_version:
                import_local_seed(self.db_path, DEFAULT_LOCAL_SEED)
                self.relink_all_jobs()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _concept_label(row: sqlite3.Row) -> str:
        return str(row["preferred_label_ko"] or row["preferred_label_en"] or row["concept_key"])

    def resolve_occupation_concepts(self, occupation_query: str) -> list[str]:
        """사용자 직무 표현과 가장 구체적으로 일치하는 검토된 직무를 찾는다."""

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

        matches: list[tuple[int, str]] = []
        for row in rows:
            if str(row["source_key"]) not in {CORE_SOURCE_KEY, CURATED_SOURCE_KEY}:
                continue
            alias = str(row["normalized_alias"])
            if _contains_alias(text, alias):
                matches.append((len(alias), str(row["concept_key"])))
        if not matches:
            return []
        longest = max(length for length, _key in matches)
        return list(dict.fromkeys(key for length, key in matches if length == longest))

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
        """명시된 기술 표현을 정확한 활성 별칭으로 해석한다."""

        terms = {
            normalize_term(item)
            for item in skill_queries
            if normalize_term(item)
        }
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT c.concept_key, c.source_key, a.normalized_alias
                FROM search_concepts AS c
                JOIN search_aliases AS a ON a.concept_id = c.id
                WHERE c.concept_type = 'skill'
                  AND c.status = 'active'
                  AND a.active = 1
                  AND a.normalized_alias IN ({placeholders})
                """,
                sorted(terms),
            ).fetchall()
        finally:
            connection.close()
        by_term: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_term.setdefault(str(row["normalized_alias"]), []).append(row)
        resolved: list[str] = []
        for term in sorted(terms):
            candidates = by_term.get(term, [])
            local = [
                row
                for row in candidates
                if row["source_key"] in {CORE_SOURCE_KEY, CURATED_SOURCE_KEY}
            ]
            selected = local or candidates
            resolved.extend(str(row["concept_key"]) for row in selected)
        return list(dict.fromkeys(resolved))

    def _concept_ids(self, connection: sqlite3.Connection, concept_keys: Iterable[str]) -> list[int]:
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
        where: list[str] = []
        params: list[Any] = []
        if constraints.sites:
            placeholders = ",".join("?" for _ in constraints.sites)
            where.append(f"LOWER(COALESCE(jobs.source_platform, '')) IN ({placeholders})")
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
            where = ["links.link_type = 'occupation'", *filters]
            params: list[Any] = [*concept_ids, *filter_params]
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
                WHERE {' AND '.join(where)}
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
                WHERE {' AND '.join(where)}
                GROUP BY jobs.id
                {having}
                """,
                params,
            ).fetchall()
            return {int(row["id"]) for row in rows}
        finally:
            connection.close()

    def _direct_children(self, concept_key: str) -> list[dict[str, str]]:
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
        now = _now()
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

    def enrich_constraints(self, constraints: InvestigationConstraints) -> InvestigationConstraints:
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
            requirement.occupation_domain_query
            or constraints.occupation_domain_query
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

    @staticmethod
    def _concept_option_id(concept_key: str) -> str:
        digest = hashlib.sha1(concept_key.encode("utf-8")).hexdigest()[:10]
        return f"concept-{digest}"

    def _counted_scope_item(
        self,
        item: dict[str, str],
        constraints: InvestigationConstraints,
    ) -> dict[str, Any]:
        concept_key = str(item["concept_key"])
        return {
            **item,
            "matching_count": len(
                self.matching_occupation_job_ids([concept_key], constraints)
            ),
            "concept_count": self.occupation_descendant_count(concept_key),
        }

    def build_domain_question(
        self,
        constraints: InvestigationConstraints,
        *,
        answered_question_ids: Iterable[str] = (),
    ) -> ClarificationQuestion | None:
        """일반 공고 요청에 업무 기능 기준 최상위 영역을 제시한다."""

        question_id = "occupation_domain"
        if question_id in set(answered_question_ids):
            return None
        domains = [
            item
            for item in self._direct_children(OCCUPATION_DOMAIN_ROOT_KEY)
            if item["concept_type"] == "domain"
        ]
        if not domains:
            return None
        counted = [self._counted_scope_item(item, constraints) for item in domains]
        options = [
            ClarificationOption(
                option_id=self._concept_option_id(str(item["concept_key"])),
                label=str(item["label"]),
                value=str(item["concept_key"]),
                collection_search_term=str(item["label"]),
                matching_count=int(item["matching_count"]),
                concept_count=int(item["concept_count"]),
                description=(
                    f"저장 공고 {item['matching_count']}건 · "
                    f"사전 직무 {item['concept_count']}개"
                ),
            )
            for item in counted
        ]
        return ClarificationQuestion(
            question_id=question_id,
            field="occupation_domain_concept_keys",
            question="어떤 업무 영역의 채용공고를 찾을까요?",
            options=options,
            allow_custom=True,
            reason=(
                "회사의 업종이 아니라 실제 수행할 업무를 기준으로 선택합니다. "
                "원하는 직무가 명확하면 직접 입력할 수 있습니다."
            ),
            candidate_count=len(
                self.matching_occupation_job_ids(
                    [OCCUPATION_DOMAIN_ROOT_KEY],
                    constraints,
                )
            ),
            concept_count=self.occupation_descendant_count(
                OCCUPATION_DOMAIN_ROOT_KEY
            ),
            facet_type="occupation_domain",
        )

    def _family_children(self, domain_key: str) -> list[dict[str, str]]:
        children = [
            item
            for item in self._direct_children(domain_key)
            if item["concept_type"] == "occupation"
        ]
        if len(children) != 1:
            return children
        nested = [
            item
            for item in self._direct_children(str(children[0]["concept_key"]))
            if item["concept_type"] == "occupation"
        ]
        return [children[0], *nested] if len(nested) >= 2 else children

    def build_family_question(
        self,
        constraints: InvestigationConstraints,
        *,
        answered_question_ids: Iterable[str] = (),
    ) -> ClarificationQuestion | None:
        """선택된 업무 영역 아래의 검토된 직무군을 모두 제시한다."""

        domain_keys = list(constraints.occupation_domain_concept_keys)
        if not domain_keys:
            return None
        fingerprint = hashlib.sha1("|".join(sorted(domain_keys)).encode("utf-8")).hexdigest()[:10]
        question_id = f"occupation_family:{fingerprint}"
        if question_id in set(answered_question_ids):
            return None
        families: dict[str, dict[str, str]] = {}
        for domain_key in domain_keys:
            for item in self._family_children(domain_key):
                families[str(item["concept_key"])] = item
        if not families:
            return None
        counted = [
            self._counted_scope_item(item, constraints)
            for item in families.values()
        ]
        counted.sort(
            key=lambda item: (-int(item["matching_count"]), str(item["label"]))
        )
        options = [
            ClarificationOption(
                option_id=self._concept_option_id(str(item["concept_key"])),
                label=str(item["label"]),
                value=str(item["concept_key"]),
                collection_search_term=str(item["label"]),
                matching_count=int(item["matching_count"]),
                concept_count=int(item["concept_count"]),
                description=(
                    f"저장 공고 {item['matching_count']}건 · "
                    f"사전 직무 {item['concept_count']}개"
                ),
            )
            for item in counted
        ]
        domain_labels = ", ".join(self.concept_label(key) for key in domain_keys)
        return ClarificationQuestion(
            question_id=question_id,
            field="occupation_concept_keys",
            question=f"{domain_labels} 중 어떤 직무군을 찾을까요?",
            options=options,
            allow_custom=True,
            reason=(
                "상위 직무군은 모든 하위 직무를 포함합니다. 더 좁은 직무군을 "
                "선택하거나 원하는 직무명을 직접 입력할 수 있습니다."
            ),
            candidate_count=len(
                self.matching_occupation_job_ids(domain_keys, constraints)
            ),
            concept_count=len(self.occupation_resolution_candidates(domain_keys)),
            facet_type="occupation_family",
        )

    def build_next_scope_question(
        self,
        constraints: InvestigationConstraints,
        *,
        answered_question_ids: Iterable[str] = (),
    ) -> ClarificationQuestion | None:
        """현재 확정 수준에 맞는 다음 직무 범위 질문 하나를 만든다."""

        answered = tuple(answered_question_ids)
        if not constraints.occupation_concept_keys:
            if (
                not constraints.occupation_domain_concept_keys
                and constraints.occupation_scope_required
                and not constraints.occupation_query
            ):
                return self.build_domain_question(
                    constraints,
                    answered_question_ids=answered,
                )
            if (
                constraints.occupation_domain_concept_keys
                and not constraints.occupation_query
            ):
                if constraints.occupation_scope_mode == "all":
                    return None
                return self.build_family_question(
                    constraints,
                    answered_question_ids=answered,
                )
            return None
        return self.build_scope_question(
            constraints,
            answered_question_ids=answered,
        )

    def build_scope_question(
        self,
        constraints: InvestigationConstraints,
        *,
        answered_question_ids: Iterable[str] = (),
    ) -> ClarificationQuestion | None:
        """실제 공고가 있는 하위 직무만 카디널리티와 함께 제시한다."""

        if constraints.occupation_scope_mode == "all":
            return None
        answered = set(answered_question_ids)
        for concept_key in constraints.occupation_concept_keys:
            question_id = f"occupation_scope:{concept_key}"
            if question_id in answered:
                continue
            children = self._direct_children(concept_key)
            counted = [
                {
                    **child,
                    "count": len(
                        self.matching_occupation_job_ids(
                            [child["concept_key"]],
                            constraints,
                        )
                    ),
                }
                for child in children
            ]
            counted = [item for item in counted if item["count"] > 0]
            if len(counted) < 2:
                continue
            counted.sort(key=lambda item: (-int(item["count"]), str(item["label"])))
            total_count = len(
                self.matching_occupation_job_ids([concept_key], constraints)
            )
            options = [
                ClarificationOption(
                    option_id=self._concept_option_id(str(item["concept_key"])),
                    label=f"{item['label']} ({item['count']}건)",
                    value=str(item["concept_key"]),
                    collection_search_term=str(item["label"]),
                    matching_count=int(item["count"]),
                    concept_count=self.occupation_descendant_count(
                        str(item["concept_key"])
                    ),
                    description=(
                        f"현재 조건에서 이 직무로 연결된 공고 {item['count']}건"
                    ),
                )
                for item in counted
            ]
            options.append(
                ClarificationOption(
                    option_id="all-descendants",
                    label=f"전체 범위 ({total_count}건)",
                    value=concept_key,
                    matching_count=total_count,
                    concept_count=self.occupation_descendant_count(concept_key),
                    description="현재 검색어를 유지하고 모든 하위 직무를 포함",
                )
            )
            return ClarificationQuestion(
                question_id=question_id,
                field="occupation_concept_keys",
                question=f"{self.concept_label(concept_key)} 공고 {total_count}건 중 어떤 범위로 좁힐까요?",
                options=options,
                allow_custom=False,
                reason=(
                    "각 수치는 해당 필터를 적용했을 때의 결과 수이며, 복합 직무 공고는 "
                    "여러 선택지에 포함될 수 있습니다."
                ),
                candidate_count=total_count,
                concept_count=self.occupation_descendant_count(concept_key),
                facet_type="occupation",
            )
        return None

    def _occupation_alias_rows(self, connection: sqlite3.Connection) -> list[sqlite3.Row]:
        if self._occupation_alias_cache is None:
            self._occupation_alias_cache = connection.execute(
                """
                SELECT c.id, c.concept_key, c.source_key,
                       a.source_key AS alias_source_key,
                       a.alias, a.normalized_alias
                FROM search_concepts AS c
                JOIN search_aliases AS a ON a.concept_id = c.id
                WHERE c.concept_type = 'occupation'
                  AND c.status = 'active'
                  AND a.active = 1
                ORDER BY LENGTH(a.normalized_alias) DESC
                """
            ).fetchall()
        return self._occupation_alias_cache

    @staticmethod
    def _most_specific_matches(
        connection: sqlite3.Connection,
        concept_ids: set[int],
    ) -> set[int]:
        if len(concept_ids) < 2:
            return concept_ids
        broader_ids: set[int] = set()
        frontier = set(concept_ids)
        while frontier:
            placeholders = ",".join("?" for _ in frontier)
            rows = connection.execute(
                f"""
                SELECT source_concept_id, target_concept_id
                FROM search_concept_relations
                WHERE relation_type = 'broader'
                  AND source_concept_id IN ({placeholders})
                """,
                list(frontier),
            ).fetchall()
            next_frontier = {int(row["target_concept_id"]) for row in rows}
            unseen = next_frontier - broader_ids
            broader_ids.update(next_frontier)
            frontier = unseen
        return concept_ids - broader_ids

    def _record_term_candidate(
        self,
        connection: sqlite3.Connection,
        *,
        term: str,
        job_id: int,
    ) -> bool:
        normalized = normalize_term(term)
        if not normalized:
            return False
        now = _now()
        connection.execute(
            """
            INSERT INTO search_term_candidates (
                normalized_term, display_term, proposed_type, status,
                observation_count, first_seen_at, last_seen_at, sample_job_id,
                metadata_json
            ) VALUES (?, ?, 'skill', 'candidate', 1, ?, ?, ?, '{}')
            ON CONFLICT(normalized_term, proposed_type) DO UPDATE SET
                display_term = excluded.display_term,
                last_seen_at = excluded.last_seen_at,
                sample_job_id = COALESCE(search_term_candidates.sample_job_id, excluded.sample_job_id)
            """,
            (normalized, term.strip(), now, now, job_id),
        )
        candidate = connection.execute(
            """
            SELECT id FROM search_term_candidates
            WHERE normalized_term = ? AND proposed_type = 'skill'
            """,
            (normalized,),
        ).fetchone()
        if candidate is None:
            return False
        connection.execute(
            """
            INSERT OR IGNORE INTO search_term_candidate_observations (
                candidate_id, job_id, observed_at
            ) VALUES (?, ?, ?)
            """,
            (int(candidate["id"]), job_id, now),
        )
        connection.execute(
            """
            UPDATE search_term_candidates
            SET observation_count = (
                SELECT COUNT(*) FROM search_term_candidate_observations
                WHERE candidate_id = search_term_candidates.id
            )
            WHERE id = ?
            """,
            (int(candidate["id"]),),
        )
        status = connection.execute(
            "SELECT status FROM search_term_candidates WHERE id = ?",
            (int(candidate["id"]),),
        ).fetchone()
        return bool(status is not None and status["status"] == "candidate")

    def _skill_alias_rows(self, connection: sqlite3.Connection) -> list[sqlite3.Row]:
        if self._skill_alias_cache is None:
            self._skill_alias_cache = connection.execute(
                """
                SELECT c.id, c.source_key, c.preferred_label_ko,
                       c.preferred_label_en, a.normalized_alias
                FROM search_concepts AS c
                JOIN search_aliases AS a ON a.concept_id = c.id
                WHERE c.concept_type = 'skill'
                  AND c.status = 'active'
                  AND a.active = 1
                ORDER BY LENGTH(a.normalized_alias) DESC
                """
            ).fetchall()
        return self._skill_alias_cache

    @staticmethod
    def _json_text_list(value: Any) -> list[str]:
        try:
            parsed = json.loads(str(value or "[]"))
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()]

    @staticmethod
    def _preferred_skill_rows(rows: Iterable[sqlite3.Row]) -> list[sqlite3.Row]:
        candidates = list(rows)
        local = [
            row
            for row in candidates
            if str(row["source_key"]) in {CORE_SOURCE_KEY, CURATED_SOURCE_KEY}
        ]
        selected = local or candidates
        seen: set[int] = set()
        result: list[sqlite3.Row] = []
        for row in selected:
            concept_id = int(row["id"])
            if concept_id not in seen:
                seen.add(concept_id)
                result.append(row)
        return result

    def link_job(self, job_id: int) -> dict[str, int]:
        """구조화 필드와 검토된 별칭으로 공고를 직무·기술 개념에 연결한다."""

        connection = self._connect()
        counts = {
            "occupations": 0,
            "skills": 0,
            "candidate_observations": 0,
        }
        try:
            with connection:
                job = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if job is None:
                    return counts
                connection.execute(
                    "DELETE FROM job_concept_links WHERE job_id = ? AND linked_by IN ('exact_alias', 'contained_alias')",
                    (job_id,),
                )
                occupation_matches: dict[int, tuple[str, str, float]] = {}
                category = normalize_term(str(job["job_category"] or ""))
                category_parts = {
                    normalize_term(part)
                    for part in re.split(r"[/,|]", str(job["job_category"] or ""))
                    if normalize_term(part)
                }
                position = normalize_term(str(job["position"] or ""))
                for alias_row in self._occupation_alias_rows(connection):
                    concept_id = int(alias_row["id"])
                    alias = str(alias_row["normalized_alias"])
                    is_reviewed_local = str(alias_row["source_key"]) in {
                        CORE_SOURCE_KEY,
                        CURATED_SOURCE_KEY,
                    } or str(alias_row["alias_source_key"]) in {
                        CORE_SOURCE_KEY,
                        CURATED_SOURCE_KEY,
                    }
                    if category and category == alias:
                        occupation_matches[concept_id] = (
                            "job_category",
                            str(job["job_category"] or ""),
                            1.0,
                        )
                    elif alias in category_parts:
                        occupation_matches[concept_id] = (
                            "job_category",
                            str(job["job_category"] or ""),
                            0.96,
                        )
                    if position and (
                        position == alias
                        or (is_reviewed_local and _contains_alias(position, alias))
                    ):
                        current = occupation_matches.get(concept_id)
                        confidence = 0.98 if position == alias else 0.9
                        if current is None or confidence > current[2]:
                            occupation_matches[concept_id] = (
                                "position",
                                str(job["position"] or ""),
                                confidence,
                            )
                selected_occupations = self._most_specific_matches(
                    connection,
                    set(occupation_matches),
                )
                now = _now()
                for concept_id in selected_occupations:
                    evidence_field, evidence_text, confidence = occupation_matches[concept_id]
                    connection.execute(
                        """
                        INSERT INTO job_concept_links (
                            job_id, concept_id, link_type, evidence_field,
                            evidence_text, confidence, linked_by, created_at, updated_at
                        ) VALUES (?, ?, 'occupation', ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(job_id, concept_id, link_type, evidence_field) DO UPDATE SET
                            evidence_text = excluded.evidence_text,
                            confidence = excluded.confidence,
                            linked_by = excluded.linked_by,
                            updated_at = excluded.updated_at
                        """,
                        (
                            job_id,
                            concept_id,
                            evidence_field,
                            evidence_text,
                            confidence,
                            "exact_alias" if confidence >= 0.98 else "contained_alias",
                            now,
                            now,
                        ),
                    )
                    counts["occupations"] += 1

                sections = (
                    ("tech_stack", self._json_text_list(job["tech_stack"]), "mentioned", True),
                    ("requirements", self._json_text_list(job["requirements"]), "required", False),
                    ("preferred", self._json_text_list(job["preferred"]), "preferred", False),
                    ("main_tasks", self._json_text_list(job["main_tasks"]), "mentioned", False),
                )
                alias_rows = self._skill_alias_rows(connection)
                aliases: dict[str, list[sqlite3.Row]] = {}
                for alias_row in alias_rows:
                    aliases.setdefault(str(alias_row["normalized_alias"]), []).append(alias_row)
                inserted_links: set[tuple[int, str]] = set()
                for evidence_field, texts, requirement_type, exact_only in sections:
                    for evidence_text in texts:
                        normalized_text = normalize_term(evidence_text)
                        if exact_only:
                            matched_rows = self._preferred_skill_rows(
                                aliases.get(normalized_text, [])
                            )
                        else:
                            matched_rows = self._preferred_skill_rows(
                                row
                                for alias, rows in aliases.items()
                                if _contains_alias(normalized_text, alias)
                                for row in rows
                            )
                        if exact_only and not matched_rows:
                            recorded = self._record_term_candidate(
                                connection,
                                term=evidence_text,
                                job_id=job_id,
                            )
                            counts["candidate_observations"] += int(recorded)
                            continue
                        for alias_row in matched_rows:
                            concept_id = int(alias_row["id"])
                            link_key = (concept_id, evidence_field)
                            if link_key in inserted_links:
                                continue
                            inserted_links.add(link_key)
                            confidence = 1.0 if exact_only else 0.95
                            linked_by = "exact_alias" if exact_only else "contained_alias"
                            connection.execute(
                                """
                                INSERT INTO job_concept_links (
                                    job_id, concept_id, link_type, evidence_field,
                                    evidence_text, requirement_type, confidence,
                                    linked_by, created_at, updated_at
                                ) VALUES (?, ?, 'skill', ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(job_id, concept_id, link_type, evidence_field) DO UPDATE SET
                                    evidence_text = excluded.evidence_text,
                                    requirement_type = excluded.requirement_type,
                                    confidence = excluded.confidence,
                                    linked_by = excluded.linked_by,
                                    updated_at = excluded.updated_at
                                """,
                                (
                                    job_id,
                                    concept_id,
                                    evidence_field,
                                    evidence_text,
                                    requirement_type,
                                    confidence,
                                    linked_by,
                                    now,
                                    now,
                                ),
                            )
                            counts["skills"] += 1
        finally:
            connection.close()
        return counts

    def restore_original_tech_stacks(self) -> int:
        """과거 색인이 덮어쓴 기술 목록을 수집 당시 원본으로 복원한다."""

        connection = self._connect()
        restored = 0
        try:
            with connection:
                rows = connection.execute(
                    "SELECT id, tech_stack, raw_json FROM jobs"
                ).fetchall()
                for row in rows:
                    try:
                        payload = json.loads(str(row["raw_json"] or "{}"))
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    original = payload.get("tech_stack")
                    if not isinstance(original, list):
                        continue
                    encoded = json.dumps(original, ensure_ascii=False)
                    if encoded == str(row["tech_stack"] or "[]"):
                        continue
                    connection.execute(
                        "UPDATE jobs SET tech_stack = ? WHERE id = ?",
                        (encoded, int(row["id"])),
                    )
                    restored += 1
        finally:
            connection.close()
        return restored

    def relink_all_jobs(self) -> dict[str, int]:
        """기존 공고 전체를 현재 사전 버전으로 다시 연결한다."""

        self._occupation_alias_cache = None
        self._skill_alias_cache = None
        restored_tech_stacks = self.restore_original_tech_stacks()
        connection = self._connect()
        try:
            job_ids = [int(row[0]) for row in connection.execute("SELECT id FROM jobs").fetchall()]
        finally:
            connection.close()
        totals = {
            "jobs": len(job_ids),
            "occupations": 0,
            "skills": 0,
            "candidate_observations": 0,
            "resolved_candidates": 0,
            "restored_tech_stacks": restored_tech_stacks,
        }
        for job_id in job_ids:
            linked = self.link_job(job_id)
            for key in ("occupations", "skills", "candidate_observations"):
                totals[key] += int(linked[key])
        totals["resolved_candidates"] = self.reconcile_candidates()
        return totals

    def reconcile_candidates(self) -> int:
        """활성 사전의 단일 별칭과 일치하는 과거 후보를 자동 해소한다."""

        connection = self._connect()
        resolved = 0
        try:
            with connection:
                candidates = connection.execute(
                    """
                    SELECT id, normalized_term, proposed_type
                    FROM search_term_candidates
                    WHERE status = 'candidate'
                    """
                ).fetchall()
                for candidate in candidates:
                    matches = connection.execute(
                        """
                        SELECT DISTINCT concepts.concept_key, concepts.source_key
                        FROM search_aliases AS aliases
                        JOIN search_concepts AS concepts
                          ON concepts.id = aliases.concept_id
                        WHERE aliases.normalized_alias = ?
                          AND aliases.active = 1
                          AND concepts.status = 'active'
                          AND concepts.concept_type = ?
                        """,
                        (
                            str(candidate["normalized_term"]),
                            str(candidate["proposed_type"]),
                        ),
                    ).fetchall()
                    local_matches = [
                        row
                        for row in matches
                        if str(row["source_key"])
                        in {CORE_SOURCE_KEY, CURATED_SOURCE_KEY}
                    ]
                    selected = local_matches or matches
                    concept_keys = {
                        str(row["concept_key"])
                        for row in selected
                    }
                    if len(concept_keys) != 1:
                        continue
                    concept_key = next(iter(concept_keys))
                    connection.execute(
                        """
                        UPDATE search_term_candidates
                        SET status = 'accepted', reviewed_at = ?,
                            review_note = ?, accepted_concept_key = ?
                        WHERE id = ?
                        """,
                        (
                            _now(),
                            "활성 사전의 단일 별칭과 일치해 자동 해소",
                            concept_key,
                            int(candidate["id"]),
                        ),
                    )
                    resolved += 1
        finally:
            connection.close()
        return resolved


__all__ = [
    "DEFAULT_LOCAL_SEED",
    "LOCAL_SOURCE_KEY",
    "OCCUPATION_DOMAIN_ROOT_KEY",
    "SearchTaxonomyService",
]

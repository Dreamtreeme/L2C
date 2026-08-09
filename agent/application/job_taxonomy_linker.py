"""수집한 채용공고를 검토된 직무·기술 개념에 연결한다."""

from __future__ import annotations

import json
import re
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


class JobTaxonomyLinker:
    """공고 필드의 검토된 별칭만 DB 개념 링크로 저장한다."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._occupation_alias_cache: list[sqlite3.Row] | None = None
        self._skill_alias_cache: list[sqlite3.Row] | None = None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _occupation_alias_rows(
        self,
        connection: sqlite3.Connection,
    ) -> list[sqlite3.Row]:
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
            next_frontier = {
                int(row["target_concept_id"])
                for row in rows
            }
            unseen = next_frontier - broader_ids
            broader_ids.update(next_frontier)
            frontier = unseen
        return concept_ids - broader_ids

    @staticmethod
    def _record_term_candidate(
        connection: sqlite3.Connection,
        *,
        term: str,
        job_id: int,
    ) -> bool:
        normalized = normalize_term(term)
        if not normalized:
            return False
        now = taxonomy_timestamp()
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
                sample_job_id = COALESCE(
                    search_term_candidates.sample_job_id,
                    excluded.sample_job_id
                )
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
        candidate_id = int(candidate["id"])
        connection.execute(
            """
            INSERT OR IGNORE INTO search_term_candidate_observations (
                candidate_id, job_id, observed_at
            ) VALUES (?, ?, ?)
            """,
            (candidate_id, job_id, now),
        )
        connection.execute(
            """
            UPDATE search_term_candidates
            SET observation_count = (
                SELECT COUNT(*)
                FROM search_term_candidate_observations
                WHERE candidate_id = search_term_candidates.id
            )
            WHERE id = ?
            """,
            (candidate_id,),
        )
        status = connection.execute(
            "SELECT status FROM search_term_candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
        return bool(
            status is not None
            and status["status"] == "candidate"
        )

    def _skill_alias_rows(
        self,
        connection: sqlite3.Connection,
    ) -> list[sqlite3.Row]:
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
        return [
            str(item).strip()
            for item in parsed
            if str(item).strip()
        ]

    @staticmethod
    def _preferred_skill_rows(
        rows: Iterable[sqlite3.Row],
    ) -> list[sqlite3.Row]:
        candidates = list(rows)
        local = [
            row
            for row in candidates
            if str(row["source_key"])
            in {CORE_SOURCE_KEY, CURATED_SOURCE_KEY}
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

    @staticmethod
    def _clear_generated_links(
        connection: sqlite3.Connection,
        job_id: int,
    ) -> None:
        connection.execute(
            """
            DELETE FROM job_concept_links
            WHERE job_id = ?
              AND linked_by IN ('exact_alias', 'contained_alias')
            """,
            (job_id,),
        )

    def _match_occupations(
        self,
        connection: sqlite3.Connection,
        job: sqlite3.Row,
    ) -> dict[int, tuple[str, str, float]]:
        matches: dict[int, tuple[str, str, float]] = {}
        category_text = str(job["job_category"] or "")
        category = normalize_term(category_text)
        category_parts = {
            normalize_term(part)
            for part in re.split(r"[/,|]", category_text)
            if normalize_term(part)
        }
        position_text = str(job["position"] or "")
        position = normalize_term(position_text)

        for alias_row in self._occupation_alias_rows(connection):
            concept_id = int(alias_row["id"])
            alias = str(alias_row["normalized_alias"])
            is_reviewed_local = (
                str(alias_row["source_key"])
                in {CORE_SOURCE_KEY, CURATED_SOURCE_KEY}
                or str(alias_row["alias_source_key"])
                in {CORE_SOURCE_KEY, CURATED_SOURCE_KEY}
            )
            if category and category == alias:
                matches[concept_id] = (
                    "job_category",
                    category_text,
                    1.0,
                )
            elif alias in category_parts:
                matches[concept_id] = (
                    "job_category",
                    category_text,
                    0.96,
                )
            if position and (
                position == alias
                or (
                    is_reviewed_local
                    and contains_taxonomy_alias(position, alias)
                )
            ):
                current = matches.get(concept_id)
                confidence = 0.98 if position == alias else 0.9
                if current is None or confidence > current[2]:
                    matches[concept_id] = (
                        "position",
                        position_text,
                        confidence,
                    )
        return matches

    def _save_occupation_links(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: int,
        job: sqlite3.Row,
        linked_at: str,
    ) -> int:
        matches = self._match_occupations(connection, job)
        selected_ids = self._most_specific_matches(
            connection,
            set(matches),
        )
        for concept_id in selected_ids:
            evidence_field, evidence_text, confidence = matches[concept_id]
            connection.execute(
                """
                INSERT INTO job_concept_links (
                    job_id, concept_id, link_type, evidence_field,
                    evidence_text, confidence, linked_by,
                    created_at, updated_at
                ) VALUES (?, ?, 'occupation', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    job_id, concept_id, link_type, evidence_field
                ) DO UPDATE SET
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
                    (
                        "exact_alias"
                        if confidence >= 0.98
                        else "contained_alias"
                    ),
                    linked_at,
                    linked_at,
                ),
            )
        return len(selected_ids)

    @classmethod
    def _skill_sections(
        cls,
        job: sqlite3.Row,
    ) -> tuple[tuple[str, list[str], str, bool], ...]:
        return (
            (
                "tech_stack",
                cls._json_text_list(job["tech_stack"]),
                "mentioned",
                True,
            ),
            (
                "requirements",
                cls._json_text_list(job["requirements"]),
                "required",
                False,
            ),
            (
                "preferred",
                cls._json_text_list(job["preferred"]),
                "preferred",
                False,
            ),
            (
                "main_tasks",
                cls._json_text_list(job["main_tasks"]),
                "mentioned",
                False,
            ),
        )

    def _skill_alias_index(
        self,
        connection: sqlite3.Connection,
    ) -> dict[str, list[sqlite3.Row]]:
        aliases: dict[str, list[sqlite3.Row]] = {}
        for alias_row in self._skill_alias_rows(connection):
            aliases.setdefault(
                str(alias_row["normalized_alias"]),
                [],
            ).append(alias_row)
        return aliases

    def _matching_skill_rows(
        self,
        aliases: dict[str, list[sqlite3.Row]],
        evidence_text: str,
        *,
        exact_only: bool,
    ) -> list[sqlite3.Row]:
        normalized_text = normalize_term(evidence_text)
        if exact_only:
            return self._preferred_skill_rows(
                aliases.get(normalized_text, [])
            )
        return self._preferred_skill_rows(
            row
            for alias, rows in aliases.items()
            if contains_taxonomy_alias(normalized_text, alias)
            for row in rows
        )

    @staticmethod
    def _upsert_skill_link(
        connection: sqlite3.Connection,
        *,
        job_id: int,
        concept_id: int,
        evidence_field: str,
        evidence_text: str,
        requirement_type: str,
        exact_only: bool,
        linked_at: str,
    ) -> None:
        confidence = 1.0 if exact_only else 0.95
        linked_by = "exact_alias" if exact_only else "contained_alias"
        connection.execute(
            """
            INSERT INTO job_concept_links (
                job_id, concept_id, link_type,
                evidence_field, evidence_text,
                requirement_type, confidence, linked_by,
                created_at, updated_at
            ) VALUES (
                ?, ?, 'skill', ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(
                job_id, concept_id, link_type,
                evidence_field
            ) DO UPDATE SET
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
                linked_at,
                linked_at,
            ),
        )

    def _save_skill_links(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: int,
        job: sqlite3.Row,
        linked_at: str,
    ) -> tuple[int, int]:
        aliases = self._skill_alias_index(connection)
        inserted_links: set[tuple[int, str]] = set()
        skill_count = 0
        candidate_count = 0

        for evidence_field, texts, requirement_type, exact_only in (
            self._skill_sections(job)
        ):
            for evidence_text in texts:
                matched_rows = self._matching_skill_rows(
                    aliases,
                    evidence_text,
                    exact_only=exact_only,
                )
                if exact_only and not matched_rows:
                    candidate_count += int(
                        self._record_term_candidate(
                            connection,
                            term=evidence_text,
                            job_id=job_id,
                        )
                    )
                    continue
                for alias_row in matched_rows:
                    concept_id = int(alias_row["id"])
                    link_key = (concept_id, evidence_field)
                    if link_key in inserted_links:
                        continue
                    inserted_links.add(link_key)
                    self._upsert_skill_link(
                        connection,
                        job_id=job_id,
                        concept_id=concept_id,
                        evidence_field=evidence_field,
                        evidence_text=evidence_text,
                        requirement_type=requirement_type,
                        exact_only=exact_only,
                        linked_at=linked_at,
                    )
                    skill_count += 1
        return skill_count, candidate_count

    @staticmethod
    def _mark_indexed(
        connection: sqlite3.Connection,
        *,
        job_id: int,
        indexed_at: str,
    ) -> None:
        connection.execute(
            """
            UPDATE jobs
            SET taxonomy_index_status = 'indexed',
                taxonomy_index_error = NULL,
                taxonomy_index_attempts = taxonomy_index_attempts + 1,
                taxonomy_indexed_at = ?
            WHERE id = ?
            """,
            (indexed_at, job_id),
        )

    def _mark_index_failed(self, job_id: int, error: Exception) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    UPDATE jobs
                    SET taxonomy_index_status = 'failed',
                        taxonomy_index_error = ?,
                        taxonomy_index_attempts = taxonomy_index_attempts + 1,
                        taxonomy_indexed_at = NULL
                    WHERE id = ?
                    """,
                    (
                        f"{type(error).__name__}: {error}"[:1000],
                        job_id,
                    ),
                )
        finally:
            connection.close()

    def link_job(self, job_id: int) -> dict[str, int]:
        """구조화 필드와 검토된 별칭으로 공고를 직무·기술에 연결한다."""

        connection = self._connect()
        counts = {
            "occupations": 0,
            "skills": 0,
            "candidate_observations": 0,
        }
        try:
            with connection:
                job = connection.execute(
                    "SELECT * FROM jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
                if job is None:
                    return counts
                self._clear_generated_links(connection, job_id)
                indexed_at = taxonomy_timestamp()
                counts["occupations"] = self._save_occupation_links(
                    connection,
                    job_id=job_id,
                    job=job,
                    linked_at=indexed_at,
                )
                (
                    counts["skills"],
                    counts["candidate_observations"],
                ) = self._save_skill_links(
                    connection,
                    job_id=job_id,
                    job=job,
                    linked_at=indexed_at,
                )
                self._mark_indexed(
                    connection,
                    job_id=job_id,
                    indexed_at=indexed_at,
                )
        except Exception as exc:
            self._mark_index_failed(job_id, exc)
            raise
        finally:
            connection.close()
        return counts

    def relink_all_jobs(self) -> dict[str, int]:
        """기존 공고 전체를 현재 사전 버전으로 다시 연결한다."""

        self._occupation_alias_cache = None
        self._skill_alias_cache = None
        connection = self._connect()
        try:
            job_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM jobs"
                ).fetchall()
            ]
        finally:
            connection.close()
        totals = {
            "jobs": len(job_ids),
            "occupations": 0,
            "skills": 0,
            "candidate_observations": 0,
            "resolved_candidates": 0,
        }
        for job_id in job_ids:
            linked = self.link_job(job_id)
            for key in (
                "occupations",
                "skills",
                "candidate_observations",
            ):
                totals[key] += int(linked[key])
        totals["resolved_candidates"] = self.reconcile_candidates()
        return totals

    def relink_pending_jobs(
        self,
        *,
        limit: int = 100,
        max_attempts: int = 2,
    ) -> dict[str, int]:
        """아직 색인되지 않은 공고를 제한된 횟수 안에서 다시 연결한다."""

        connection = self._connect()
        try:
            job_ids = [
                int(row["id"])
                for row in connection.execute(
                    """
                    SELECT id
                    FROM jobs
                    WHERE taxonomy_index_status IN ('pending', 'failed')
                      AND taxonomy_index_attempts < ?
                    ORDER BY updated_at ASC, id ASC
                    LIMIT ?
                    """,
                    (max(1, int(max_attempts)), max(1, int(limit))),
                ).fetchall()
            ]
        finally:
            connection.close()

        totals = {
            "jobs": len(job_ids),
            "indexed": 0,
            "failed": 0,
            "occupations": 0,
            "skills": 0,
            "candidate_observations": 0,
        }
        for job_id in job_ids:
            try:
                linked = self.link_job(job_id)
            except Exception:
                totals["failed"] += 1
                continue
            totals["indexed"] += 1
            for key in (
                "occupations",
                "skills",
                "candidate_observations",
            ):
                totals[key] += int(linked[key])
        return totals

    def reconcile_candidates(self) -> int:
        """활성 사전의 단일 별칭과 일치하는 과거 후보를 해소한다."""

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
                        SELECT DISTINCT
                            concepts.concept_key,
                            concepts.source_key
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
                        SET status = 'accepted',
                            reviewed_at = ?,
                            review_note = ?,
                            accepted_concept_key = ?
                        WHERE id = ?
                        """,
                        (
                            taxonomy_timestamp(),
                            "활성 사전의 단일 별칭과 일치해 자동 해소",
                            concept_key,
                            int(candidate["id"]),
                        ),
                    )
                    resolved += 1
        finally:
            connection.close()
        return resolved


__all__ = ["JobTaxonomyLinker"]

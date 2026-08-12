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
                SELECT c.id, c.concept_key, a.alias, a.normalized_alias
                FROM search_concepts AS c
                JOIN search_aliases AS a ON a.concept_id = c.id
                WHERE c.concept_type = 'occupation'
                  AND c.status = 'active'
                  AND c.source_key = ?
                  AND a.source_key = ?
                  AND a.active = 1
                ORDER BY LENGTH(a.normalized_alias) DESC
                """,
                (CORE_SOURCE_KEY, CORE_SOURCE_KEY),
            ).fetchall()
        return self._occupation_alias_cache

    def _skill_alias_rows(
        self,
        connection: sqlite3.Connection,
    ) -> list[sqlite3.Row]:
        if self._skill_alias_cache is None:
            self._skill_alias_cache = connection.execute(
                """
                SELECT c.id, c.preferred_label_ko,
                       c.preferred_label_en, a.normalized_alias
                FROM search_concepts AS c
                JOIN search_aliases AS a ON a.concept_id = c.id
                WHERE c.concept_type = 'skill'
                  AND c.status = 'active'
                  AND c.source_key = ?
                  AND a.source_key = ?
                  AND a.active = 1
                ORDER BY LENGTH(a.normalized_alias) DESC
                """,
                (CORE_SOURCE_KEY, CORE_SOURCE_KEY),
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
    def _unique_skill_rows(
        rows: Iterable[sqlite3.Row],
    ) -> list[sqlite3.Row]:
        seen: set[int] = set()
        result: list[sqlite3.Row] = []
        for row in rows:
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
        main_task_texts = self._json_text_list(job["main_tasks"])

        for alias_row in self._occupation_alias_rows(connection):
            concept_id = int(alias_row["id"])
            alias = str(alias_row["normalized_alias"])
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
                or contains_taxonomy_alias(position, alias)
            ):
                current = matches.get(concept_id)
                confidence = 0.98 if position == alias else 0.9
                if current is None or confidence > current[2]:
                    matches[concept_id] = (
                        "position",
                        position_text,
                        confidence,
                    )
            for main_task_text in main_task_texts:
                if not contains_taxonomy_alias(
                    normalize_term(main_task_text),
                    alias,
                ):
                    continue
                current = matches.get(concept_id)
                if current is None or current[2] < 0.75:
                    matches[concept_id] = (
                        "main_tasks",
                        main_task_text,
                        0.75,
                    )
                break
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
        for concept_id in matches:
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
        return len(matches)

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
            return self._unique_skill_rows(
                aliases.get(normalized_text, [])
            )
        return self._unique_skill_rows(
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
    ) -> int:
        aliases = self._skill_alias_index(connection)
        inserted_links: set[tuple[int, str]] = set()
        skill_count = 0

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
        return skill_count

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
                counts["skills"] = self._save_skill_links(
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
        }
        for job_id in job_ids:
            linked = self.link_job(job_id)
            for key in ("occupations", "skills"):
                totals[key] += int(linked[key])
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
        }
        for job_id in job_ids:
            try:
                linked = self.link_job(job_id)
            except Exception:
                totals["failed"] += 1
                continue
            totals["indexed"] += 1
            for key in ("occupations", "skills"):
                totals[key] += int(linked[key])
        return totals


__all__ = ["JobTaxonomyLinker"]

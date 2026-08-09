"""수집 중 발견된 미등록 검색어를 사람이 검토해 사전에 반영한다."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from agent.application.search_taxonomy_import_service import normalize_term
from agent.application.job_taxonomy_linker import JobTaxonomyLinker
from agent.application.search_taxonomy_utils import CURATED_SOURCE_KEY
from shared.db.database import Database


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _language(value: str) -> str:
    return "ko" if any("가" <= char <= "힣" for char in value) else "en"


class SearchTaxonomyReviewService:
    """자동 승격 없이 검토된 용어만 활성 사전에 추가한다."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        Database(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _ensure_curated_source(connection: sqlite3.Connection) -> None:
        now = _now()
        connection.execute(
            """
            INSERT INTO taxonomy_sources (
                source_key, name, version, source_url, license,
                imported_at, metadata_json
            ) VALUES (?, ?, '1', ?, ?, ?, '{}')
            ON CONFLICT(source_key) DO UPDATE SET imported_at = excluded.imported_at
            """,
            (
                CURATED_SOURCE_KEY,
                "L2C 사용자 검토 사전",
                "local://search-taxonomy-review",
                "로컬 사용자 검토 데이터",
                now,
            ),
        )

    @staticmethod
    def _candidate(connection: sqlite3.Connection, candidate_id: int) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM search_term_candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"검색어 후보를 찾을 수 없습니다: {candidate_id}")
        return row

    @staticmethod
    def _observed_job_ids(
        connection: sqlite3.Connection,
        candidate_id: int,
    ) -> list[int]:
        return [
            int(row["job_id"])
            for row in connection.execute(
                """
                SELECT job_id
                FROM search_term_candidate_observations
                WHERE candidate_id = ?
                ORDER BY job_id
                """,
                (candidate_id,),
            ).fetchall()
        ]

    @staticmethod
    def _insert_alias(
        connection: sqlite3.Connection,
        *,
        concept_id: int,
        alias: str,
        alias_type: str = "exact",
    ) -> None:
        normalized = normalize_term(alias)
        if not normalized:
            return
        connection.execute(
            """
            INSERT INTO search_aliases (
                concept_id, alias, normalized_alias, language, alias_type,
                source_key, active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(concept_id, normalized_alias, language) DO UPDATE SET
                alias = excluded.alias,
                alias_type = excluded.alias_type,
                source_key = excluded.source_key,
                active = 1
            """,
            (
                concept_id,
                alias.strip(),
                normalized,
                _language(alias),
                alias_type,
                CURATED_SOURCE_KEY,
                _now(),
            ),
        )

    @staticmethod
    def _mark_accepted(
        connection: sqlite3.Connection,
        *,
        candidate_id: int,
        concept_key: str,
        note: str,
    ) -> None:
        connection.execute(
            """
            UPDATE search_term_candidates
            SET status = 'accepted', reviewed_at = ?, review_note = ?,
                accepted_concept_key = ?
            WHERE id = ?
            """,
            (_now(), note.strip() or None, concept_key, candidate_id),
        )

    def _relink_observations(self, job_ids: Iterable[int]) -> dict[str, int]:
        taxonomy = JobTaxonomyLinker(self.db_path)
        totals = {
            "jobs": 0,
            "occupations": 0,
            "skills": 0,
            "candidate_observations": 0,
        }
        for job_id in dict.fromkeys(int(item) for item in job_ids):
            linked = taxonomy.link_job(job_id)
            totals["jobs"] += 1
            for key in ("occupations", "skills", "candidate_observations"):
                totals[key] += int(linked[key])
        return totals

    def list_candidates(
        self,
        *,
        status: str = "candidate",
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """관찰 횟수가 많은 후보부터 검토 자료와 함께 반환한다."""

        if status not in {"candidate", "accepted", "rejected", "all"}:
            raise ValueError(f"지원하지 않는 후보 상태입니다: {status}")
        where = "" if status == "all" else "WHERE candidates.status = ?"
        params: list[object] = [] if status == "all" else [status]
        params.append(max(1, min(int(limit), 500)))
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT candidates.*, jobs.company_name, jobs.position
                FROM search_term_candidates AS candidates
                LEFT JOIN jobs ON jobs.id = candidates.sample_job_id
                {where}
                ORDER BY candidates.observation_count DESC,
                         candidates.last_seen_at DESC,
                         candidates.id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def add_reviewed_alias(
        self,
        concept_key: str,
        alias: str,
        *,
        note: str = "",
    ) -> dict[str, object]:
        """사용자가 확인한 직무 표현을 활성 직무 개념의 별칭으로 승격한다."""

        normalized = normalize_term(alias)
        if not normalized:
            raise ValueError("승격할 직무 표현이 필요합니다.")
        connection = self._connect()
        try:
            with connection:
                concept = connection.execute(
                    """
                    SELECT id, concept_key, concept_type
                    FROM search_concepts
                    WHERE concept_key = ? AND status = 'active'
                    """,
                    (concept_key,),
                ).fetchone()
                if concept is None:
                    raise ValueError(f"활성 개념을 찾을 수 없습니다: {concept_key}")
                if str(concept["concept_type"]) != "occupation":
                    raise ValueError(
                        "사용자 확인 별칭은 직무 개념에만 추가할 수 있습니다."
                    )
                self._ensure_curated_source(connection)
                self._insert_alias(
                    connection,
                    concept_id=int(concept["id"]),
                    alias=alias,
                )
                now = _now()
                connection.execute(
                    """
                    INSERT INTO search_term_candidates (
                        normalized_term, display_term, proposed_type, status,
                        observation_count, first_seen_at, last_seen_at,
                        sample_job_id, metadata_json, reviewed_at,
                        review_note, accepted_concept_key
                    ) VALUES (?, ?, 'occupation', 'accepted', 1, ?, ?, NULL,
                              '{}', ?, ?, ?)
                    ON CONFLICT(normalized_term, proposed_type) DO UPDATE SET
                        display_term = excluded.display_term,
                        status = 'accepted',
                        last_seen_at = excluded.last_seen_at,
                        reviewed_at = excluded.reviewed_at,
                        review_note = excluded.review_note,
                        accepted_concept_key = excluded.accepted_concept_key
                    """,
                    (
                        normalized,
                        alias.strip(),
                        now,
                        now,
                        now,
                        note.strip() or "사용자 의미 확인으로 승인",
                        concept_key,
                    ),
                )
        finally:
            connection.close()
        return {
            "status": "accepted",
            "concept_key": concept_key,
            "alias": alias.strip(),
            "relinked": JobTaxonomyLinker(self.db_path).relink_all_jobs(),
        }

    def accept_as_alias(
        self,
        candidate_id: int,
        concept_key: str,
        *,
        note: str = "",
    ) -> dict[str, object]:
        """후보 표현을 이미 존재하는 개념의 검토된 별칭으로 추가한다."""

        connection = self._connect()
        try:
            with connection:
                candidate = self._candidate(connection, candidate_id)
                if str(candidate["status"]) == "accepted":
                    raise ValueError("이미 승인된 후보는 다시 매핑할 수 없습니다.")
                concept = connection.execute(
                    """
                    SELECT id, concept_key, concept_type
                    FROM search_concepts
                    WHERE concept_key = ? AND status = 'active'
                    """,
                    (concept_key,),
                ).fetchone()
                if concept is None:
                    raise ValueError(f"활성 개념을 찾을 수 없습니다: {concept_key}")
                if str(candidate["proposed_type"]) != str(concept["concept_type"]):
                    raise ValueError("후보 유형과 대상 개념 유형이 다릅니다.")
                self._ensure_curated_source(connection)
                self._insert_alias(
                    connection,
                    concept_id=int(concept["id"]),
                    alias=str(candidate["display_term"]),
                )
                self._mark_accepted(
                    connection,
                    candidate_id=candidate_id,
                    concept_key=str(concept["concept_key"]),
                    note=note,
                )
                job_ids = self._observed_job_ids(connection, candidate_id)
        finally:
            connection.close()
        return {
            "candidate_id": candidate_id,
            "status": "accepted",
            "concept_key": concept_key,
            "relinked": self._relink_observations(job_ids),
        }

    def accept_as_new_concept(
        self,
        candidate_id: int,
        canonical_label: str,
        *,
        aliases: Iterable[str] = (),
        broader_concept_key: str = "",
        note: str = "",
    ) -> dict[str, object]:
        """검토자가 확정한 대표명으로 로컬 개념을 새로 만든다."""

        normalized_label = normalize_term(canonical_label)
        if not normalized_label:
            raise ValueError("새 개념의 대표명이 필요합니다.")
        connection = self._connect()
        try:
            with connection:
                candidate = self._candidate(connection, candidate_id)
                if str(candidate["status"]) == "accepted":
                    raise ValueError("이미 승인된 후보는 다시 매핑할 수 없습니다.")
                concept_type = str(candidate["proposed_type"])
                self._ensure_curated_source(connection)
                digest = hashlib.sha1(
                    f"{concept_type}:{normalized_label}".encode("utf-8")
                ).hexdigest()[:16]
                concept_key = f"l2c:curated:{concept_type}:{digest}"
                now = _now()
                ko_label = (
                    canonical_label.strip()
                    if _language(canonical_label) == "ko"
                    else None
                )
                en_label = canonical_label.strip() if ko_label is None else None
                connection.execute(
                    """
                    INSERT INTO search_concepts (
                        concept_key, concept_type, preferred_label_ko,
                        preferred_label_en, definition, status, source_key,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, NULL, 'active', ?, ?, ?)
                    ON CONFLICT(concept_key) DO UPDATE SET
                        preferred_label_ko = excluded.preferred_label_ko,
                        preferred_label_en = excluded.preferred_label_en,
                        status = 'active',
                        updated_at = excluded.updated_at
                    """,
                    (
                        concept_key,
                        concept_type,
                        ko_label,
                        en_label,
                        CURATED_SOURCE_KEY,
                        now,
                        now,
                    ),
                )
                concept = connection.execute(
                    "SELECT id FROM search_concepts WHERE concept_key = ?",
                    (concept_key,),
                ).fetchone()
                if concept is None:
                    raise RuntimeError("새 검색 개념 저장에 실패했습니다.")
                concept_id = int(concept["id"])
                self._insert_alias(
                    connection,
                    concept_id=concept_id,
                    alias=canonical_label,
                    alias_type="preferred",
                )
                self._insert_alias(
                    connection,
                    concept_id=concept_id,
                    alias=str(candidate["display_term"]),
                )
                for alias in aliases:
                    self._insert_alias(
                        connection,
                        concept_id=concept_id,
                        alias=str(alias),
                    )
                if broader_concept_key:
                    broader = connection.execute(
                        """
                        SELECT id, concept_type
                        FROM search_concepts
                        WHERE concept_key = ? AND status = 'active'
                        """,
                        (broader_concept_key,),
                    ).fetchone()
                    if broader is None:
                        raise ValueError(
                            f"상위 개념을 찾을 수 없습니다: {broader_concept_key}"
                        )
                    if (
                        concept_type != "occupation"
                        or broader["concept_type"] != "occupation"
                    ):
                        raise ValueError(
                            "상하위 관계는 직무 개념 사이에서만 지정할 수 있습니다."
                        )
                    connection.execute(
                        """
                        INSERT INTO search_concept_relations (
                            source_concept_id, target_concept_id, relation_type,
                            source_key, metadata_json, created_at
                        ) VALUES (?, ?, 'broader', ?, '{}', ?)
                        ON CONFLICT DO NOTHING
                        """,
                        (concept_id, int(broader["id"]), CURATED_SOURCE_KEY, now),
                    )
                self._mark_accepted(
                    connection,
                    candidate_id=candidate_id,
                    concept_key=concept_key,
                    note=note,
                )
                job_ids = self._observed_job_ids(connection, candidate_id)
        finally:
            connection.close()
        return {
            "candidate_id": candidate_id,
            "status": "accepted",
            "concept_key": concept_key,
            "relinked": self._relink_observations(job_ids),
        }

    def reject(self, candidate_id: int, *, note: str = "") -> dict[str, object]:
        """사전 검색 단위가 아닌 후보를 거절하고 재관찰만 누적한다."""

        connection = self._connect()
        try:
            with connection:
                candidate = self._candidate(connection, candidate_id)
                if str(candidate["status"]) == "accepted":
                    raise ValueError("승인된 후보는 거절 상태로 바꿀 수 없습니다.")
                connection.execute(
                    """
                    UPDATE search_term_candidates
                    SET status = 'rejected', reviewed_at = ?, review_note = ?,
                        accepted_concept_key = NULL
                    WHERE id = ?
                    """,
                    (_now(), note.strip() or None, candidate_id),
                )
        finally:
            connection.close()
        return {"candidate_id": candidate_id, "status": "rejected"}


__all__ = ["SearchTaxonomyReviewService"]

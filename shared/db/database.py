"""
SQLite 로컬 DB — raw SQL.
URL 기준으로 중복을 막고 추출 이력을 보관합니다.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from shared.db.reflex_schema import (
    REFLEX_MEMORY_SCHEMA,
    ensure_feedback_episode_schema,
    ensure_recipe_candidate_queue_schema,
)
from shared.db.search_taxonomy_schema import SEARCH_TAXONOMY_SCHEMA

logger = logging.getLogger(__name__)


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT    NOT NULL UNIQUE,
    company_name    TEXT,
    position        TEXT,
    job_category    TEXT,
    experience_level TEXT,
    education       TEXT,
    employment_type TEXT,
    location        TEXT,
    posted_at       TEXT,
    posted_at_text  TEXT,
    deadline        TEXT,
    salary          TEXT,
    tech_stack      TEXT,
    main_tasks      TEXT,
    requirements    TEXT,
    preferred       TEXT,
    benefits        TEXT,
    raw_json        TEXT,
    screenshot_path TEXT,
    ocr_text_path   TEXT,
    source_platform TEXT,
    raw_ocr_text    TEXT,
    content_hash    TEXT UNIQUE,
    evidence_hash   TEXT,
    experience_min  INTEGER,
    experience_max  INTEGER,
    experience_text TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_name);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

CREATE TABLE IF NOT EXISTS job_versions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id              INTEGER NOT NULL,
    version_number      INTEGER NOT NULL,
    observed_at         TEXT NOT NULL,
    source_url          TEXT NOT NULL,
    source_platform     TEXT,
    evidence_hash       TEXT NOT NULL,
    changed_fields_json TEXT NOT NULL,
    content_json        TEXT NOT NULL,
    UNIQUE(job_id, version_number),
    UNIQUE(job_id, evidence_hash),
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_job_versions_job ON job_versions(job_id, version_number DESC);
CREATE INDEX IF NOT EXISTS idx_job_versions_observed ON job_versions(observed_at);

CREATE TABLE IF NOT EXISTS recipes (
    recipe_key    TEXT PRIMARY KEY,
    site          TEXT NOT NULL,
    goal          TEXT,
    path_json     TEXT NOT NULL,
    metadata_json TEXT,
    success_count INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recipes_site ON recipes(site);

CREATE TABLE IF NOT EXISTS recipe_sources (
    recipe_key   TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (recipe_key, candidate_id),
    FOREIGN KEY (recipe_key) REFERENCES recipes(recipe_key) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_recipe_sources_candidate
ON recipe_sources(candidate_id);

{REFLEX_MEMORY_SCHEMA}
{SEARCH_TAXONOMY_SCHEMA}
"""


class Database:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Database init: {self.db_path}")
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            ensure_feedback_episode_schema(conn)
            ensure_recipe_candidate_queue_schema(conn)
            
            # 마이그레이션 지원: 기존 테이블에 신규 컬럼이 없을 경우 동적 추가
            cursor = conn.execute("PRAGMA table_info(jobs)")
            columns = [row["name"] for row in cursor.fetchall()]
            
            new_cols = {
                "source_platform": "TEXT",
                "raw_ocr_text": "TEXT",
                "content_hash": "TEXT",
                "experience_min": "INTEGER",
                "experience_max": "INTEGER",
                "experience_text": "TEXT",
                "posted_at": "TEXT",
                "posted_at_text": "TEXT",
                "evidence_hash": "TEXT",
            }
            for col, col_type in new_cols.items():
                if col not in columns:
                    try:
                        conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {col_type}")
                        logger.info(f"마이그레이션: jobs 테이블에 컬럼 '{col}' 추가 완료")
                    except sqlite3.OperationalError as e:
                        logger.debug(f"컬럼 '{col}' 추가 건너뜀: {e}")
            
            link_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(job_concept_links)").fetchall()
            }
            if "requirement_type" not in link_columns:
                conn.execute(
                    "ALTER TABLE job_concept_links "
                    "ADD COLUMN requirement_type TEXT NOT NULL DEFAULT 'mentioned'"
                )
            if "minimum_months" not in link_columns:
                conn.execute(
                    "ALTER TABLE job_concept_links ADD COLUMN minimum_months INTEGER"
                )

            candidate_columns = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(search_term_candidates)"
                ).fetchall()
            }
            for column in ("reviewed_at", "review_note", "accepted_concept_key"):
                if column not in candidate_columns:
                    conn.execute(
                        f"ALTER TABLE search_term_candidates ADD COLUMN {column} TEXT"
                    )

            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_content_hash ON jobs(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_posted_at ON jobs(posted_at)")
            
        logger.debug("schema 확인/생성 및 마이그레이션 완료")

    def exists(self, url: str) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT 1 FROM jobs WHERE url = ? LIMIT 1", (url,)).fetchone()
        result = row is not None
        logger.debug(f"exists({url}) = {result}")
        return result

    def upsert(
        self,
        url: str,
        data: dict[str, Any],
        screenshot_path: str | None = None,
        ocr_text_path: str | None = None,
    ) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        from shared.integrity import source_evidence_hash

        evidence_hash = source_evidence_hash(url, data)
        canonical_data = dict(data)
        canonical_data["evidence_hash"] = evidence_hash
        payload = {
            "url": url,
            "company_name": data.get("company_name"),
            "position": data.get("position"),
            "job_category": data.get("job_category"),
            "experience_level": data.get("experience_level"),
            "education": data.get("education"),
            "employment_type": data.get("employment_type"),
            "location": data.get("location"),
            "posted_at": data.get("posted_at"),
            "posted_at_text": data.get("posted_at_text"),
            "deadline": data.get("deadline"),
            "salary": data.get("salary"),
            "tech_stack": json.dumps(data.get("tech_stack") or [], ensure_ascii=False),
            "main_tasks": json.dumps(data.get("main_tasks") or [], ensure_ascii=False),
            "requirements": json.dumps(data.get("requirements") or [], ensure_ascii=False),
            "preferred": json.dumps(data.get("preferred") or [], ensure_ascii=False),
            "benefits": json.dumps(data.get("benefits") or [], ensure_ascii=False),
            "source_platform": data.get("source_platform"),
            "raw_ocr_text": data.get("raw_ocr_text"),
            "content_hash": data.get("content_hash"),
            "evidence_hash": evidence_hash,
            "experience_min": data.get("experience_min"),
            "experience_max": data.get("experience_max"),
            "experience_text": data.get("experience_text"),
            "raw_json": json.dumps(canonical_data, ensure_ascii=False),
            "screenshot_path": screenshot_path,
            "ocr_text_path": ocr_text_path,
            "updated_at": now,
        }

        with self._conn() as conn:
            existing = None
            if payload.get("content_hash"):
                existing = conn.execute(
                    "SELECT id, raw_json, evidence_hash FROM jobs WHERE url = ? OR content_hash = ?",
                    (url, payload["content_hash"]),
                ).fetchone()
            else:
                existing = conn.execute(
                    "SELECT id, raw_json, evidence_hash FROM jobs WHERE url = ?", (url,)
                ).fetchone()

            if existing:
                cols = ", ".join(f"{k} = :{k}" for k in payload.keys())
                conn.execute(f"UPDATE jobs SET {cols} WHERE id = {existing['id']}", payload)
                if existing["evidence_hash"] != evidence_hash:
                    self._insert_job_version(
                        conn,
                        job_id=int(existing["id"]),
                        observed_at=now,
                        url=url,
                        data=canonical_data,
                        evidence_hash=evidence_hash,
                        previous_raw_json=existing["raw_json"],
                    )
                logger.info(f"DB UPDATE id={existing['id']} url={url} (hash={payload.get('content_hash')})")
                return existing["id"]

            payload["created_at"] = now
            cols = ", ".join(payload.keys())
            placeholders = ", ".join(f":{k}" for k in payload.keys())
            cur = conn.execute(
                f"INSERT INTO jobs ({cols}) VALUES ({placeholders})",
                payload,
            )
            new_id = cur.lastrowid
            self._insert_job_version(
                conn,
                job_id=int(new_id),
                observed_at=now,
                url=url,
                data=canonical_data,
                evidence_hash=evidence_hash,
                previous_raw_json=None,
            )
            logger.info(f"DB INSERT id={new_id} url={url} company={data.get('company_name')!r}")
            return new_id

    @staticmethod
    def _changed_fields(previous_raw_json: str | None, data: dict[str, Any]) -> list[str]:
        if not previous_raw_json:
            return sorted(str(key) for key in data.keys())
        try:
            previous = json.loads(previous_raw_json)
        except (TypeError, json.JSONDecodeError):
            previous = {}
        return sorted(
            str(key)
            for key in set(previous) | set(data)
            if previous.get(key) != data.get(key)
        )

    def _insert_job_version(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: int,
        observed_at: str,
        url: str,
        data: dict[str, Any],
        evidence_hash: str,
        previous_raw_json: str | None,
    ) -> None:
        """새로운 출처 증거일 때만 변경 스냅샷을 추가한다."""

        row = conn.execute(
            "SELECT COALESCE(MAX(version_number), 0) + 1 AS next_version "
            "FROM job_versions WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        version_number = int(row["next_version"] if row else 1)
        conn.execute(
            "INSERT OR IGNORE INTO job_versions ("
            "job_id, version_number, observed_at, source_url, source_platform, "
            "evidence_hash, changed_fields_json, content_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                version_number,
                observed_at,
                url,
                data.get("source_platform"),
                evidence_hash,
                json.dumps(self._changed_fields(previous_raw_json, data), ensure_ascii=False),
                json.dumps(data, ensure_ascii=False),
            ),
        )

    def list_versions(self, job_id: int) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, job_id, version_number, observed_at, source_url, "
                "source_platform, evidence_hash, changed_fields_json, content_json "
                "FROM job_versions WHERE job_id = ? ORDER BY version_number DESC",
                (job_id,),
            ).fetchall()
        versions: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for field in ("changed_fields_json", "content_json"):
                try:
                    item[field.removesuffix("_json")] = json.loads(item.pop(field))
                except (TypeError, json.JSONDecodeError):
                    item[field.removesuffix("_json")] = item.pop(field)
            versions.append(item)
        return versions

    def _fetch_one(self, where_clause: str, param: Any) -> dict | None:
        """WHERE 절 하나로 단일 행을 조회하는 공통 헬퍼."""
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT * FROM jobs WHERE {where_clause}", (param,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get(self, job_id: int) -> dict | None:
        result = self._fetch_one("id = ?", job_id)
        logger.debug(f"get(id={job_id}) found={result is not None}")
        return result

    def get_by_url(self, url: str) -> dict | None:
        result = self._fetch_one("url = ?", url)
        logger.debug(f"get_by_url({url}) found={result is not None}")
        return result

    def list_recent(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, url, company_name, position, posted_at, posted_at_text, created_at "
                "FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        logger.debug(f"list_recent(limit={limit}) → {len(rows)}건")
        return [dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        for k in ("tech_stack", "main_tasks", "requirements", "preferred", "benefits"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except json.JSONDecodeError:
                    pass
        if d.get("raw_json"):
            try:
                d["raw_json"] = json.loads(d["raw_json"])
            except json.JSONDecodeError:
                pass
        return d

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

from shared.db.search_taxonomy_schema import SEARCH_TAXONOMY_SCHEMA
from shared.schema.jd_schema import JobCollectionEvidence, JobPosting, StoredJob

logger = logging.getLogger(__name__)


JOBS_COLUMNS_SQL = """(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT    NOT NULL UNIQUE,
    company_name    TEXT,
    position        TEXT,
    job_category    TEXT,
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
    content_hash    TEXT,
    evidence_hash   TEXT,
    taxonomy_index_status TEXT NOT NULL DEFAULT 'pending',
    taxonomy_index_error TEXT,
    taxonomy_index_attempts INTEGER NOT NULL DEFAULT 0,
    taxonomy_indexed_at TEXT,
    experience_min  INTEGER,
    experience_max  INTEGER,
    experience_text TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
)"""


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS jobs {JOBS_COLUMNS_SQL};

CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_name);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

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
        logger.debug("schema 확인/생성 완료")

    def exists(self, url: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM jobs WHERE url = ? LIMIT 1", (url,)
            ).fetchone()
        result = row is not None
        logger.debug(f"exists({url}) = {result}")
        return result

    def upsert(
        self,
        posting: JobPosting,
        *,
        evidence: JobCollectionEvidence | None = None,
    ) -> int:
        url = str(posting.url or "").strip()
        if not url:
            raise ValueError("JobPosting.url은 SQLite 저장에 필요합니다.")
        data = posting.model_dump(mode="json")
        evidence = evidence or JobCollectionEvidence()
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
            "education": data.get("education"),
            "employment_type": data.get("employment_type"),
            "location": data.get("location"),
            "posted_at": data.get("posted_at"),
            "posted_at_text": data.get("posted_at_text"),
            "deadline": data.get("deadline"),
            "salary": data.get("salary"),
            "tech_stack": json.dumps(data.get("tech_stack") or [], ensure_ascii=False),
            "main_tasks": json.dumps(data.get("main_tasks") or [], ensure_ascii=False),
            "requirements": json.dumps(
                data.get("requirements") or [], ensure_ascii=False
            ),
            "preferred": json.dumps(data.get("preferred") or [], ensure_ascii=False),
            "benefits": json.dumps(data.get("benefits") or [], ensure_ascii=False),
            "source_platform": data.get("source_platform"),
            "raw_ocr_text": data.get("raw_ocr_text"),
            "content_hash": data.get("content_hash"),
            "evidence_hash": evidence_hash,
            "taxonomy_index_status": "pending",
            "taxonomy_index_error": None,
            "taxonomy_index_attempts": 0,
            "taxonomy_indexed_at": None,
            "experience_min": data.get("experience_min"),
            "experience_max": data.get("experience_max"),
            "experience_text": data.get("experience_text"),
            "raw_json": json.dumps(canonical_data, ensure_ascii=False),
            "screenshot_path": evidence.screenshot_path or None,
            "ocr_text_path": evidence.ocr_text_path or None,
            "updated_at": now,
        }

        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM jobs WHERE url = ?",
                (url,),
            ).fetchone()

            if existing:
                cols = ", ".join(f"{k} = :{k}" for k in payload)
                conn.execute(
                    f"UPDATE jobs SET {cols} WHERE id = {existing['id']}", payload
                )
                logger.info(
                    f"DB UPDATE id={existing['id']} url={url} (hash={payload.get('content_hash')})"
                )
                return existing["id"]

            payload["created_at"] = now
            cols = ", ".join(payload.keys())
            placeholders = ", ".join(f":{k}" for k in payload)
            cur = conn.execute(
                f"INSERT INTO jobs ({cols}) VALUES ({placeholders})",
                payload,
            )
            new_id = cur.lastrowid
            logger.info(
                f"DB INSERT id={new_id} url={url} company={data.get('company_name')!r}"
            )
            return new_id

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

    def load_jobs(self, job_ids: list[int]) -> list[StoredJob]:
        """SQLite 표현을 정규 공고 타입으로 복원해 반환한다."""

        ids = list(
            dict.fromkeys(int(job_id) for job_id in job_ids if int(job_id) > 0)
        )
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        documents_by_id: dict[int, StoredJob] = {}
        for row in rows:
            decoded = self._row_to_dict(row)
            document = StoredJob.model_validate(
                {
                    "id": decoded["id"],
                    **{
                        field: decoded.get(field)
                        for field in JobPosting.model_fields
                    },
                }
            )
            documents_by_id[int(document.id)] = document
        return [
            documents_by_id[job_id]
            for job_id in ids
            if job_id in documents_by_id
        ]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        for k in ("tech_stack", "main_tasks", "requirements", "preferred", "benefits"):
            if d.get(k):
                d[k] = json.loads(d[k])
        if d.get("raw_json"):
            d["raw_json"] = json.loads(d["raw_json"])
        return d

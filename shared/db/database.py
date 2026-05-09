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

logger = logging.getLogger(__name__)


SCHEMA = """
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
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_name);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
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
        payload = {
            "url": url,
            "company_name": data.get("company_name"),
            "position": data.get("position"),
            "job_category": data.get("job_category"),
            "experience_level": data.get("experience_level"),
            "education": data.get("education"),
            "employment_type": data.get("employment_type"),
            "location": data.get("location"),
            "deadline": data.get("deadline"),
            "salary": data.get("salary"),
            "tech_stack": json.dumps(data.get("tech_stack") or [], ensure_ascii=False),
            "main_tasks": json.dumps(data.get("main_tasks") or [], ensure_ascii=False),
            "requirements": json.dumps(data.get("requirements") or [], ensure_ascii=False),
            "preferred": json.dumps(data.get("preferred") or [], ensure_ascii=False),
            "benefits": json.dumps(data.get("benefits") or [], ensure_ascii=False),
            "raw_json": json.dumps(data, ensure_ascii=False),
            "screenshot_path": screenshot_path,
            "ocr_text_path": ocr_text_path,
            "updated_at": now,
        }

        with self._conn() as conn:
            existing = conn.execute("SELECT id FROM jobs WHERE url = ?", (url,)).fetchone()
            if existing:
                cols = ", ".join(f"{k} = :{k}" for k in payload.keys())
                conn.execute(f"UPDATE jobs SET {cols} WHERE url = :url", payload)
                logger.info(f"DB UPDATE id={existing['id']} url={url}")
                return existing["id"]

            payload["created_at"] = now
            cols = ", ".join(payload.keys())
            placeholders = ", ".join(f":{k}" for k in payload.keys())
            cur = conn.execute(
                f"INSERT INTO jobs ({cols}) VALUES ({placeholders})",
                payload,
            )
            new_id = cur.lastrowid
            logger.info(f"DB INSERT id={new_id} url={url} company={data.get('company_name')!r}")
            return new_id

    def get(self, job_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        logger.debug(f"get(id={job_id}) found={row is not None}")
        return self._row_to_dict(row) if row else None

    def get_by_url(self, url: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE url = ?", (url,)).fetchone()
        logger.debug(f"get_by_url({url}) found={row is not None}")
        return self._row_to_dict(row) if row else None

    def list_recent(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, url, company_name, position, created_at "
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

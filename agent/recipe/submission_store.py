"""SQLite store for commander-reviewed worker submissions."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from shared.db.reflex_schema import WORKER_SUBMISSIONS_INDEX_SQL, WORKER_SUBMISSIONS_TABLE_SQL


class SubmissionStore:
    def __init__(self, db_path=None):
        if db_path is None:
            from shared.config import DB_PATH
            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(WORKER_SUBMISSIONS_TABLE_SQL)
            for sql in WORKER_SUBMISSIONS_INDEX_SQL:
                conn.execute(sql)

    @staticmethod
    def _dump(payload: dict[str, Any] | None) -> str:
        return json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)

    def commit_submission(
        self,
        submission: dict[str, Any],
        review: dict[str, Any] | None = None,
        source: str = "vision_worker",
    ) -> str:
        review = review or {}
        run_id = submission.get("run_id") or "run-unknown"
        attempt = int(submission.get("review_attempt") or 0)
        submission_id = f"{run_id}:{attempt}"
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT created_at FROM worker_submissions WHERE submission_id=?",
                (submission_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute(
                """
                INSERT OR REPLACE INTO worker_submissions (
                    submission_id, run_id, source, site, goal, keyword, run_status,
                    review_attempt, review_decision, review_confidence, feedback_to_worker,
                    payload_json, review_json, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    submission_id,
                    run_id,
                    source,
                    submission.get("site", "") or "",
                    submission.get("goal", "") or "",
                    submission.get("keyword", "") or "",
                    submission.get("run_status", "") or "",
                    attempt,
                    review.get("decision", "") or "",
                    float(review.get("confidence") or 0.0),
                    review.get("feedback_to_worker", "") or "",
                    self._dump(submission),
                    self._dump(review),
                    created_at,
                    now,
                ),
            )
        return submission_id

    def list_recent(self, limit: int = 20, decision: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM worker_submissions"
        params: list[Any] = []
        if decision:
            sql += " WHERE review_decision=?"
            params.append(decision)
        sql += " ORDER BY updated_at DESC, submission_id DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            for source_key, target_key in (("payload_json", "payload"), ("review_json", "review")):
                try:
                    item[target_key] = json.loads(item.pop(source_key) or "{}")
                except json.JSONDecodeError:
                    item[target_key] = {}
            out.append(item)
        return out
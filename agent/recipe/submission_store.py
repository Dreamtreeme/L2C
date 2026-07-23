"""SQLite store for commander-reviewed worker submissions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.recipe.payload_sanitizer import strip_state_debug_fields
from agent.recipe.sqlite_store import SQLiteStore
from shared.db.reflex_schema import WORKER_SUBMISSIONS_INDEX_SQL, WORKER_SUBMISSIONS_TABLE_SQL


class SubmissionStore(SQLiteStore):
    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(WORKER_SUBMISSIONS_TABLE_SQL)
            for sql in WORKER_SUBMISSIONS_INDEX_SQL:
                conn.execute(sql)

    def commit_submission(
        self,
        submission: dict[str, Any],
        review: dict[str, Any] | None = None,
        source: str = "vision_worker",
    ) -> str:
        review = review or {}
        clean_submission = strip_state_debug_fields(submission)
        run_id = clean_submission.get("run_id") or "run-unknown"
        attempt = int(clean_submission.get("review_attempt") or 0)
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
                    clean_submission.get("site", "") or "",
                    clean_submission.get("goal", "") or "",
                    clean_submission.get("keyword", "") or "",
                    clean_submission.get("run_status", "") or "",
                    attempt,
                    review.get("decision", "") or "",
                    float(review.get("confidence") or 0.0),
                    review.get("feedback_to_worker", "") or "",
                    self.dump_json(clean_submission),
                    self.dump_json(review),
                    created_at,
                    now,
                ),
            )
        return submission_id

    def _row_to_item(self, row: Any) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = strip_state_debug_fields(
            self.load_json(item.pop("payload_json", ""), {})
        )
        item["review"] = self.load_json(item.pop("review_json", ""), {})
        return item

    def get_submission(self, submission_id: str) -> dict[str, Any] | None:
        """제출물 식별자로 단일 작업자 제출물을 조회한다."""

        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM worker_submissions WHERE submission_id=?",
                (submission_id,),
            ).fetchone()
        return self._row_to_item(row) if row else None

    def get_run_attempt(
        self,
        run_id: str,
        review_attempt: int | None = None,
    ) -> dict[str, Any] | None:
        """실행 ID와 검토 시도로 제출물을 조회한다.

        검토 시도를 생략하면 같은 실행에서 가장 최근 제출물을 반환한다.
        """

        sql = "SELECT * FROM worker_submissions WHERE run_id=?"
        params: list[Any] = [run_id]
        if review_attempt is not None:
            sql += " AND review_attempt=?"
            params.append(review_attempt)
        sql += " ORDER BY review_attempt DESC, updated_at DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return self._row_to_item(row) if row else None

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
        return [self._row_to_item(row) for row in rows]

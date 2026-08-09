"""작업자 제출물을 참조하는 Reflex 후보 검토 큐."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from agent.recipe.payload_sanitizer import strip_full_screen_signatures
from agent.recipe.sqlite_store import SQLiteStore
from shared.db.reflex_schema import (
    RECIPE_CANDIDATES_INDEX_SQL,
    RECIPE_CANDIDATES_QUEUE_INDEX_SQL,
    RECIPE_CANDIDATES_TABLE_SQL,
)
from shared.schema.feedback_schema import RecipeCandidate, WorkerSubmission


_CANDIDATE_SELECT = """
SELECT c.*, s.run_id, s.source, s.payload_json
FROM recipe_candidates AS c
JOIN worker_submissions AS s ON s.submission_id = c.submission_id
"""


class RecipeCandidateStore(SQLiteStore):
    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(RECIPE_CANDIDATES_TABLE_SQL)
            for sql in (
                *RECIPE_CANDIDATES_INDEX_SQL,
                *RECIPE_CANDIDATES_QUEUE_INDEX_SQL,
            ):
                conn.execute(sql)

    def commit_candidate(
        self,
        submission: WorkerSubmission,
        *,
        submission_id: str,
        status: str = "pending_replay",
    ) -> str:
        if not submission_id or not submission.recorded_steps:
            return ""
        candidate_id = submission_id
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT created_at FROM recipe_candidates WHERE candidate_id=?",
                (candidate_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute(
                """
                INSERT OR REPLACE INTO recipe_candidates (
                    candidate_id, submission_id, status, validation_json,
                    review_attempts, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (candidate_id, submission_id, status, "", 0, created_at, now),
            )
        return candidate_id

    def _load(self, conn, candidate_id: str):
        return conn.execute(
            _CANDIDATE_SELECT + " WHERE c.candidate_id=?",
            (candidate_id,),
        ).fetchone()

    def get_candidate(self, candidate_id: str) -> RecipeCandidate | None:
        with self._conn() as conn:
            row = self._load(conn, candidate_id)
        return self._row_to_item(row) if row else None

    def update_status(
        self,
        candidate_id: str,
        status: str,
        validation: dict[str, Any] | None = None,
    ) -> bool:
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            result = conn.execute(
                """
                UPDATE recipe_candidates
                SET status=?, validation_json=?, review_started_at=NULL,
                    next_review_at=NULL, review_error='', updated_at=?
                WHERE candidate_id=?
                """,
                (status, self.dump_json(validation or {}), now, candidate_id),
            )
            return result.rowcount > 0

    def enqueue_review(self, candidate_id: str) -> bool:
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            result = conn.execute(
                """
                UPDATE recipe_candidates
                SET status='pending_review', review_started_at=NULL,
                    next_review_at=NULL, review_error='', updated_at=?
                WHERE candidate_id=? AND status='pending_replay'
                """,
                (now, candidate_id),
            )
            return result.rowcount > 0

    def recover_interrupted_reviews(self) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            result = conn.execute(
                """
                UPDATE recipe_candidates
                SET status='pending_review', review_started_at=NULL,
                    next_review_at=NULL,
                    review_error='review_worker_interrupted', updated_at=?
                WHERE status='reviewing'
                """,
                (now,),
            )
            return result.rowcount

    def claim_review(self, candidate_id: str | None = None) -> RecipeCandidate | None:
        now = datetime.now().isoformat(timespec="seconds")
        candidate_filter = "AND candidate_id=?" if candidate_id else ""
        params: tuple[Any, ...] = (now, candidate_id) if candidate_id else (now,)
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"""
                SELECT candidate_id FROM recipe_candidates
                WHERE status='pending_review'
                  AND (next_review_at IS NULL OR next_review_at <= ?)
                  {candidate_filter}
                ORDER BY created_at, candidate_id LIMIT 1
                """,
                params,
            ).fetchone()
            if row is None:
                return None
            resolved_id = str(row["candidate_id"])
            claimed = conn.execute(
                """
                UPDATE recipe_candidates
                SET status='reviewing', review_started_at=?,
                    review_attempts=review_attempts + 1,
                    review_error='', updated_at=?
                WHERE candidate_id=? AND status='pending_review'
                """,
                (now, now, resolved_id),
            )
            claimed_row = (
                self._load(conn, resolved_id) if claimed.rowcount == 1 else None
            )
        return self._row_to_item(claimed_row) if claimed_row else None

    def defer_review(
        self,
        candidate_id: str,
        error: str,
        *,
        retry_delay_sec: float,
        terminal: bool = False,
    ) -> bool:
        now_value = datetime.now()
        next_review_at = (
            None
            if terminal
            else (now_value + timedelta(seconds=max(0.0, retry_delay_sec))).isoformat(
                timespec="seconds"
            )
        )
        now = now_value.isoformat(timespec="seconds")
        with self._conn() as conn:
            result = conn.execute(
                """
                UPDATE recipe_candidates
                SET status=?, review_started_at=NULL, next_review_at=?,
                    review_error=?, updated_at=?
                WHERE candidate_id=? AND status='reviewing'
                """,
                (
                    "review_failed" if terminal else "pending_review",
                    next_review_at,
                    str(error or "")[:1000],
                    now,
                    candidate_id,
                ),
            )
            return result.rowcount > 0

    def list_recent(
        self,
        limit: int = 20,
        status: str | None = None,
    ) -> list[RecipeCandidate]:
        where = " WHERE c.status=?" if status else ""
        params: tuple[Any, ...] = (status, limit) if status else (limit,)
        with self._conn() as conn:
            rows = conn.execute(
                _CANDIDATE_SELECT
                + where
                + " ORDER BY c.updated_at DESC, c.candidate_id DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def _row_to_item(self, row) -> RecipeCandidate:
        item = dict(row)
        submission = WorkerSubmission.model_validate(
            strip_full_screen_signatures(
                self.load_json(item.pop("payload_json", ""), {})
            )
        )
        item["validation"] = self.load_json(item.pop("validation_json", ""), {})
        item["review_error"] = str(item.get("review_error") or "")
        item["submission"] = submission
        return RecipeCandidate.model_validate(item)

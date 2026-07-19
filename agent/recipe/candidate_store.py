"""SQLite store for commander-reviewed Reflex recipe candidates."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from agent.recipe.payload_sanitizer import strip_state_debug_fields
from agent.recipe.sqlite_store import SQLiteStore
from shared.db.reflex_schema import (
    RECIPE_CANDIDATES_INDEX_SQL,
    RECIPE_CANDIDATES_TABLE_SQL,
    ensure_recipe_candidate_queue_schema,
)


class RecipeCandidateStore(SQLiteStore):
    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(RECIPE_CANDIDATES_TABLE_SQL)
            for sql in RECIPE_CANDIDATES_INDEX_SQL:
                conn.execute(sql)
            ensure_recipe_candidate_queue_schema(conn)

    def commit_candidate(
        self,
        submission: dict[str, Any],
        review: dict[str, Any] | None = None,
        source: str = "vision_worker",
        submission_id: str = "",
        status: str = "pending_replay",
    ) -> str:
        review = review or {}
        if review.get("decision") != "accept" or not review.get("recipe_candidate"):
            return ""
        clean_submission = strip_state_debug_fields(submission)
        steps = [step for step in clean_submission.get("recorded_steps", []) or [] if isinstance(step, dict)]
        if not steps:
            return ""

        run_id = clean_submission.get("run_id") or "run-unknown"
        attempt = int(clean_submission.get("review_attempt") or 0)
        candidate_id = submission_id or f"{run_id}:{attempt}"
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
                    candidate_id, run_id, submission_id, source, site, goal, keyword,
                    status, review_confidence, steps_json, payload_json, review_json,
                    validation_json, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    candidate_id,
                    run_id,
                    submission_id or candidate_id,
                    source,
                    clean_submission.get("site", "") or "",
                    clean_submission.get("goal", "") or "",
                    clean_submission.get("keyword", "") or "",
                    status,
                    float(review.get("confidence") or 0.0),
                    self.dump_json(steps),
                    self.dump_json(clean_submission),
                    self.dump_json(review),
                    "",
                    created_at,
                    now,
                ),
            )
        return candidate_id

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM recipe_candidates WHERE candidate_id=?",
                (candidate_id,),
            ).fetchone()
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
        """아직 검토되지 않은 후보를 영속 승격 대기열에 넣는다."""

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
        """이전 프로세스가 처리 중이던 후보를 다시 대기 상태로 돌린다."""

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

    def claim_next_review(self) -> dict[str, Any] | None:
        """가장 오래 대기한 후보 하나를 원자적으로 선점한다."""

        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT candidate_id
                FROM recipe_candidates
                WHERE status='pending_review'
                  AND (next_review_at IS NULL OR next_review_at <= ?)
                ORDER BY created_at ASC, candidate_id ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            candidate_id = str(row["candidate_id"])
            claimed = conn.execute(
                """
                UPDATE recipe_candidates
                SET status='reviewing', review_started_at=?,
                    review_attempts=review_attempts + 1,
                    review_error='', updated_at=?
                WHERE candidate_id=? AND status='pending_review'
                """,
                (now, now, candidate_id),
            )
            if claimed.rowcount != 1:
                return None
            claimed_row = conn.execute(
                "SELECT * FROM recipe_candidates WHERE candidate_id=?",
                (candidate_id,),
            ).fetchone()
        return self._row_to_item(claimed_row) if claimed_row else None

    def defer_review(
        self,
        candidate_id: str,
        error: str,
        *,
        retry_delay_sec: float,
        terminal: bool = False,
    ) -> bool:
        """일시 오류는 재시도하고, 한도를 넘긴 오류는 별도 실패 상태로 남긴다."""

        now_value = datetime.now()
        next_review_at = (
            None
            if terminal
            else (now_value + timedelta(seconds=max(0.0, retry_delay_sec))).isoformat(
                timespec="seconds"
            )
        )
        now = now_value.isoformat(timespec="seconds")
        status = "review_failed" if terminal else "pending_review"
        with self._conn() as conn:
            result = conn.execute(
                """
                UPDATE recipe_candidates
                SET status=?, review_started_at=NULL, next_review_at=?,
                    review_error=?, updated_at=?
                WHERE candidate_id=? AND status='reviewing'
                """,
                (status, next_review_at, str(error or "")[:1000], now, candidate_id),
            )
            return result.rowcount > 0

    def list_recent(self, limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM recipe_candidates"
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY updated_at DESC, candidate_id DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_item(row) for row in rows]

    def _row_to_item(self, row) -> dict[str, Any]:
        item = dict(row)
        item["steps"] = strip_state_debug_fields(self.load_json(item.pop("steps_json", ""), []))
        item["payload"] = strip_state_debug_fields(self.load_json(item.pop("payload_json", ""), {}))
        item["review"] = self.load_json(item.pop("review_json", ""), {})
        item["validation"] = self.load_json(item.pop("validation_json", ""), {})
        return item

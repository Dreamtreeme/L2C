"""SQLite store for validated worker submissions."""

from __future__ import annotations

from datetime import datetime
from agent.recipe.sqlite_store import SQLiteStore
from shared.db.reflex_schema import (
    WORKER_SUBMISSIONS_TABLE_SQL,
)
from shared.schema.feedback_schema import WorkerSubmission


class SubmissionStore(SQLiteStore):
    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(WORKER_SUBMISSIONS_TABLE_SQL)

    def commit_submission(
        self,
        submission: WorkerSubmission,
        source: str = "vision_worker",
    ) -> str:
        payload = submission.model_dump(mode="json")
        run_id = submission.run_id or "run-unknown"
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT created_at FROM worker_submissions WHERE run_id=?",
                (run_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute(
                """
                INSERT OR REPLACE INTO worker_submissions (
                    run_id, source, payload_json,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?)
                """,
                (
                    run_id,
                    source,
                    self.dump_json(payload),
                    created_at,
                    now,
                ),
            )
        return run_id

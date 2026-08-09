"""SQLite store for validated worker submissions."""

from __future__ import annotations

from datetime import datetime
from agent.recipe.payload_sanitizer import strip_full_screen_signatures
from agent.recipe.sqlite_store import SQLiteStore
from shared.db.reflex_schema import (
    WORKER_SUBMISSIONS_INDEX_SQL,
    WORKER_SUBMISSIONS_TABLE_SQL,
)
from shared.schema.feedback_schema import StoredWorkerSubmission, WorkerSubmission


class SubmissionStore(SQLiteStore):
    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(WORKER_SUBMISSIONS_TABLE_SQL)
            for sql in WORKER_SUBMISSIONS_INDEX_SQL:
                conn.execute(sql)

    def commit_submission(
        self,
        submission: WorkerSubmission,
        source: str = "vision_worker",
    ) -> str:
        clean_submission = strip_full_screen_signatures(
            submission.model_dump(mode="json")
        )
        run_id = submission.run_id or "run-unknown"
        submission_id = run_id
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
                    submission_id, run_id, source, payload_json,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?)
                """,
                (
                    submission_id,
                    run_id,
                    source,
                    self.dump_json(clean_submission),
                    created_at,
                    now,
                ),
            )
        return submission_id

    def find_submission(
        self,
        *,
        submission_id: str = "",
        run_id: str = "",
    ) -> StoredWorkerSubmission | None:
        if submission_id:
            where, params = "submission_id=?", (submission_id,)
        elif run_id:
            where, params = "run_id=?", (run_id,)
        else:
            where, params = "1=1", ()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT submission_id, run_id, source, payload_json "
                f"FROM worker_submissions WHERE {where} "
                "ORDER BY updated_at DESC LIMIT 1",
                params,
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = WorkerSubmission.model_validate(
            strip_full_screen_signatures(self.load_json(item.pop("payload_json"), {}))
        )
        return StoredWorkerSubmission.model_validate(item)

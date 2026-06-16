"""SQLite store for commander-reviewed Reflex recipe candidates."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.recipe.sqlite_store import SQLiteStore
from shared.db.reflex_schema import RECIPE_CANDIDATES_INDEX_SQL, RECIPE_CANDIDATES_TABLE_SQL


class RecipeCandidateStore(SQLiteStore):
    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(RECIPE_CANDIDATES_TABLE_SQL)
            for sql in RECIPE_CANDIDATES_INDEX_SQL:
                conn.execute(sql)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(recipe_candidates)").fetchall()}
            if "validation_json" not in columns:
                conn.execute("ALTER TABLE recipe_candidates ADD COLUMN validation_json TEXT")

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
        steps = [step for step in submission.get("recorded_steps", []) or [] if isinstance(step, dict)]
        if not steps:
            return ""

        run_id = submission.get("run_id") or "run-unknown"
        attempt = int(submission.get("review_attempt") or 0)
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
                    submission.get("site", "") or "",
                    submission.get("goal", "") or "",
                    submission.get("keyword", "") or "",
                    status,
                    float(review.get("confidence") or 0.0),
                    self.dump_json(steps),
                    self.dump_json(submission),
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
                SET status=?, validation_json=?, updated_at=?
                WHERE candidate_id=?
                """,
                (status, self.dump_json(validation or {}), now, candidate_id),
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
        item["steps"] = self.load_json(item.pop("steps_json", ""), [])
        item["payload"] = self.load_json(item.pop("payload_json", ""), {})
        item["review"] = self.load_json(item.pop("review_json", ""), {})
        item["validation"] = self.load_json(item.pop("validation_json", ""), {})
        return item

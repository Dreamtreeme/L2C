"""SQLite-backed memory for feedback episodes.

Episodes are persisted after a vision run so later Critic/Memory phases can
promote repeated successful patterns without adding latency to every action.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from agent.recipe.payload_sanitizer import strip_full_screen_signatures
from agent.recipe.sqlite_store import SQLiteStore
from shared.db.reflex_schema import (
    FEEDBACK_EPISODES_INDEX_SQL,
    FEEDBACK_EPISODES_TABLE_SQL,
)


class FeedbackStore(SQLiteStore):
    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(FEEDBACK_EPISODES_TABLE_SQL)
            for sql in FEEDBACK_EPISODES_INDEX_SQL:
                conn.execute(sql)

    def commit_episodes(
        self,
        episodes: list[dict[str, Any]],
        run_id: str | None = None,
        run_status: str = "",
        source: str = "vision_run",
    ) -> int:
        clean = [strip_full_screen_signatures(episode) for episode in episodes or [] if isinstance(episode, dict)]
        if not clean:
            return 0
        run_id = run_id or f"run-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat(timespec="seconds")
        rows = []
        for idx, episode in enumerate(clean):
            proposal = episode.get("proposal") if isinstance(episode.get("proposal"), dict) else {}
            feedback = episode.get("feedback") if isinstance(episode.get("feedback"), dict) else {}
            try:
                action_sequence = max(0, int(episode.get("seq", idx)))
            except (TypeError, ValueError):
                action_sequence = idx
            episode_id = f"{run_id}:action:{action_sequence:04d}"
            rows.append(
                (
                    episode_id,
                    run_id,
                    run_status,
                    source,
                    episode.get("site", "") or "",
                    episode.get("goal", "") or "",
                    proposal.get("action", "") or "",
                    feedback.get("label", "") or "",
                    feedback.get("reason", "") or "",
                    float(feedback.get("confidence") or 0.0),
                    self.dump_json(episode),
                    now,
                )
            )
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO feedback_episodes (
                    episode_id, run_id, run_status, source, site, goal,
                    action, feedback_label, feedback_reason, feedback_confidence,
                    payload_json, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            return conn.total_changes

    def list_recent(self, limit: int = 20, feedback_label: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM feedback_episodes"
        params: list[Any] = []
        if feedback_label:
            sql += " WHERE feedback_label=?"
            params.append(feedback_label)
        sql += " ORDER BY created_at DESC, episode_id DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["payload"] = strip_full_screen_signatures(self.load_json(item.pop("payload_json", ""), {}))
            out.append(item)
        return out

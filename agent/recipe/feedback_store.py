"""SQLite-backed memory for feedback episodes.

Episodes are persisted after a vision run so later Critic/Memory phases can
promote repeated successful patterns without adding latency to every action.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


class FeedbackStore:
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_episodes (
                    episode_id         TEXT PRIMARY KEY,
                    run_id             TEXT NOT NULL,
                    run_status         TEXT,
                    source             TEXT,
                    site               TEXT,
                    goal               TEXT,
                    page_state_key     TEXT,
                    action             TEXT,
                    feedback_label     TEXT,
                    feedback_reason    TEXT,
                    feedback_confidence REAL,
                    payload_json       TEXT NOT NULL,
                    created_at         TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_run ON feedback_episodes(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_site_label ON feedback_episodes(site, feedback_label)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_action ON feedback_episodes(action)")

    @staticmethod
    def _dump_episode(episode: dict[str, Any]) -> str:
        return json.dumps(episode, ensure_ascii=False, sort_keys=True)

    def commit_episodes(
        self,
        episodes: list[dict[str, Any]],
        run_id: str | None = None,
        run_status: str = "",
        source: str = "vision_run",
    ) -> int:
        clean = [episode for episode in episodes or [] if isinstance(episode, dict)]
        if not clean:
            return 0
        run_id = run_id or f"run-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat(timespec="seconds")
        rows = []
        for idx, episode in enumerate(clean):
            proposal = episode.get("proposal") if isinstance(episode.get("proposal"), dict) else {}
            feedback = episode.get("feedback") if isinstance(episode.get("feedback"), dict) else {}
            episode_id = f"{run_id}:{episode.get('seq', idx)}"
            rows.append(
                (
                    episode_id,
                    run_id,
                    run_status,
                    source,
                    episode.get("site", "") or "",
                    episode.get("goal", "") or "",
                    episode.get("page_state_key", "") or "",
                    proposal.get("action", "") or "",
                    feedback.get("label", "") or "",
                    feedback.get("reason", "") or "",
                    float(feedback.get("confidence") or 0.0),
                    self._dump_episode(episode),
                    now,
                )
            )
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO feedback_episodes (
                    episode_id, run_id, run_status, source, site, goal, page_state_key,
                    action, feedback_label, feedback_reason, feedback_confidence,
                    payload_json, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            try:
                item["payload"] = json.loads(item.pop("payload_json") or "{}")
            except json.JSONDecodeError:
                item["payload"] = {}
            out.append(item)
        return out
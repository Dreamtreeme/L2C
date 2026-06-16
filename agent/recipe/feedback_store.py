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

from shared.db.reflex_schema import FEEDBACK_EPISODES_INDEX_SQL, FEEDBACK_EPISODES_TABLE_SQL


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
            conn.execute(FEEDBACK_EPISODES_TABLE_SQL)
            for sql in FEEDBACK_EPISODES_INDEX_SQL:
                conn.execute(sql)

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
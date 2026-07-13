"""진행 중인 조사 상태를 로컬 SQLite에 저장한다."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from shared.schema.investigation_schema import InvestigationRequest


INVESTIGATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS investigation_sessions (
    investigation_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    status TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_investigation_conversation
ON investigation_sessions(conversation_id, updated_at DESC);
"""


class InvestigationStore:
    """조사 상태의 저장과 재개만 담당하는 저장소."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(INVESTIGATION_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, investigation: InvestigationRequest) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = investigation.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO investigation_sessions (
                    investigation_id, conversation_id, status, state_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(investigation_id) DO UPDATE SET
                    conversation_id = excluded.conversation_id,
                    status = excluded.status,
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (
                    investigation.investigation_id,
                    investigation.conversation_id,
                    investigation.status.value,
                    payload,
                    now,
                    now,
                ),
            )

    def get(self, investigation_id: str) -> InvestigationRequest | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM investigation_sessions WHERE investigation_id = ?",
                (str(investigation_id),),
            ).fetchone()
        if row is None:
            return None
        return InvestigationRequest.model_validate(json.loads(row["state_json"]))

    def latest_for_conversation(self, conversation_id: str) -> InvestigationRequest | None:
        if not conversation_id:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT state_json
                FROM investigation_sessions
                WHERE conversation_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (str(conversation_id),),
            ).fetchone()
        if row is None:
            return None
        return InvestigationRequest.model_validate(json.loads(row["state_json"]))


__all__ = ["InvestigationStore"]

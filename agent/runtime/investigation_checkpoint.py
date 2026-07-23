"""조사 LangGraph의 로컬 체크포인트 연결을 관리한다."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


def build_investigation_checkpoint_path(db_path: str | Path) -> Path:
    """업무 DB와 분리된 조사 체크포인트 경로를 만든다."""

    resolved_db_path = Path(db_path)
    suffix = resolved_db_path.suffix or ".db"
    return resolved_db_path.with_name(
        f"{resolved_db_path.stem}.investigation_checkpoints{suffix}"
    )


class InvestigationCheckpointRuntime:
    """한 프로세스에서 재사용할 SQLite 체크포인터와 연결을 소유한다."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        checkpoint_path: str | Path | None = None,
    ):
        self.checkpoint_path = Path(
            checkpoint_path or build_investigation_checkpoint_path(db_path)
        )
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = sqlite3.connect(
            self.checkpoint_path,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self.saver = SqliteSaver(
            self._connection,
            serde=JsonPlusSerializer(allowed_msgpack_modules=[]),
        )
        self.saver.setup()

    def close(self) -> None:
        """체크포인트 연결을 한 번만 닫는다."""

        if self._connection is None:
            return
        self._connection.close()
        self._connection = None


__all__ = [
    "InvestigationCheckpointRuntime",
    "build_investigation_checkpoint_path",
]

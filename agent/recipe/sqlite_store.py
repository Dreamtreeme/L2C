"""Shared SQLite helpers for recipe memory stores."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class SQLiteStore:
    def __init__(self, db_path=None):
        if db_path is None:
            from shared.config import DB_PATH

            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        raise NotImplementedError

    @staticmethod
    def dump_json(payload: Any) -> str:
        return json.dumps(payload if payload is not None else {}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def load_json(value: str | None, default: Any = None) -> Any:
        if default is None:
            default = {}
        try:
            return json.loads(value or "")
        except (TypeError, json.JSONDecodeError):
            return default

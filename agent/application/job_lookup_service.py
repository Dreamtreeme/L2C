"""수집 에이전트가 사용할 채용공고 읽기 전용 조회 서비스."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.utils.logger import logger


def find_job_id_by_url(url: str, *, db_path: str | Path | None = None) -> int | None:
    """끝 슬래시 차이만 허용해 동일한 저장 URL의 공고 ID를 찾는다."""

    normalized = str(url or "").strip().rstrip("/")
    if not normalized:
        return None
    if db_path is None:
        from shared.config import DB_PATH

        db_path = DB_PATH
    resolved = Path(db_path)
    if not resolved.exists():
        return None

    try:
        connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT id FROM jobs WHERE url = ? OR url = ? LIMIT 1",
                (normalized, f"{normalized}/"),
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        logger.warning("Job URL lookup failed", url=normalized, error=str(exc))
        return None
    return int(row[0]) if row else None


__all__ = ["find_job_id_by_url"]

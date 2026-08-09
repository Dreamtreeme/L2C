"""로컬 실행 산출물과 감사 이력의 보존 정책을 계산하고 적용한다."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from agent.config import get_settings
from shared.db.database import Database


@dataclass(frozen=True)
class RetentionPolicy:
    log_days: int = 30
    artifact_days: int = 90
    job_version_days: int = 180
    keep_job_versions: int = 5

    @classmethod
    def from_env(cls) -> "RetentionPolicy":
        settings = get_settings().retention
        return cls(
            log_days=settings.log_days,
            artifact_days=settings.artifact_days,
            job_version_days=settings.job_version_days,
            keep_job_versions=settings.keep_job_versions,
        )


def _expired(timestamp: str, cutoff: datetime) -> bool:
    try:
        observed = datetime.fromisoformat(str(timestamp))
    except (TypeError, ValueError):
        return False
    if observed.tzinfo is not None:
        observed = observed.replace(tzinfo=None)
    return observed < cutoff


def _file_candidates(root: Path, cutoff: datetime, referenced: set[Path]) -> list[Path]:
    if not root.exists():
        return []
    resolved_root = root.resolve()
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_root) or resolved in referenced:
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        if modified < cutoff:
            candidates.append(path)
    return candidates


def _referenced_job_artifacts(conn: sqlite3.Connection) -> set[Path]:
    referenced: set[Path] = set()
    for row in conn.execute(
        "SELECT screenshot_path, ocr_text_path FROM jobs "
        "WHERE screenshot_path IS NOT NULL OR ocr_text_path IS NOT NULL"
    ):
        for value in row:
            if value:
                referenced.add(Path(str(value)).resolve())
    return referenced


def _job_version_candidates(
    conn: sqlite3.Connection,
    policy: RetentionPolicy,
    now: datetime,
) -> list[int]:
    version_cutoff = now - timedelta(days=policy.job_version_days)
    candidates: list[int] = []

    version_rows = conn.execute(
        "SELECT id, job_id, version_number, observed_at FROM job_versions "
        "ORDER BY job_id, version_number DESC"
    ).fetchall()
    per_job_count: dict[int, int] = {}
    for row in version_rows:
        job_id = int(row["job_id"])
        per_job_count[job_id] = per_job_count.get(job_id, 0) + 1
        if per_job_count[job_id] > policy.keep_job_versions and _expired(
            row["observed_at"], version_cutoff
        ):
            candidates.append(int(row["id"]))
    return candidates


def run_retention(
    *,
    db_path: str | Path,
    logs_dir: str | Path,
    screenshot_dir: str | Path,
    policy: RetentionPolicy | None = None,
    dry_run: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """보존 만료 후보를 보고하고 명시적으로 요청된 경우에만 삭제한다."""

    policy = policy or RetentionPolicy.from_env()
    current = now or datetime.now()
    Database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        referenced = _referenced_job_artifacts(conn)
        version_candidates = _job_version_candidates(conn, policy, current)
        log_files = _file_candidates(
            Path(logs_dir),
            current - timedelta(days=policy.log_days),
            set(),
        )
        artifact_files = _file_candidates(
            Path(screenshot_dir),
            current - timedelta(days=policy.artifact_days),
            referenced,
        )
        files = log_files + artifact_files
        reclaimable_bytes = sum(path.stat().st_size for path in files if path.exists())
        if not dry_run:
            if version_candidates:
                placeholders = ",".join("?" for _ in version_candidates)
                conn.execute(
                    f"DELETE FROM job_versions WHERE id IN ({placeholders})",
                    version_candidates,
                )
            for path in files:
                path.unlink(missing_ok=True)
            conn.commit()
        inventory = {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "jobs",
                "job_versions",
            )
        }
        return {
            "dry_run": dry_run,
            "policy": asdict(policy),
            "files": {
                "log_count": len(log_files),
                "artifact_count": len(artifact_files),
                "reclaimable_bytes": reclaimable_bytes,
            },
            "database": {"job_versions": len(version_candidates)},
            "inventory": inventory,
        }
    finally:
        conn.close()


__all__ = ["RetentionPolicy", "run_retention"]

"""로컬 실행 산출물과 감사 이력의 보존 정책을 계산하고 적용한다."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetentionPolicy:
    log_days: int = 30
    artifact_days: int = 14
    audit_days: int = 90
    job_version_days: int = 180
    keep_job_versions: int = 5

    @classmethod
    def from_env(cls) -> "RetentionPolicy":
        def value(name: str, default: int) -> int:
            try:
                return max(1, int(os.getenv(name, str(default))))
            except ValueError:
                return default

        return cls(
            log_days=value("RETENTION_LOG_DAYS", 30),
            artifact_days=value("RETENTION_ARTIFACT_DAYS", 14),
            audit_days=value("RETENTION_AUDIT_DAYS", 90),
            job_version_days=value("RETENTION_JOB_VERSION_DAYS", 180),
            keep_job_versions=value("RETENTION_KEEP_JOB_VERSIONS", 5),
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


def _database_cleanup_candidates(
    conn: sqlite3.Connection,
    policy: RetentionPolicy,
    now: datetime,
) -> dict[str, list[Any]]:
    audit_cutoff = now - timedelta(days=policy.audit_days)
    version_cutoff = now - timedelta(days=policy.job_version_days)
    candidates: dict[str, list[Any]] = {
        "job_versions": [],
        "feedback_episodes": [],
        "worker_submissions": [],
        "recipe_candidates": [],
    }

    version_rows = conn.execute(
        "SELECT id, job_id, version_number, observed_at FROM job_versions "
        "ORDER BY job_id, version_number DESC"
    ).fetchall()
    per_job_count: dict[int, int] = {}
    for row in version_rows:
        job_id = int(row["job_id"])
        per_job_count[job_id] = per_job_count.get(job_id, 0) + 1
        if per_job_count[job_id] > policy.keep_job_versions and _expired(row["observed_at"], version_cutoff):
            candidates["job_versions"].append(int(row["id"]))

    for row in conn.execute("SELECT episode_id, created_at FROM feedback_episodes"):
        if _expired(row["created_at"], audit_cutoff):
            candidates["feedback_episodes"].append(str(row["episode_id"]))
    for row in conn.execute("SELECT submission_id, updated_at FROM worker_submissions"):
        if _expired(row["updated_at"], audit_cutoff):
            candidates["worker_submissions"].append(str(row["submission_id"]))
    for row in conn.execute(
        "SELECT candidate_id, status, updated_at FROM recipe_candidates "
        "WHERE status IN ('rejected', 'revise', 'review_failed')"
    ):
        if _expired(row["updated_at"], audit_cutoff):
            candidates["recipe_candidates"].append(str(row["candidate_id"]))
    return candidates


def _delete_database_candidates(conn: sqlite3.Connection, candidates: dict[str, list[Any]]) -> None:
    contracts = {
        "job_versions": ("job_versions", "id"),
        "feedback_episodes": ("feedback_episodes", "episode_id"),
        "worker_submissions": ("worker_submissions", "submission_id"),
        "recipe_candidates": ("recipe_candidates", "candidate_id"),
    }
    for name, values in candidates.items():
        if not values:
            continue
        table, column = contracts[name]
        placeholders = ",".join("?" for _ in values)
        conn.execute(
            f"DELETE FROM {table} WHERE {column} IN ({placeholders})",
            values,
        )


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

    from shared.db.database import Database

    policy = policy or RetentionPolicy.from_env()
    current = now or datetime.now()
    Database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        referenced = _referenced_job_artifacts(conn)
        db_candidates = _database_cleanup_candidates(conn, policy, current)
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
            _delete_database_candidates(conn, db_candidates)
            for path in files:
                path.unlink(missing_ok=True)
            conn.commit()
        inventory = {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "jobs",
                "job_versions",
                "recipes",
                "recipe_candidates",
                "worker_submissions",
                "feedback_episodes",
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
            "database": {name: len(values) for name, values in db_candidates.items()},
            "inventory": inventory,
        }
    finally:
        conn.close()


__all__ = ["RetentionPolicy", "run_retention"]

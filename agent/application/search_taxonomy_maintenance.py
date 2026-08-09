"""검색 사전 적재와 공고 색인을 애플리케이션 시작 단계에서 준비한다."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent.application.job_taxonomy_linker import JobTaxonomyLinker
from agent.application.search_taxonomy_import_service import import_local_seed
from agent.application.search_taxonomy_service import (
    DEFAULT_LOCAL_SEED,
    SearchTaxonomyService,
)
from agent.application.search_taxonomy_utils import CORE_SOURCE_KEY
from shared.db.database import Database


def _installed_seed_version(db_path: Path) -> str:
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT version FROM taxonomy_sources WHERE source_key = ?",
            (CORE_SOURCE_KEY,),
        ).fetchone()
    finally:
        connection.close()
    return str(row[0]) if row is not None else ""


def prepare_search_taxonomy(
    db_path: str | Path,
    *,
    seed_path: str | Path = DEFAULT_LOCAL_SEED,
) -> SearchTaxonomyService:
    """DB 스키마와 로컬 사전을 준비하고 미완료 공고 색인을 복구한다."""

    resolved_db_path = Path(db_path)
    resolved_seed_path = Path(seed_path)
    Database(resolved_db_path)
    service = SearchTaxonomyService(resolved_db_path)
    linker = JobTaxonomyLinker(resolved_db_path)

    seed_changed = False
    if resolved_seed_path.exists():
        payload = json.loads(resolved_seed_path.read_text(encoding="utf-8"))
        expected_version = str(payload.get("source", {}).get("version") or "")
        seed_changed = _installed_seed_version(resolved_db_path) != expected_version
        if seed_changed:
            import_local_seed(resolved_db_path, resolved_seed_path)

    if seed_changed:
        linker.relink_all_jobs()
    else:
        linker.relink_pending_jobs(limit=100, max_attempts=2)
    return service


__all__ = ["prepare_search_taxonomy"]

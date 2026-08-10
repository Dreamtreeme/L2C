"""버전 관리되는 로컬 검색 사전을 SQLite에 적재한다."""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from shared.db.database import Database


def normalize_term(value: str) -> str:
    """유니코드, 공백과 대소문자를 검색 비교 형식으로 정규화한다."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return " ".join(normalized.split())


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    Database(db_path)
    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _upsert_source(
    connection: sqlite3.Connection,
    *,
    source_key: str,
    name: str,
    version: str,
    source_url: str,
    license_name: str,
    metadata: dict[str, Any],
) -> None:
    connection.execute(
        """
        INSERT INTO taxonomy_sources (
            source_key, name, version, source_url, license,
            imported_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_key) DO UPDATE SET
            name = excluded.name,
            version = excluded.version,
            source_url = excluded.source_url,
            license = excluded.license,
            imported_at = excluded.imported_at,
            metadata_json = excluded.metadata_json
        """,
        (
            source_key,
            name,
            version,
            source_url,
            license_name,
            _now(),
            json.dumps(metadata, ensure_ascii=False),
        ),
    )


def _upsert_concept(
    connection: sqlite3.Connection,
    *,
    concept_key: str,
    concept_type: str,
    source_key: str,
    preferred_label_ko: str = "",
    preferred_label_en: str = "",
    definition: str = "",
) -> int:
    now = _now()
    connection.execute(
        """
        INSERT INTO search_concepts (
            concept_key, concept_type, preferred_label_ko, preferred_label_en,
            definition, status, source_key, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
        ON CONFLICT(concept_key) DO UPDATE SET
            concept_type = excluded.concept_type,
            preferred_label_ko = excluded.preferred_label_ko,
            preferred_label_en = excluded.preferred_label_en,
            definition = excluded.definition,
            status = 'active',
            source_key = excluded.source_key,
            updated_at = excluded.updated_at
        """,
        (
            concept_key,
            concept_type,
            preferred_label_ko or None,
            preferred_label_en or None,
            definition or None,
            source_key,
            now,
            now,
        ),
    )
    row = connection.execute(
        "SELECT id FROM search_concepts WHERE concept_key = ?",
        (concept_key,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"검색 개념 저장 실패: {concept_key}")
    return int(row["id"])


def _upsert_alias(
    connection: sqlite3.Connection,
    *,
    concept_id: int,
    alias: str,
    language: str,
    source_key: str,
    alias_type: str = "exact",
) -> bool:
    normalized = normalize_term(alias)
    if not normalized:
        return False
    connection.execute(
        """
        INSERT INTO search_aliases (
            concept_id, alias, normalized_alias, language, alias_type,
            source_key, active, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
        ON CONFLICT(concept_id, normalized_alias, language) DO UPDATE SET
            alias = excluded.alias,
            alias_type = excluded.alias_type,
            source_key = excluded.source_key,
            active = 1
        """,
        (
            concept_id,
            alias.strip(),
            normalized,
            language,
            alias_type,
            source_key,
            _now(),
        ),
    )
    return True


def _upsert_broader_relation(
    connection: sqlite3.Connection,
    *,
    concept_id: int,
    broader_id: int,
    source_key: str,
) -> None:
    connection.execute(
        """
        INSERT INTO search_concept_relations (
            source_concept_id, target_concept_id, relation_type,
            source_key, metadata_json, created_at
        ) VALUES (?, ?, 'broader', ?, '{}', ?)
        ON CONFLICT(source_concept_id, target_concept_id, relation_type, source_key)
        DO UPDATE SET metadata_json = excluded.metadata_json
        """,
        (concept_id, broader_id, source_key, _now()),
    )


def import_local_seed(db_path: str | Path, seed_path: str | Path) -> dict[str, int]:
    """검토된 로컬 직무 계층과 기술 별칭을 적재한다."""

    seed = Path(seed_path)
    payload = json.loads(seed.read_text(encoding="utf-8"))
    source = dict(payload["source"])
    source_key = str(source["source_key"])
    concepts = list(payload.get("concepts") or [])
    counts = {"concepts": 0, "aliases": 0, "relations": 0}

    connection = _connect(db_path)
    try:
        with connection:
            _upsert_source(
                connection,
                source_key=source_key,
                name=str(source["name"]),
                version=str(source["version"]),
                source_url=str(source["source_url"]),
                license_name=str(source.get("license") or ""),
                metadata={"seed": seed.name},
            )
            connection.execute(
                "UPDATE search_aliases SET active = 0 WHERE source_key = ?",
                (source_key,),
            )
            connection.execute(
                "UPDATE search_concepts SET status = 'deprecated' WHERE source_key = ?",
                (source_key,),
            )
            connection.execute(
                "DELETE FROM search_concept_relations WHERE source_key = ?",
                (source_key,),
            )

            concept_ids: dict[str, int] = {}
            for item in concepts:
                concept_key = str(item["concept_key"])
                concept_id = _upsert_concept(
                    connection,
                    concept_key=concept_key,
                    concept_type=str(item["concept_type"]),
                    source_key=source_key,
                    preferred_label_ko=str(item.get("preferred_label_ko") or ""),
                    preferred_label_en=str(item.get("preferred_label_en") or ""),
                    definition=str(item.get("definition") or ""),
                )
                concept_ids[concept_key] = concept_id
                for language, label in (
                    ("ko", item.get("preferred_label_ko")),
                    ("en", item.get("preferred_label_en")),
                ):
                    if label and _upsert_alias(
                        connection,
                        concept_id=concept_id,
                        alias=str(label),
                        language=language,
                        alias_type="preferred",
                        source_key=source_key,
                    ):
                        counts["aliases"] += 1
                for alias in item.get("aliases") or []:
                    alias_text = str(alias)
                    if _upsert_alias(
                        connection,
                        concept_id=concept_id,
                        alias=alias_text,
                        language=(
                            "ko"
                            if any("가" <= char <= "힣" for char in alias_text)
                            else "en"
                        ),
                        source_key=source_key,
                    ):
                        counts["aliases"] += 1
                counts["concepts"] += 1

            for item in concepts:
                broader_key = str(item.get("broader") or "")
                if not broader_key:
                    continue
                _upsert_broader_relation(
                    connection,
                    concept_id=concept_ids[str(item["concept_key"])],
                    broader_id=concept_ids[broader_key],
                    source_key=source_key,
                )
                counts["relations"] += 1
    finally:
        connection.close()
    return counts


def taxonomy_counts(db_path: str | Path) -> dict[str, int]:
    """정적 사전과 공고 연결 건수를 반환한다."""

    connection = _connect(db_path)
    try:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "taxonomy_sources",
                "search_concepts",
                "search_aliases",
                "search_concept_relations",
                "job_concept_links",
            )
        }
    finally:
        connection.close()


__all__ = ["import_local_seed", "normalize_term", "taxonomy_counts"]

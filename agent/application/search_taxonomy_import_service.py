"""외부 직업·기술 사전과 한국어 핵심 개념을 SQLite에 적재한다."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from shared.db.database import Database


ONET_SOURCE_KEY = "onet_30_3"
ONET_VERSION = "30.3"
ONET_SOURCE_URL = "https://www.onetcenter.org/database.html"


def normalize_term(value: str) -> str:
    """표시 문자는 보존하면서 유니코드와 공백, 대소문자만 정규화한다."""

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
    downloaded_at: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO taxonomy_sources (
            source_key, name, version, source_url, license,
            downloaded_at, imported_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_key) DO UPDATE SET
            name = excluded.name,
            version = excluded.version,
            source_url = excluded.source_url,
            license = excluded.license,
            downloaded_at = excluded.downloaded_at,
            imported_at = excluded.imported_at,
            metadata_json = excluded.metadata_json
        """,
        (
            source_key,
            name,
            version,
            source_url,
            license_name,
            downloaded_at or None,
            _now(),
            json.dumps(metadata or {}, ensure_ascii=False),
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
    status: str = "active",
) -> int:
    now = _now()
    connection.execute(
        """
        INSERT INTO search_concepts (
            concept_key, concept_type, preferred_label_ko, preferred_label_en,
            definition, status, source_key, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(concept_key) DO UPDATE SET
            concept_type = excluded.concept_type,
            preferred_label_ko = excluded.preferred_label_ko,
            preferred_label_en = excluded.preferred_label_en,
            definition = excluded.definition,
            status = excluded.status,
            source_key = excluded.source_key,
            updated_at = excluded.updated_at
        """,
        (
            concept_key,
            concept_type,
            preferred_label_ko or None,
            preferred_label_en or None,
            definition or None,
            status,
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
) -> None:
    normalized = normalize_term(alias)
    if not normalized:
        return
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


def _upsert_external_mapping(
    connection: sqlite3.Connection,
    *,
    concept_id: int,
    source_key: str,
    external_id: str,
    source_version: str,
    external_url: str = "",
) -> None:
    connection.execute(
        """
        INSERT INTO search_external_mappings (
            concept_id, source_key, external_id, external_url,
            source_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(concept_id, source_key, external_id) DO UPDATE SET
            external_url = excluded.external_url,
            source_version = excluded.source_version
        """,
        (
            concept_id,
            source_key,
            external_id,
            external_url or None,
            source_version,
            _now(),
        ),
    )


def _upsert_relation(
    connection: sqlite3.Connection,
    *,
    source_concept_id: int,
    target_concept_id: int,
    relation_type: str,
    source_key: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO search_concept_relations (
            source_concept_id, target_concept_id, relation_type,
            source_key, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_concept_id, target_concept_id, relation_type, source_key)
        DO UPDATE SET metadata_json = excluded.metadata_json
        """,
        (
            source_concept_id,
            target_concept_id,
            relation_type,
            source_key,
            json.dumps(metadata or {}, ensure_ascii=False),
            _now(),
        ),
    )


def _zip_rows(archive: zipfile.ZipFile, filename: str) -> Iterable[dict[str, str]]:
    with archive.open(filename) as raw_stream:
        text_stream = io.TextIOWrapper(raw_stream, encoding="utf-8-sig", newline="")
        yield from csv.DictReader(text_stream)


def import_onet_archive(db_path: str | Path, archive_path: str | Path) -> dict[str, int]:
    """O*NET 세부 직업과 소프트웨어 기술 어휘를 적재한다."""

    archive_path = Path(archive_path)
    downloaded_at = datetime.fromtimestamp(archive_path.stat().st_mtime).astimezone().isoformat(
        timespec="seconds"
    )
    connection = _connect(db_path)
    occupation_ids: set[str] = set()
    skill_ids: dict[str, int] = {}
    counts = {"occupations": 0, "occupation_aliases": 0, "skills": 0, "relations": 0}
    try:
        with connection, zipfile.ZipFile(archive_path) as archive:
            _upsert_source(
                connection,
                source_key=ONET_SOURCE_KEY,
                name="O*NET Database",
                version=ONET_VERSION,
                source_url=ONET_SOURCE_URL,
                license_name="CC BY 4.0",
                downloaded_at=downloaded_at,
                metadata={"archive": archive_path.name},
            )
            connection.execute(
                "UPDATE search_concepts SET status = 'deprecated' WHERE source_key = ?",
                (ONET_SOURCE_KEY,),
            )
            connection.execute(
                "UPDATE search_aliases SET active = 0 WHERE source_key = ?",
                (ONET_SOURCE_KEY,),
            )

            occupation_filename = "db_30_3_csv/occupation_data.csv"
            if occupation_filename in archive.namelist():
                for row in _zip_rows(archive, occupation_filename):
                    external_id = str(row.get("O*NET-SOC Code") or "").strip()
                    title = str(row.get("Title") or "").strip()
                    if not external_id or not title:
                        continue
                    concept_key = f"onet:occupation:{external_id}"
                    concept_id = _upsert_concept(
                        connection,
                        concept_key=concept_key,
                        concept_type="occupation",
                        preferred_label_en=title,
                        definition=str(row.get("Description") or "").strip(),
                        source_key=ONET_SOURCE_KEY,
                    )
                    _upsert_alias(
                        connection,
                        concept_id=concept_id,
                        alias=title,
                        language="en",
                        alias_type="preferred",
                        source_key=ONET_SOURCE_KEY,
                    )
                    _upsert_external_mapping(
                        connection,
                        concept_id=concept_id,
                        source_key=ONET_SOURCE_KEY,
                        external_id=external_id,
                        source_version=ONET_VERSION,
                        external_url=(
                            f"https://www.onetonline.org/link/summary/{external_id}"
                        ),
                    )
                    if concept_key not in occupation_ids:
                        occupation_ids.add(concept_key)
                        counts["occupations"] += 1
                        counts["occupation_aliases"] += 1

            for row in _zip_rows(archive, "db_30_3_csv/software_skills.csv"):
                label = str(row.get("Workplace Example") or "").strip()
                normalized = normalize_term(label)
                if not normalized:
                    continue
                skill_key = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]
                concept_key = f"onet:skill:{skill_key}"
                skill_id = skill_ids.get(concept_key)
                if skill_id is None:
                    skill_id = _upsert_concept(
                        connection,
                        concept_key=concept_key,
                        concept_type="skill",
                        preferred_label_en=label,
                        source_key=ONET_SOURCE_KEY,
                    )
                    _upsert_alias(
                        connection,
                        concept_id=skill_id,
                        alias=label,
                        language="en",
                        alias_type="preferred",
                        source_key=ONET_SOURCE_KEY,
                    )
                    skill_ids[concept_key] = skill_id
                    counts["skills"] += 1
            connection.execute(
                """
                UPDATE search_term_candidates
                SET status = 'candidate', reviewed_at = NULL,
                    review_note = '외부 사전에서 개념이 제거되어 재검토 필요',
                    accepted_concept_key = NULL
                WHERE status = 'accepted'
                  AND accepted_concept_key IN (
                      SELECT concept_key
                      FROM search_concepts
                      WHERE source_key = ? AND status = 'deprecated'
                  )
                """,
                (ONET_SOURCE_KEY,),
            )
            connection.execute(
                """
                DELETE FROM search_concepts
                WHERE source_key = ? AND status = 'deprecated'
                """,
                (ONET_SOURCE_KEY,),
            )
            connection.execute(
                "UPDATE taxonomy_sources SET metadata_json = ? WHERE source_key = ?",
                (json.dumps({**counts, "archive": archive_path.name}, ensure_ascii=False), ONET_SOURCE_KEY),
            )
    finally:
        connection.close()
    return counts


def import_local_seed(db_path: str | Path, seed_path: str | Path) -> dict[str, int]:
    """검토된 한국어 직무 계층과 최신 기술 별칭을 적재한다."""

    payload = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    source = dict(payload["source"])
    concepts = list(payload.get("concepts") or [])
    connection = _connect(db_path)
    concept_ids: dict[str, int] = {}
    counts = {"concepts": 0, "aliases": 0, "relations": 0, "external_mappings": 0}
    try:
        with connection:
            _upsert_source(
                connection,
                source_key=str(source["source_key"]),
                name=str(source["name"]),
                version=str(source["version"]),
                source_url=str(source["source_url"]),
                license_name=str(source.get("license") or ""),
                metadata={"seed": Path(seed_path).name},
            )
            source_key = str(source["source_key"])
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
            connection.execute(
                """
                DELETE FROM search_external_mappings
                WHERE concept_id IN (
                    SELECT id FROM search_concepts WHERE source_key = ?
                )
                """,
                (source_key,),
            )
            for item in concepts:
                concept_id = _upsert_concept(
                    connection,
                    concept_key=str(item["concept_key"]),
                    concept_type=str(item["concept_type"]),
                    preferred_label_ko=str(item.get("preferred_label_ko") or ""),
                    preferred_label_en=str(item.get("preferred_label_en") or ""),
                    definition=str(item.get("definition") or ""),
                    source_key=str(source["source_key"]),
                )
                concept_ids[str(item["concept_key"])] = concept_id
                for language, label in (
                    ("ko", item.get("preferred_label_ko")),
                    ("en", item.get("preferred_label_en")),
                ):
                    if label:
                        _upsert_alias(
                            connection,
                            concept_id=concept_id,
                            alias=str(label),
                            language=language,
                            alias_type="preferred",
                            source_key=str(source["source_key"]),
                        )
                for alias in item.get("aliases") or []:
                    _upsert_alias(
                        connection,
                        concept_id=concept_id,
                        alias=str(alias),
                        language="ko" if any("가" <= char <= "힣" for char in str(alias)) else "en",
                        source_key=str(source["source_key"]),
                    )
                    counts["aliases"] += 1
                counts["concepts"] += 1

            for item in concepts:
                source_id = concept_ids[str(item["concept_key"])]
                broader_key = str(item.get("broader") or "")
                if broader_key:
                    _upsert_relation(
                        connection,
                        source_concept_id=source_id,
                        target_concept_id=concept_ids[broader_key],
                        relation_type="broader",
                        source_key=str(source["source_key"]),
                    )
                    counts["relations"] += 1
                for mapping in item.get("external_mappings") or []:
                    mapping_source = str(mapping["source_key"])
                    source_row = connection.execute(
                        "SELECT version FROM taxonomy_sources WHERE source_key = ?",
                        (mapping_source,),
                    ).fetchone()
                    if source_row is None:
                        continue
                    external_id = str(mapping["external_id"])
                    _upsert_external_mapping(
                        connection,
                        concept_id=source_id,
                        source_key=mapping_source,
                        external_id=external_id,
                        source_version=str(source_row["version"]),
                        external_url=f"https://www.onetonline.org/link/summary/{external_id}",
                    )
                    counts["external_mappings"] += 1

            onet_occupations = connection.execute(
                """
                SELECT concepts.id, mappings.external_id
                FROM search_concepts AS concepts
                JOIN search_external_mappings AS mappings
                  ON mappings.concept_id = concepts.id
                 AND mappings.source_key = ?
                WHERE concepts.source_key = ?
                  AND concepts.concept_type = 'occupation'
                  AND concepts.status = 'active'
                """,
                (ONET_SOURCE_KEY, ONET_SOURCE_KEY),
            ).fetchall()
            for item in concepts:
                major_groups = {
                    str(value).strip()
                    for value in item.get("soc_major_groups") or []
                    if str(value).strip()
                }
                if not major_groups:
                    continue
                parent_id = concept_ids[str(item["concept_key"])]
                for occupation in onet_occupations:
                    external_id = str(occupation["external_id"])
                    if external_id[:2] not in major_groups:
                        continue
                    _upsert_relation(
                        connection,
                        source_concept_id=int(occupation["id"]),
                        target_concept_id=parent_id,
                        relation_type="broader",
                        source_key=source_key,
                        metadata={"soc_major_group": external_id[:2]},
                    )
                    counts["relations"] += 1
    finally:
        connection.close()
    return counts


def taxonomy_counts(db_path: str | Path) -> dict[str, int]:
    """적재 검증과 운영 화면에서 사용할 사전 테이블별 건수를 반환한다."""

    connection = _connect(db_path)
    try:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "taxonomy_sources",
                "search_concepts",
                "search_aliases",
                "search_concept_relations",
                "search_external_mappings",
                "job_concept_links",
                "search_term_candidates",
                "search_term_candidate_observations",
            )
        }
    finally:
        connection.close()


__all__ = [
    "ONET_SOURCE_KEY",
    "ONET_SOURCE_URL",
    "ONET_VERSION",
    "import_local_seed",
    "import_onet_archive",
    "normalize_term",
    "taxonomy_counts",
]

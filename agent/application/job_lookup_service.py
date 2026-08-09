"""수집 에이전트가 사용할 채용공고 읽기 전용 조회 서비스."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from agent.config import get_settings
from agent.runtime.job_identity import (
    canonical_company_name,
    canonical_position_title,
    source_card_key,
)
from agent.sites.loader import SiteProfileError, load_site_profile
from agent.utils.logger import logger
from agent.utils.text import site_of


def _site_identity_scope(site_url: str) -> tuple[str, set[str]] | None:
    site_host = site_of(site_url)
    if not site_host:
        return None
    try:
        site_entry = load_site_profile(site_host)
    except SiteProfileError:
        return None
    source_platform = str(site_entry.source_platform or "").strip().casefold()
    site_domains = {
        str(domain or "").strip().lower().removeprefix("www.")
        for domain in site_entry.domains
        if str(domain or "").strip()
    }
    return source_platform, site_domains


def _load_job_identity_rows(db_path: Path) -> list[tuple[Any, ...]] | None:
    try:
        connection = sqlite3.connect(
            f"file:{db_path.as_posix()}?mode=ro",
            uri=True,
        )
        try:
            return connection.execute(
                "SELECT id, company_name, position, url, source_platform FROM jobs"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        logger.warning("Job card identity lookup failed", error=str(exc))
        return None


def _index_job_identities(
    rows: list[tuple[Any, ...]],
    *,
    source_platform: str,
    site_domains: set[str],
    card_keys: list[str],
) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    matches: dict[tuple[str, str], int] = {}
    card_key_matches: dict[str, int] = {}
    for job_id, stored_company, stored_position, stored_url, stored_source in rows:
        same_source = bool(
            source_platform
            and str(stored_source or "").strip().casefold() == source_platform
        )
        stored_host = site_of(str(stored_url or ""))
        if not same_source and not (stored_host and stored_host in site_domains):
            continue
        stored_url_text = str(stored_url or "")
        for card_key in card_keys:
            if card_key and f"l2c-card={card_key}" in stored_url_text:
                card_key_matches.setdefault(card_key, int(job_id))
        key = (
            canonical_company_name(stored_company),
            canonical_position_title(stored_position),
        )
        if key[0] and key[1]:
            matches.setdefault(key, int(job_id))
    return matches, card_key_matches


def find_job_id_by_url(url: str, *, db_path: str | Path | None = None) -> int | None:
    """끝 슬래시 차이만 허용해 동일한 저장 URL의 공고 ID를 찾는다."""

    normalized = str(url or "").strip().rstrip("/")
    if not normalized:
        return None
    if db_path is None:
        db_path = get_settings().paths.db_path
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


def find_job_ids_by_card_identities(
    identities: list[tuple[Any, Any]],
    site_url: str,
    *,
    db_path: str | Path | None = None,
) -> list[int | None]:
    """여러 카드의 동일 사이트·회사명·제목 일치 공고를 한 번의 DB 조회로 찾는다."""

    keys = [
        (
            canonical_company_name(company_name),
            canonical_position_title(position),
        )
        for company_name, position in identities
    ]
    card_keys = [
        source_card_key(site_url, company_name, position)
        for company_name, position in identities
    ]
    site_scope = _site_identity_scope(site_url)
    if not keys or site_scope is None:
        return [None] * len(keys)
    source_platform, site_domains = site_scope

    if db_path is None:
        db_path = get_settings().paths.db_path
    resolved = Path(db_path)
    if not resolved.exists():
        return [None] * len(keys)

    rows = _load_job_identity_rows(resolved)
    if rows is None:
        return [None] * len(keys)

    matches, card_key_matches = _index_job_identities(
        rows,
        source_platform=source_platform,
        site_domains=site_domains,
        card_keys=card_keys,
    )
    return [
        card_key_matches.get(card_key) or matches.get(key)
        if key[0] and key[1]
        else None
        for key, card_key in zip(keys, card_keys)
    ]


__all__ = [
    "find_job_ids_by_card_identities",
    "find_job_id_by_url",
]

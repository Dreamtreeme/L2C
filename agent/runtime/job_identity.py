"""화면 카드와 저장 공고를 연결하는 결정론적 정체성 유틸리티."""

from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from agent.utils.text import normalize_text, site_of
from agent.sites.loader import SiteProfileError, load_site_profile


_COMPANY_DESIGNATORS = (
    "주식회사",
    "유한회사",
    "(주)",
    "㈜",
    "(유)",
)


def canonical_company_name(value: Any) -> str:
    """법인 표기와 문장부호를 제외한 회사명 비교 키를 만든다."""

    text = normalize_text(value).casefold()
    for designator in _COMPANY_DESIGNATORS:
        text = text.replace(designator, "")
    return "".join(char for char in text if char.isalnum())


def canonical_position_title(value: Any) -> str:
    """공백과 문장부호 차이를 제외한 화면 공고 제목 키를 만든다."""

    text = normalize_text(value).casefold()
    return "".join(char for char in text if char.isalnum())


def source_card_key(site_url: str, company_name: Any, position: Any) -> str:
    """사이트와 화면 카드 정체성으로 재현 가능한 짧은 키를 만든다."""

    company_key = canonical_company_name(company_name)
    position_key = canonical_position_title(position)
    if not company_key or not position_key:
        return ""
    try:
        site = load_site_profile(site_of(site_url)).slug.strip().casefold()
    except SiteProfileError:
        site = site_of(site_url)
    if not site:
        return ""
    payload = f"{site}\n{company_key}\n{position_key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def url_with_source_card_key(url: str, card_key: str) -> str:
    """검색 URL을 보존하면서 화면 카드별 로컬 식별 fragment를 붙인다."""

    raw_url = str(url or "").strip()
    if not raw_url or not card_key:
        return raw_url
    parts = urlsplit(raw_url)
    fragment_items = dict(parse_qsl(parts.fragment, keep_blank_values=True))
    fragment_items["l2c-card"] = card_key
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            parts.query,
            urlencode(fragment_items),
        )
    )


def canonical_job_url(value: Any) -> str:
    """사이트가 노출한 공고 식별자만 남긴 상세 URL을 반환한다."""

    raw_url = str(value or "").strip()
    if not raw_url:
        return ""
    parts = urlsplit(raw_url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return raw_url.rstrip("/")
    try:
        profile = load_site_profile(site_of(raw_url))
    except SiteProfileError:
        return raw_url.rstrip("/")

    detail = profile.page_guidance.get("job_detail")
    target = f"{parts.path}?{parts.query}" if parts.query else parts.path
    if not detail or not any(
        re.search(pattern, target, flags=re.IGNORECASE)
        for pattern in detail.url_patterns
    ):
        return raw_url.rstrip("/")

    query_items = parse_qsl(parts.query, keep_blank_values=True)
    identity_keys = {
        key.casefold(): key for key in profile.job_identity_query_keys
    }
    if identity_keys:
        selected = {
            key.casefold(): value
            for key, value in query_items
            if key.casefold() in identity_keys and value
        }
        if not selected:
            return raw_url.rstrip("/")
        canonical_query = urlencode(
            [
                (
                    identity_keys[key.casefold()],
                    selected[key.casefold()],
                )
                for key in profile.job_identity_query_keys
                if key.casefold() in selected
            ]
        )
    else:
        canonical_query = ""

    canonical_host = urlsplit(profile.base_url).netloc.lower()
    return urlunsplit(
        (
            "https",
            canonical_host,
            parts.path.rstrip("/") or "/",
            canonical_query,
            "",
        )
    )


__all__ = [
    "canonical_company_name",
    "canonical_job_url",
    "canonical_position_title",
    "source_card_key",
    "url_with_source_card_key",
]

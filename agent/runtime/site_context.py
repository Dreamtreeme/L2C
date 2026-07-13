"""현재 URL에 대응하는 사이트 프로필과 페이지 정책을 조회한다."""

from __future__ import annotations

import re
from urllib.parse import urlparse


def site_profile_for_url(url: str) -> dict:
    """URL 호스트와 일치하는 활성·비활성 사이트 프로필을 반환한다."""

    try:
        host = (urlparse(url or "").netloc or "").lower()
    except Exception:
        return {}
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return {}
    try:
        from agent.sites import list_supported_sites, load_site_profile

        for entry in list_supported_sites(enabled_only=False):
            domains = [str(domain or "").lower() for domain in entry.get("domains", [])]
            if any(host == domain or host.endswith("." + domain) for domain in domains):
                return load_site_profile(str(entry.get("slug") or ""))
    except Exception:
        return {}
    return {}


def persistence_policy_for_url(url: str) -> dict:
    """URL에 해당하는 사이트의 저장 정책을 반환한다."""

    profile = site_profile_for_url(url)
    manual = profile.get("manual", {}) if isinstance(profile, dict) else {}
    policy = manual.get("persistence_policy", {}) if isinstance(manual, dict) else {}
    return policy if isinstance(policy, dict) else {}


def looks_like_job_detail_url(url: str) -> bool:
    """사이트 프로필의 상세 URL 정규식으로 현재 페이지 역할을 판정한다."""

    pattern = str(persistence_policy_for_url(url).get("detail_url_pattern") or "").strip()
    return bool(url and pattern and re.search(pattern, url))


__all__ = [
    "looks_like_job_detail_url",
    "persistence_policy_for_url",
    "site_profile_for_url",
]

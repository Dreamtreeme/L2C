"""현재 URL에 대응하는 사이트 프로필과 선언형 화면 안내를 조회한다."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from agent.recipe.page_context import normalize_page_role
from agent.recipe.text_utils import normalize_text
from agent.sites.profile import PageGuidance, SiteProfile


@lru_cache(maxsize=32)
def _site_profile_for_host(host: str) -> SiteProfile | None:
    """같은 작업 중 반복 조회되는 사이트 프로필을 호스트 단위로 재사용한다."""

    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None
    try:
        from agent.sites import list_supported_sites, load_site_profile

        for profile in list_supported_sites(enabled_only=False):
            domains = [str(domain or "").lower() for domain in profile.domains]
            if any(host == domain or host.endswith("." + domain) for domain in domains):
                return load_site_profile(profile.slug)
    except Exception:
        return None
    return None


def site_profile_for_url(url: str) -> SiteProfile | None:
    """URL 호스트와 일치하는 활성·비활성 사이트 프로필을 반환한다."""

    try:
        host = (urlparse(url or "").netloc or "").lower()
    except Exception:
        return None
    return _site_profile_for_host(host)


def persistence_policy_for_url(url: str) -> dict:
    """URL에 해당하는 사이트의 저장 정책을 반환한다."""

    profile = site_profile_for_url(url)
    return profile.persistence_policy.model_dump(mode="json") if profile else {}


def page_guidance_for_url(url: str, page_role: str) -> dict[str, Any]:
    """현재 사이트와 화면 역할에 해당하는 선언형 안내만 반환한다."""

    profile = site_profile_for_url(url)
    guidance = profile.page_guidance if profile else {}
    role = normalize_page_role(page_role)
    value = guidance.get(role)
    return value.model_dump(mode="json") if isinstance(value, PageGuidance) else {}


def _matches_declared_url_pattern(url: str, patterns: list[Any]) -> bool:
    for raw_pattern in patterns:
        pattern = str(raw_pattern or "").strip()
        if not pattern:
            continue
        try:
            if re.search(pattern, url or "", flags=re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def _matched_visible_cues(marker_texts: list[Any], cues: list[Any]) -> list[str]:
    page_text = normalize_text(" ".join(str(text or "") for text in marker_texts)).casefold()
    page_text = re.sub(r"\s+", "", page_text)
    matched: list[str] = []
    for raw_cue in cues:
        cue = re.sub(r"\s+", "", normalize_text(raw_cue).casefold())
        if cue and cue in page_text:
            matched.append(str(raw_cue))
    return matched


def infer_site_page_role(url: str, marker_texts: list[Any] | None = None) -> str:
    """사이트 설명서에 선언된 증거만으로 화면 역할을 보수적으로 추정한다."""

    texts = list(marker_texts or [])
    profile = site_profile_for_url(url)
    guidance = profile.page_guidance if profile else {}
    if guidance:
        cue_guidance = sorted(
            guidance.items(),
            key=lambda item: normalize_page_role(item[0]) != "job_detail",
        )
        for raw_role, config in cue_guidance:
            raw_config = config.model_dump(mode="json")
            role = normalize_page_role(raw_role)
            patterns = list(raw_config.get("url_patterns") or [])
            if url and _matches_declared_url_pattern(url, patterns):
                return role

        for raw_role, config in cue_guidance:
            raw_config = config.model_dump(mode="json")
            cues = list(raw_config.get("visible_cues") or [])
            matched = _matched_visible_cues(texts, cues)
            try:
                minimum = max(1, int(raw_config.get("minimum_visible_cues") or 2))
            except (TypeError, ValueError):
                minimum = 2
            if cues and len(matched) >= minimum:
                return normalize_page_role(raw_role)
    try:
        parsed = urlparse(url or "")
        if profile and parsed.netloc and (parsed.path or "/") in {"", "/"}:
            return "home"
    except Exception:
        pass
    return ""


def job_detail_context_evidence(
    url: str,
    *,
    page_role: str = "",
    marker_texts: list[Any] | None = None,
) -> dict[str, Any]:
    """상세 화면 여부를 URL 하나에 고정하지 않고 독립적인 증거로 판정한다."""

    role = normalize_page_role(page_role)
    guidance = page_guidance_for_url(url, "job_detail")
    patterns = list(guidance.get("url_patterns") or [])
    url_matched = bool(url and _matches_declared_url_pattern(url, patterns))
    cues = list(guidance.get("visible_cues") or [])
    matched_cues = _matched_visible_cues(list(marker_texts or []), cues)
    try:
        minimum = max(1, int(guidance.get("minimum_visible_cues") or 2))
    except (TypeError, ValueError):
        minimum = 2
    role_matched = role == "job_detail"
    cue_matched = bool(cues and len(matched_cues) >= minimum)
    if role_matched:
        reason = "page_role"
    elif url_matched:
        reason = "declared_url_pattern"
    elif cue_matched:
        reason = "declared_visible_cues"
    else:
        reason = "no_detail_evidence"
    return {
        "matched": role_matched or url_matched or cue_matched,
        "reason": reason,
        "page_role": role,
        "url_pattern_matched": url_matched,
        "matched_visible_cues": matched_cues,
    }


def is_job_detail_context(
    url: str,
    *,
    page_role: str = "",
    marker_texts: list[Any] | None = None,
) -> bool:
    """현재 화면이 상세 공고라는 충분한 증거가 있는지 반환한다."""

    return bool(
        job_detail_context_evidence(
            url,
            page_role=page_role,
            marker_texts=marker_texts,
        )["matched"]
    )


def site_runtime_guidance(url: str, page_role: str) -> str:
    """현재 사이트의 현재 화면에 필요한 안내만 짧게 렌더링한다."""

    profile = site_profile_for_url(url)
    if not profile:
        return ""
    role = normalize_page_role(page_role) or "unknown"
    guidance = page_guidance_for_url(url, role)
    if not guidance:
        return ""
    display_name = profile.display_name or profile.slug
    parts = [f"현재 사이트 안내: {display_name} / {role}"]
    instructions = [str(item).strip() for item in guidance.get("instructions", []) if str(item).strip()]
    reading_targets = [str(item).strip() for item in guidance.get("reading_targets", []) if str(item).strip()]
    navigation_notes = [str(item).strip() for item in guidance.get("navigation_notes", []) if str(item).strip()]
    if instructions:
        parts.append("- 이 화면에서 할 일: " + " ".join(instructions[:3]))
    if reading_targets:
        parts.append("- 확인할 정보: " + ", ".join(reading_targets[:12]))
    if navigation_notes:
        parts.append("- 이동 참고: " + " ".join(navigation_notes[:2]))
    return "\n".join(parts) + "\n\n"


def looks_like_job_detail_url(url: str) -> bool:
    """사이트 선언의 상세 URL 패턴이 맞는지 보조 신호로 확인한다."""

    guidance = page_guidance_for_url(url, "job_detail")
    patterns = list(guidance.get("url_patterns") or [])
    return bool(url and _matches_declared_url_pattern(url, patterns))


__all__ = [
    "infer_site_page_role",
    "is_job_detail_context",
    "job_detail_context_evidence",
    "looks_like_job_detail_url",
    "page_guidance_for_url",
    "persistence_policy_for_url",
    "site_runtime_guidance",
    "site_profile_for_url",
]

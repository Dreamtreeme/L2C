"""단일 `profile.json` 계약을 사용하는 사이트 프로필 저장소."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from agent.sites.profile import SiteProfile


SITES_DIR = Path(__file__).resolve().parent
PROFILE_FILE_NAME = "profile.json"


class SiteProfileError(ValueError):
    """사이트 프로필이 없거나 계약을 만족하지 않을 때 발생한다."""


def _load_profile_file(path: Path) -> SiteProfile:
    try:
        return SiteProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SiteProfileError(f"사이트 프로필 파일을 찾을 수 없습니다: {path}") from exc
    except (ValidationError, ValueError) as exc:
        raise SiteProfileError(f"잘못된 사이트 프로필입니다: {path}: {exc}") from exc


@lru_cache(maxsize=1)
def _all_profiles() -> tuple[SiteProfile, ...]:
    paths = sorted(SITES_DIR.glob(f"*/{PROFILE_FILE_NAME}"))
    profiles = tuple(
        sorted(
            (_load_profile_file(path) for path in paths),
            key=lambda profile: (profile.registration_order, profile.slug),
        )
    )
    if not profiles:
        raise SiteProfileError("등록된 사이트 profile.json이 없습니다.")

    seen_slugs: set[str] = set()
    seen_orders: set[int] = set()
    seen_domains: dict[str, str] = {}
    for profile in profiles:
        if profile.slug in seen_slugs:
            raise SiteProfileError(f"중복 사이트 slug입니다: {profile.slug}")
        seen_slugs.add(profile.slug)
        if profile.registration_order in seen_orders:
            raise SiteProfileError(
                f"중복 사이트 registration_order입니다: {profile.registration_order}"
            )
        seen_orders.add(profile.registration_order)
        for domain in profile.domains:
            owner = seen_domains.get(domain)
            if owner and owner != profile.slug:
                raise SiteProfileError(
                    f"사이트 도메인이 중복 등록되었습니다: {domain} ({owner}, {profile.slug})"
                )
            seen_domains[domain] = profile.slug
    return profiles


def list_supported_sites(enabled_only: bool = True) -> list[SiteProfile]:
    profiles = list(_all_profiles())
    return [profile for profile in profiles if profile.enabled] if enabled_only else profiles


def load_site_profile(site: str) -> SiteProfile:
    needle = str(site or "").strip()
    if not needle:
        raise SiteProfileError("site 값이 필요합니다.")
    for profile in _all_profiles():
        if profile.matches(needle):
            return profile
    raise SiteProfileError(f"지원하지 않는 사이트입니다: {site}")


def get_official_site_url(site: str) -> str:
    return load_site_profile(site).base_url.rstrip("/")


def validate_site_profiles() -> tuple[SiteProfile, ...]:
    """서버 시작 시 전체 프로필과 사이트 간 유일성을 한 번에 검증한다."""

    return _all_profiles()


__all__ = [
    "PROFILE_FILE_NAME",
    "SITES_DIR",
    "SiteProfileError",
    "get_official_site_url",
    "list_supported_sites",
    "load_site_profile",
    "validate_site_profiles",
]

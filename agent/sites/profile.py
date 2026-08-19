"""비전 작업자가 사용할 사이트별 선언을 검증하는 타입 계약."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SiteModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PageGuidance(SiteModel):
    url_patterns: tuple[str, ...] = ()
    visible_cues: tuple[str, ...] = ()
    minimum_visible_cues: int = Field(2, ge=1)
    reveal_controls: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()
    reading_targets: tuple[str, ...] = ()
    navigation_notes: tuple[str, ...] = ()

    @field_validator("url_patterns")
    @classmethod
    def validate_patterns(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"잘못된 화면 URL 패턴입니다: {pattern}: {exc}") from exc
        return patterns


class SiteProfile(SiteModel):
    registration_order: int = Field(100, ge=0)
    slug: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    display_name: str
    source_platform: str
    aliases: tuple[str, ...] = ()
    domains: tuple[str, ...]
    base_url: str
    job_identity_query_keys: tuple[str, ...] = ()
    enabled: bool = True
    page_guidance: dict[str, PageGuidance]
    guidance: str = ""
    capabilities: dict[str, str] = Field(default_factory=dict)

    @field_validator("domains")
    @classmethod
    def normalize_domains(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(value).strip().lower() for value in values if str(value).strip())
        if not normalized:
            raise ValueError("사이트 도메인이 하나 이상 필요합니다.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("사이트 도메인이 중복되었습니다.")
        return normalized

    @model_validator(mode="after")
    def validate_identity(self) -> "SiteProfile":
        parsed = urlsplit(self.base_url)
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise ValueError("base_url은 공식 HTTPS 주소여야 합니다.")
        if parsed.hostname.lower() not in self.domains:
            raise ValueError("base_url 호스트가 domains에 선언되어야 합니다.")
        required_roles = {"home", "search", "job_detail"}
        missing_roles = sorted(required_roles - set(self.page_guidance))
        if missing_roles:
            raise ValueError(f"필수 화면 역할이 없습니다: {', '.join(missing_roles)}")
        return self

    def matches(self, value: str) -> bool:
        raw = str(value or "").strip().casefold()
        if not raw:
            return False
        parsed = urlsplit(raw if "://" in raw else f"//{raw}")
        host = str(parsed.hostname or "").lower()
        candidates = {
            self.slug.casefold(),
            self.display_name.casefold(),
            *(alias.casefold() for alias in self.aliases),
            *(domain.casefold() for domain in self.domains),
        }
        if raw in candidates or host in self.domains:
            return True
        return any(raw.endswith(domain) or host.endswith("." + domain) for domain in self.domains)

__all__ = [
    "PageGuidance",
    "SiteProfile",
]

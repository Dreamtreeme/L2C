"""구조화 추출 직후 공고의 결정론적 파생값을 한 번 계산한다."""

from __future__ import annotations

import hashlib
import re

from agent.runtime.site_context import site_profile_for_url
from shared.schema.jd_schema import JobPosting


_EXPERIENCE_RANGE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})\s*(?:년\s*)?[-~–—]\s*\d{1,2}\s*년"
)
_MINIMUM_EXPERIENCE_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*년\s*이상")


def source_platform_for_url(url: str | None) -> str | None:
    """사이트 레지스트리에 선언된 저장용 플랫폼 이름을 반환한다."""

    profile = site_profile_for_url(str(url or ""))
    value = str(profile.source_platform if profile else "").strip()
    return value or None


def _job_content_hash(posting: JobPosting) -> str:
    """회사명, 직무명과 자격요건으로 중복 후보 그룹 키를 만든다."""

    def normalized(value: object) -> str:
        return re.sub(r"\s+", "", str(value or "")).casefold()

    payload = "|".join(
        (
            normalized(posting.company_name),
            normalized(posting.position),
            "".join(normalized(item) for item in posting.requirements),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_experience_minimum(posting: JobPosting) -> JobPosting:
    """명시된 필수 경력 중 가장 높은 최소 연차를 정규 필드에 반영한다."""

    candidates = (
        [posting.experience_min]
        if posting.experience_min is not None
        else []
    )
    for text in (posting.experience_text, *posting.requirements):
        value = str(text or "")
        candidates.extend(
            int(match.group(1))
            for pattern in (
                _EXPERIENCE_RANGE_PATTERN,
                _MINIMUM_EXPERIENCE_PATTERN,
            )
            for match in pattern.finditer(value)
        )
    experience_min = max(candidates) if candidates else None
    if experience_min == posting.experience_min:
        return posting
    return posting.model_copy(update={"experience_min": experience_min})


def complete_extracted_job(
    posting: JobPosting,
    *,
    current_url: str,
    raw_ocr_text: str,
) -> JobPosting:
    """화면에서 확정한 출처와 공고 파생값을 구조화 결과에 결합한다."""

    url = str(posting.url or current_url).strip()
    completed = normalize_experience_minimum(posting).model_copy(
        update={
            "url": url or None,
            "raw_ocr_text": raw_ocr_text,
            "source_platform": posting.source_platform or source_platform_for_url(url),
        }
    )
    return completed.model_copy(
        update={"content_hash": completed.content_hash or _job_content_hash(completed)}
    )


__all__ = [
    "complete_extracted_job",
    "normalize_experience_minimum",
    "source_platform_for_url",
]

"""지휘자가 실행 전에 참조하는 도구 능력 목록."""

from __future__ import annotations

from agent.sites import list_supported_sites
from agent.sites.profile import SiteProfile
from shared.schema.agent_contract import ANSWER_EVIDENCE_FIELDS
from shared.schema.investigation_schema import ToolCapability


def _site_capability(profile: SiteProfile) -> ToolCapability:
    declared = profile.capabilities
    slug = profile.slug
    return ToolCapability(
        tool_name=f"realtime_scraping:{slug}",
        purpose=f"{profile.display_name or slug} 공개 채용공고 수집",
        supported_operations=[
            "public_search",
            "visible_result_collection",
            "detail_reading",
        ],
        supported_filters={
            "keyword": "supported",
            "posted_date": str(declared.get("posted_date_filter") or "unknown"),
            "location": str(declared.get("location_filter") or "unknown"),
            "experience": str(declared.get("experience_filter") or "unknown"),
            "employment_type": str(declared.get("employment_type_filter") or "unknown"),
        },
        verifiable_fields={
            **{field: "best_effort" for field in ANSWER_EVIDENCE_FIELDS},
            "company_name": "required",
            "position": "required",
            "url": "required",
            "posted_at": str(declared.get("posted_at_evidence") or "best_effort"),
        },
        limitations=[
            "첫 번째 안정적인 검색 결과 화면에 보이는 공고를 기준으로 수집합니다.",
            "화면에 게시일 근거가 없으면 기간 조건을 검증할 수 없습니다.",
            "로그인이나 인증이 필요한 흐름은 지원하지 않습니다.",
        ],
        expected_latency="브라우저·OCR·상세 공고 수에 따라 수십 초 이상",
    )


def build_collection_capabilities() -> list[ToolCapability]:
    """행동계획 모델이 선택할 수 있는 사이트 수집 능력을 반환한다."""

    return [
        _site_capability(profile) for profile in list_supported_sites(enabled_only=True)
    ]


__all__ = ["build_collection_capabilities"]

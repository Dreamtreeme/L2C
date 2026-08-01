"""지휘자가 실행 전에 참조하는 도구 능력 목록."""

from __future__ import annotations

from shared.schema.agent_contract import EVIDENCE_FIELDS
from shared.schema.investigation_schema import ToolCapability
from agent.sites.profile import SiteProfile


def _site_capability(profile: SiteProfile) -> ToolCapability:
    declared = profile.capabilities
    slug = profile.slug
    return ToolCapability(
        tool_name=f"realtime_scraping:{slug}",
        purpose=f"{profile.display_name or slug} 공개 채용공고 수집",
        supported_operations=["public_search", "visible_result_collection", "detail_reading"],
        supported_filters={
            "keyword": "supported",
            "posted_date": str(declared.get("posted_date_filter") or "unknown"),
            "location": str(declared.get("location_filter") or "unknown"),
            "experience": str(declared.get("experience_filter") or "unknown"),
            "employment_type": str(declared.get("employment_type_filter") or "unknown"),
        },
        verifiable_fields={
            **{field: "best_effort" for field in EVIDENCE_FIELDS},
            "company_name": "required",
            "position": "required",
            "url": "required",
            "posted_at": str(declared.get("posted_at_evidence") or "best_effort"),
        },
        limitations=[
            "첫 번째 안정적인 검색 결과 화면에 보이는 공고를 기준으로 수집합니다.",
            "화면에 게시일 근거가 없으면 기간 조건을 검증할 수 없습니다.",
            "로그인이나 인증이 필요한 흐름은 사용자 승인 없이 진행하지 않습니다.",
        ],
        expected_latency="브라우저·OCR·상세 공고 수에 따라 수십 초 이상",
    )


def build_tool_capability_catalog() -> list[ToolCapability]:
    """외부 행동 없이 현재 구성에서 사용할 수 있는 도구 능력을 반환한다."""

    capabilities = [
        ToolCapability(
            tool_name="inspect_job_evidence",
            purpose="DB에 답변 근거가 충분한지 구조적으로 확인",
            supported_operations=["coverage", "date_coverage", "field_coverage", "cohort_count"],
            supported_filters={
                "keyword": "supported",
                "posted_date": "supported_when_verified",
                "site": "supported",
            },
            verifiable_fields={
                "posted_at": "verified_values_only",
                "created_at": "collection_time_only",
                "source_platform": "supported",
            },
            limitations=["created_at은 공고 게시일 근거로 사용할 수 없습니다."],
            expected_latency="1초 이내",
        ),
    ]
    from agent.sites import list_supported_sites

    capabilities.extend(
        _site_capability(profile)
        for profile in list_supported_sites(enabled_only=True)
    )
    return capabilities


__all__ = ["build_tool_capability_catalog"]

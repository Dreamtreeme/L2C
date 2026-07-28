import json
import logging
from langchain_core.tools import tool
from agent.sites.profile import SiteProfile

logger = logging.getLogger(__name__)


def _site_entry_payload(profile: SiteProfile) -> dict[str, object]:
    return {
        "slug": profile.slug,
        "display_name": profile.display_name,
        "domains": list(profile.domains),
        "base_url": profile.base_url,
        "enabled": profile.enabled,
    }


@tool
def list_collection_sites(enabled_only: bool = True) -> str:
    """
    지휘자가 실시간 채용공고 수집에 사용할 수 있는 사이트 목록을 조회합니다.
    등록된 프로필에서 slug, 공식 주소, 도메인을 JSON으로 반환합니다.
    """
    try:
        from agent.sites import list_supported_sites

        sites = [_site_entry_payload(profile) for profile in list_supported_sites(enabled_only=enabled_only)]
        return json.dumps(
            {
                "count": len(sites),
                "enabled_only": enabled_only,
                "sites": sites,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        logger.error("[list_collection_sites] Failed to load site registry: %s", e, exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def get_collection_site_profile(site: str) -> str:
    """
    특정 채용 사이트의 지휘자용 프로필을 조회합니다.
    site는 slug, 표시명, 도메인 중 하나를 사용할 수 있습니다.
    반환값에는 검증된 단일 사이트 프로필이 포함됩니다.
    """
    try:
        from agent.sites import load_site_profile

        profile = load_site_profile(site)
        payload = {
            "site": _site_entry_payload(profile),
            "profile": profile.prompt_payload(),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("[get_collection_site_profile] Failed to load site profile for %r: %s", site, e, exc_info=True)
        return json.dumps({"error": str(e), "site": site}, ensure_ascii=False)

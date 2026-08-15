"""사이트별 어댑터 패키지.

각 어댑터는 SiteAdapter ABC를 구현하며,
URL 매칭과 페이지에서 본문을 추출하는 책임만 가집니다.
브라우저 lifecycle은 capture.py가 관리합니다.
"""

from .base import CollectionSiteAdapter, SiteAdapter
from .incruit import IncruitAdapter
from .wanted import WantedAdapter
from .jobkorea import JobKoreaAdapter
from .rocketpunch import RocketpunchAdapter

# 디스패처가 순서대로 matches()를 호출하므로,
# 더 구체적인 어댑터를 위에 두는 게 안전합니다.
ADAPTERS: list[SiteAdapter] = [
    IncruitAdapter(),
    WantedAdapter(),
    JobKoreaAdapter(),
    RocketpunchAdapter(),
]


def resolve_adapter(url: str) -> SiteAdapter:
    """URL에 맞는 어댑터를 반환. 없으면 ValueError."""
    for adapter in ADAPTERS:
        if adapter.matches(url):
            return adapter
    raise ValueError(
        f"지원되지 않는 사이트입니다: {url}\n"
        f"현재 지원: {', '.join(a.name for a in ADAPTERS)}"
    )


def resolve_collection_adapter(url: str) -> CollectionSiteAdapter:
    """URL을 검색부터 처리할 수 있는 어댑터를 반환한다."""

    adapter = resolve_adapter(url)
    if isinstance(adapter, CollectionSiteAdapter):
        return adapter
    raise ValueError(f"{adapter.name} 어댑터는 홈페이지 검색을 지원하지 않습니다.")


__all__ = [
    "SiteAdapter",
    "CollectionSiteAdapter",
    "IncruitAdapter",
    "WantedAdapter",
    "JobKoreaAdapter",
    "RocketpunchAdapter",
    "ADAPTERS",
    "resolve_adapter",
    "resolve_collection_adapter",
]

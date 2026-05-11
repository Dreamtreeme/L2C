"""사이트별 어댑터 패키지.

각 어댑터는 SiteAdapter ABC를 구현하며,
URL 매칭과 페이지에서 본문을 추출하는 책임만 가집니다.
브라우저 lifecycle은 capture.py가 관리합니다.
"""

from .base import SiteAdapter
from .wanted import WantedAdapter
from .jobkorea import JobKoreaAdapter
from .saramin import SaraminAdapter
from .worknet import WorknetAdapter
from .rocketpunch import RocketpunchAdapter

# 디스패처가 순서대로 matches()를 호출하므로,
# 더 구체적인 어댑터를 위에 두는 게 안전합니다.
ADAPTERS: list[SiteAdapter] = [
    WantedAdapter(),
    JobKoreaAdapter(),
    SaraminAdapter(),
    WorknetAdapter(),
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


__all__ = [
    "SiteAdapter",
    "WantedAdapter",
    "JobKoreaAdapter",
    "SaraminAdapter",
    "WorknetAdapter",
    "RocketpunchAdapter",
    "ADAPTERS",
    "resolve_adapter",
]

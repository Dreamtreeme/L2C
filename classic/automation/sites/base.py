"""사이트 어댑터 베이스 인터페이스.

각 사이트별 어댑터는 이 ABC를 상속하여 두 가지를 구현합니다:
  - matches(url): 이 어댑터가 처리할 URL인지 판별
  - extract(page): 이미 URL에 진입한 Playwright Page에서 본문 추출

브라우저 생성·종료, page.goto는 caller(capture.py)가 처리합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class DomExtraction(TypedDict):
    """상세 페이지 어댑터가 공통 정제 단계에 넘기는 원문."""

    company_name: str | None
    position: str | None
    full_text: str | None


class SiteAdapter(ABC):
    """사이트별 추출 로직을 캡슐화하는 베이스 클래스."""

    #: 사이트 식별자 (로깅·디버깅용). 짧은 영문 슬러그.
    name: str = ""

    @abstractmethod
    def matches(self, url: str) -> bool:
        """이 어댑터가 해당 URL을 담당하는지."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, page: Any) -> DomExtraction:
        """페이지에서 채용공고 정보를 추출한다.

        반환 딕셔너리 스키마(현행 호환):
            {
              "company_name": str | None,
              "position":     str | None,
              "full_text":    str | None,   # LLM에 던질 본문 텍스트
            }

        company_name, position는 LLM이 본문에서 다시 추출하므로 None이어도 OK.
        full_text는 비어 있으면 LLM 단계에서 의미 있는 결과가 나오지 않으니,
        가능한 한 본문을 채워야 합니다.
        """
        raise NotImplementedError


class CollectionSiteAdapter(SiteAdapter):
    """홈페이지에서 검색해 상세 URL을 찾을 수 있는 Classic 어댑터."""

    @abstractmethod
    def submit_search(self, page: Any, keyword: str) -> None:
        """현재 페이지의 검색 UI로 검색어를 제출한다."""
        raise NotImplementedError

    @abstractmethod
    def list_detail_urls(self, page: Any, limit: int | None) -> list[str]:
        """검색 결과에서 상세 공고 URL을 중복 없이 반환한다."""
        raise NotImplementedError


def get_inner_text_safe(locator) -> str | None:
    """Locator에서 안전하게 inner_text를 얻는다. 보이지 않거나 실패하면 None."""
    try:
        if locator.is_visible():
            return locator.inner_text().strip()
    except Exception:
        return None
    return None

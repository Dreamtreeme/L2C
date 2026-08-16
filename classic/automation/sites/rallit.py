"""랠릿(Rallit) 홈페이지 검색과 상세 공고 DOM 추출 어댑터."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urljoin, urlsplit

from .base import CollectionSiteAdapter, DomExtraction, get_inner_text_safe


SEARCH_LABEL = "채용 공고 탐색 메인 검색창"
SEARCH_BUTTON = "검색"
MAIN_TASKS_HEADING = "주요업무"
REQUIREMENTS_HEADING = "자격요건"


def _query_keyword(url: str) -> str:
    values = parse_qs(urlsplit(url).query).get("keyword") or []
    return str(values[0]).strip() if values else ""


def _nearest_section_text(heading: Any) -> str:
    if not heading.is_visible():
        return ""
    text = heading.evaluate("(node) => node.closest('section')?.innerText || ''")
    return str(text or "").strip()


class RallitAdapter(CollectionSiteAdapter):
    """랠릿 검색 결과를 따라가 필수 채용공고 본문을 추출한다."""

    name = "rallit"

    def matches(self, url: str) -> bool:
        host = (urlsplit(url).hostname or "").casefold()
        return host in {"rallit.com", "www.rallit.com"}

    def submit_search(self, page: Any, keyword: str) -> None:
        normalized = str(keyword or "").strip()
        if not normalized:
            raise ValueError("랠릿 검색어는 비어 있을 수 없습니다.")

        page.get_by_label(SEARCH_LABEL, exact=True).fill(normalized)
        page.get_by_role("button", name=SEARCH_BUTTON, exact=True).click()
        page.wait_for_url(lambda url: _query_keyword(url) == normalized)
        page.wait_for_load_state("domcontentloaded")

    def list_detail_urls(self, page: Any, limit: int | None) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for link in page.get_by_role("link").all():
            if not link.is_visible():
                continue
            href = str(link.get_attribute("href") or "").strip()
            candidate = urljoin(page.url, href)
            parts = urlsplit(candidate)
            segments = [segment for segment in parts.path.split("/") if segment]
            if (
                not self.matches(candidate)
                or len(segments) < 2
                or segments[0] != "positions"
                or candidate in seen
            ):
                continue
            seen.add(candidate)
            urls.append(candidate)
            if limit is not None and len(urls) >= limit:
                break
        return urls

    def extract(self, page: Any) -> DomExtraction:
        position_heading = page.get_by_role("heading", level=1).first
        position_heading.wait_for(state="visible")
        header = position_heading.locator("..")
        company_heading = header.get_by_role("heading", level=2).first

        company_name = get_inner_text_safe(company_heading)
        position = get_inner_text_safe(position_heading)
        main_tasks = _nearest_section_text(
            page.get_by_role(
                "heading",
                name=MAIN_TASKS_HEADING,
                exact=True,
            ).first
        )
        requirements = _nearest_section_text(
            page.get_by_role(
                "heading",
                name=REQUIREMENTS_HEADING,
                exact=True,
            ).first
        )
        full_text = "\n\n".join(
            part
            for part in (
                f"회사명\n{company_name}" if company_name else "",
                f"직무명\n{position}" if position else "",
                main_tasks,
                requirements,
            )
            if part
        )
        return {
            "company_name": company_name,
            "position": position,
            "full_text": full_text,
            "identity_authoritative": True,
        }


__all__ = ["RallitAdapter"]

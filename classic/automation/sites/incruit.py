"""인크루트 Classic 검색·상세 DOM 어댑터."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import urljoin, urlsplit

from .base import CollectionSiteAdapter, DomExtraction


INCRUIT_HOMEPAGE = "https://job.incruit.com/"
_DETAIL_PATH = "/jobdb_info/jobpost.asp"
_CONTENT_FRAME_PATH = "/s_common/jobpost/jobpostcont.asp"


def _is_incruit_host(hostname: str | None) -> bool:
    host = str(hostname or "").strip().casefold().rstrip(".")
    return host == "incruit.com" or host.endswith(".incruit.com")


def is_incruit_detail_url(url: str) -> bool:
    """인크루트의 채용공고 상세 URL인지 판별한다."""

    parsed = urlsplit(str(url or "").strip())
    return bool(
        _is_incruit_host(parsed.hostname)
        and parsed.path.casefold() == _DETAIL_PATH
    )


def _job_posting_nodes(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _job_posting_nodes(item)
        return
    if not isinstance(value, Mapping):
        return
    node_type = value.get("@type")
    types = node_type if isinstance(node_type, list) else [node_type]
    if any(str(item or "").casefold() == "jobposting" for item in types):
        yield value
    graph = value.get("@graph")
    if isinstance(graph, (list, Mapping)):
        yield from _job_posting_nodes(graph)


def parse_job_posting_metadata(
    documents: list[str],
) -> tuple[str | None, str | None]:
    """JSON-LD 문서에서 회사명과 공고 제목을 구조적으로 읽는다."""

    company_name: str | None = None
    position: str | None = None
    for raw_document in documents:
        try:
            value = json.loads(raw_document)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for node in _job_posting_nodes(value):
            title = str(node.get("title") or "").strip()
            organization = node.get("hiringOrganization")
            organization_name = (
                str(organization.get("name") or "").strip()
                if isinstance(organization, Mapping)
                else ""
            )
            position = position or title or None
            company_name = company_name or organization_name or None
            if company_name and position:
                return company_name, position
    return company_name, position


def _normalize_text(value: str | None) -> str:
    return "\n".join(
        line for line in (line.strip() for line in str(value or "").splitlines()) if line
    )


class IncruitAdapter(CollectionSiteAdapter):
    """공식 홈페이지 검색부터 공고 상세 본문까지 처리한다."""

    name = "incruit"

    def matches(self, url: str) -> bool:
        return _is_incruit_host(urlsplit(str(url or "").strip()).hostname)

    def submit_search(self, page: Any, keyword: str) -> None:
        search_box = page.get_by_role(
            "textbox",
            name="기업명, 포지션을 검색해 보세요.",
        )
        search_box.fill(keyword)
        page.get_by_role("button", name="검색", exact=True).click()
        page.wait_for_load_state("domcontentloaded")

    def list_detail_urls(self, page: Any, limit: int | None) -> list[str]:
        links = page.get_by_role("link")
        detail_urls: list[str] = []
        seen: set[str] = set()
        for index in range(links.count()):
            href = str(links.nth(index).get_attribute("href") or "").strip()
            if not href:
                continue
            detail_url = urljoin(str(page.url), href)
            if not is_incruit_detail_url(detail_url) or detail_url in seen:
                continue
            seen.add(detail_url)
            detail_urls.append(detail_url)
            if limit is not None and len(detail_urls) >= limit:
                break
        return detail_urls

    def extract(self, page: Any) -> DomExtraction:
        structured_data = page.locator('script[type="application/ld+json"]')
        documents = [
            str(structured_data.nth(index).text_content() or "")
            for index in range(structured_data.count())
        ]
        company_name, position = parse_job_posting_metadata(documents)

        frame_selector = f'iframe[src*="{_CONTENT_FRAME_PATH}"]'
        body = page.frame_locator(frame_selector).locator("body")
        body.wait_for(state="visible")
        detail_text = _normalize_text(body.inner_text())
        if not detail_text:
            raise ValueError("인크루트 상세 공고 설명 DOM이 비어 있습니다.")

        identity = [
            value
            for value in (
                f"회사명: {company_name}" if company_name else "",
                f"공고 제목: {position}" if position else "",
            )
            if value
        ]
        full_text = "\n".join([*identity, "", detail_text]).strip()
        return {
            "company_name": company_name,
            "position": position,
            "full_text": full_text,
        }


__all__ = [
    "INCRUIT_HOMEPAGE",
    "IncruitAdapter",
    "is_incruit_detail_url",
    "parse_job_posting_metadata",
]

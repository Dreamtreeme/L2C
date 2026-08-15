"""신규 사이트 적용 공수 비교 전에 사용하는 합성 채용 사이트."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator
from urllib.parse import urljoin, urlsplit

from classic.automation.sites.base import (
    CollectionSiteAdapter,
    DomExtraction,
    get_inner_text_safe,
)
from classic.extractor.normalization import DomJobNormalizer
from shared.schema.jd_schema import (
    CollectedJob,
    JobCollectionEvidence,
    JobField,
    JobPosting,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synthetic_job_site"


class _SilentHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return None


@contextmanager
def serve_synthetic_job_site() -> Iterator[str]:
    """합성 사이트를 임의의 로컬 포트에서 제공한다."""

    handler = partial(_SilentHandler, directory=str(FIXTURE_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/index.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class SyntheticCollectionAdapter(CollectionSiteAdapter):
    """합성 사이트의 DOM 결합 지점을 명시하는 벤치마크 어댑터."""

    name = "synthetic"

    def matches(self, url: str) -> bool:
        return (urlsplit(url).hostname or "") in {"127.0.0.1", "localhost"}

    def submit_search(self, page, keyword: str) -> None:
        page.get_by_label("채용공고 검색", exact=True).fill(keyword)
        page.get_by_role("button", name="검색", exact=True).click()
        page.get_by_role("status").wait_for(state="visible")

    def list_detail_urls(self, page, limit: int | None) -> list[str]:
        urls: list[str] = []
        for link in page.get_by_role("link").all():
            if not link.is_visible():
                continue
            href = str(link.get_attribute("href") or "").strip()
            if href:
                urls.append(urljoin(page.url, href))
            if limit is not None and len(urls) >= limit:
                break
        return urls

    def extract(self, page) -> DomExtraction:
        return {
            "company_name": get_inner_text_safe(
                page.get_by_role("heading", level=2).first
            ),
            "position": get_inner_text_safe(page.get_by_role("heading", level=1).first),
            "full_text": get_inner_text_safe(page.locator("main").first),
        }


class SyntheticJobNormalizer(DomJobNormalizer):
    """외부 LLM 없이 공통 실행 경로만 검증하는 합성 정제기."""

    def normalize(
        self,
        extraction: DomExtraction,
        *,
        url: str,
        source_platform: str,
        required_fields: list[JobField],
    ) -> CollectedJob:
        full_text = str(extraction.get("full_text") or "").strip()
        posting = JobPosting(
            company_name=extraction.get("company_name"),
            position=extraction.get("position"),
            url=url,
            main_tasks=[full_text],
            requirements=[full_text],
            source_platform=source_platform,
            raw_ocr_text=full_text,
        )
        return CollectedJob(
            posting=posting,
            evidence=JobCollectionEvidence(required_fields=required_fields),
        )


__all__ = [
    "SyntheticCollectionAdapter",
    "SyntheticJobNormalizer",
    "serve_synthetic_job_site",
]

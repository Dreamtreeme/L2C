"""Classic 홈페이지 검색부터 상세 공고 저장까지의 공통 실행기."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.config import get_settings
from agent.utils.job_fields import required_job_fields
from shared.db.database import Database
from shared.schema.collection_intent import CollectionIntent, CollectionResult

from classic.automation.browser import open_browser_page
from classic.automation.sites import resolve_collection_adapter
from classic.automation.sites.base import CollectionSiteAdapter
from classic.extractor.normalization import DomJobNormalizer, LLMDomJobNormalizer

logger = logging.getLogger(__name__)


class ClassicCollectionRunner:
    """사이트 어댑터를 공통 검색·정제·저장 순서로 실행한다."""

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        normalizer: DomJobNormalizer | None = None,
    ) -> None:
        self.db = Database(db_path or get_settings().paths.db_path)
        self.normalizer = normalizer or LLMDomJobNormalizer()

    def run(
        self,
        homepage: str,
        intent: CollectionIntent,
        *,
        adapter: CollectionSiteAdapter | None = None,
    ) -> CollectionResult:
        """브라우저를 열어 홈페이지부터 수집 계약을 수행한다."""

        selected = adapter or resolve_collection_adapter(homepage)
        if not selected.matches(homepage):
            raise ValueError(
                f"{selected.name} 어댑터가 홈페이지를 처리할 수 없습니다: {homepage}"
            )
        browser_settings = get_settings().browser
        with open_browser_page() as page:
            page.goto(
                homepage,
                wait_until="domcontentloaded",
                timeout=browser_settings.playwright_timeout_ms,
            )
            return self.run_on_page(page, intent, adapter=selected)

    def run_on_page(
        self,
        page: Any,
        intent: CollectionIntent,
        *,
        adapter: CollectionSiteAdapter,
    ) -> CollectionResult:
        """열린 페이지에서 검색하고 발견한 상세 URL을 순서대로 처리한다."""

        adapter.submit_search(page, intent.search_keyword)
        required_fields = required_job_fields(intent)
        requested_count = intent.target_count or None
        detail_urls = list(
            dict.fromkeys(adapter.list_detail_urls(page, requested_count))
        )
        if requested_count is not None:
            detail_urls = detail_urls[:requested_count]

        persisted_items: list[dict[str, Any]] = []
        document_ids: list[int] = []
        rejected_count = 0
        for url in detail_urls:
            try:
                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=get_settings().browser.playwright_timeout_ms,
                )
                resolved_url = str(page.url or url)
                collected_job = self.normalizer.normalize(
                    adapter.extract(page),
                    url=resolved_url,
                    source_platform=adapter.name,
                    required_fields=required_fields,
                )
                existed = self.db.exists(resolved_url)
                job_id = int(
                    self.db.upsert(
                        collected_job.posting,
                        evidence=collected_job.evidence,
                    )
                )
                document_ids.append(job_id)
                persisted_items.append(
                    {
                        "job_id": job_id,
                        "url": str(collected_job.posting.url or url),
                        "company_name": collected_job.posting.company_name or "",
                        "position": collected_job.posting.position or "",
                        "operation": "updated" if existed else "created",
                    }
                )
            except Exception as exc:
                rejected_count += 1
                logger.exception("Classic 상세 공고 처리 실패 url=%s: %s", url, exc)

        target_count = intent.target_count or len(detail_urls)
        persisted_count = len(persisted_items)
        status = (
            "completed"
            if target_count > 0 and persisted_count >= target_count
            else "partial"
            if persisted_count
            else "failed"
        )
        return CollectionResult(
            status=status,
            message=(
                f"Classic 수집 {persisted_count}/{target_count}건 완료"
                if target_count
                else "검색 결과가 없습니다."
            ),
            error_code="" if status == "completed" else "collection_incomplete",
            site=intent.site or adapter.name,
            site_name=adapter.name,
            search_keyword=intent.search_keyword,
            task_category=intent.task_category,
            target_count=target_count,
            collected_count=persisted_count,
            resolved_count=persisted_count,
            persisted_count=persisted_count,
            created_count=sum(
                item["operation"] == "created" for item in persisted_items
            ),
            updated_count=sum(
                item["operation"] == "updated" for item in persisted_items
            ),
            rejected_count=rejected_count,
            persisted_items=persisted_items,
            document_ids=document_ids,
            scope_exhausted=(
                requested_count is not None and len(detail_urls) < requested_count
            ),
            worker_finished=True,
            worker_run_id=f"classic-{uuid4()}",
        )


__all__ = ["ClassicCollectionRunner"]

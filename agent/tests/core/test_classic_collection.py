from classic.automation.collection import ClassicCollectionRunner
from classic.automation.sites.base import CollectionSiteAdapter
from shared.schema.collection_intent import CollectionIntent
from shared.schema.jd_schema import (
    CollectedJob,
    JobCollectionEvidence,
    JobPosting,
)


class _Page:
    def __init__(self) -> None:
        self.url = "https://example.com"

    def goto(self, url, **_kwargs):
        self.url = url


class _Adapter(CollectionSiteAdapter):
    name = "example"

    def matches(self, url):
        return url.startswith("https://example.com")

    def submit_search(self, page, keyword):
        self.keyword = keyword

    def list_detail_urls(self, page, limit):
        return [
            "https://example.com/jobs/1",
            "https://example.com/jobs/2",
        ][:limit]

    def extract(self, page):
        job_id = page.url.rsplit("/", 1)[-1]
        return {
            "company_name": f"회사 {job_id}",
            "position": "백엔드 개발자",
            "full_text": f"공고 본문 {job_id}",
        }


class _Normalizer:
    def normalize(
        self,
        extraction,
        *,
        url,
        source_platform,
        required_fields,
    ):
        posting = JobPosting(
            company_name=extraction["company_name"],
            position=extraction["position"],
            url=url,
            main_tasks=[extraction["full_text"]],
            requirements=[extraction["full_text"]],
            source_platform=source_platform,
        )
        return CollectedJob(
            posting=posting,
            evidence=JobCollectionEvidence(required_fields=required_fields),
        )


def test_classic_collection_runner_searches_normalizes_and_persists(tmp_path):
    adapter = _Adapter()
    runner = ClassicCollectionRunner(
        db_path=tmp_path / "classic.db",
        normalizer=_Normalizer(),
    )
    result = runner.run_on_page(
        _Page(),
        CollectionIntent(
            site="example",
            search_keyword="백엔드 개발자",
            target_count=2,
        ),
        adapter=adapter,
    )

    assert adapter.keyword == "백엔드 개발자"
    assert result.status == "completed"
    assert result.persisted_count == 2
    assert len(runner.db.load_jobs(result.document_ids)) == 2

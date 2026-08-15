import json

from classic.automation.sites import resolve_collection_adapter
from classic.automation.sites.incruit import (
    IncruitAdapter,
    is_incruit_detail_url,
    parse_job_posting_metadata,
)


def test_incruit_adapter_matches_official_subdomains_only():
    adapter = IncruitAdapter()

    assert adapter.matches("https://job.incruit.com/") is True
    assert adapter.matches("https://search.incruit.com/list/search.asp") is True
    assert adapter.matches("https://incruit.com.example.test/") is False


def test_incruit_detail_url_uses_domain_and_path_not_posting_id():
    assert (
        is_incruit_detail_url(
            "https://job.incruit.com/jobdb_info/jobpost.asp?job=example"
        )
        is True
    )
    assert (
        is_incruit_detail_url(
            "https://job.incruit.com/jobdb_info/another.asp?job=example"
        )
        is False
    )
    assert (
        is_incruit_detail_url(
            "https://incruit.com.example.test/jobdb_info/jobpost.asp?job=example"
        )
        is False
    )


def test_incruit_json_ld_metadata_uses_job_posting_schema():
    document = json.dumps(
        {
            "@context": "https://schema.org",
            "@graph": [
                {"@type": "Organization", "name": "무관한 조직"},
                {
                    "@type": "JobPosting",
                    "title": "동적 테스트 직무",
                    "hiringOrganization": {
                        "@type": "Organization",
                        "name": "동적 테스트 회사",
                    },
                },
            ],
        },
        ensure_ascii=False,
    )

    assert parse_job_posting_metadata(["invalid", document]) == (
        "동적 테스트 회사",
        "동적 테스트 직무",
    )


def test_incruit_adapter_is_registered_for_official_homepage():
    adapter = resolve_collection_adapter("https://job.incruit.com/")

    assert isinstance(adapter, IncruitAdapter)

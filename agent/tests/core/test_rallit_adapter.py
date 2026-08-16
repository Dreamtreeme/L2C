from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import sync_playwright

from classic.automation.sites.rallit import RallitAdapter


SEARCH_PAGE = """
<!doctype html>
<html lang="ko">
  <body>
    <main>
      <input aria-label="채용 공고 탐색 메인 검색창" />
      <button
        type="button"
        onclick="location.href='/?keyword=' + encodeURIComponent(document.querySelector('input').value)"
      >검색</button>
      <a href="/positions/dynamic-a/first">동적 공고 A</a>
      <a href="/positions/dynamic-b/second">동적 공고 B</a>
      <a href="/positions/dynamic-a/first">중복 공고</a>
      <a href="https://example.com/positions/external">외부 공고</a>
      <a href="/companies/example">회사</a>
      <a href="/positions/hidden" style="display:none">숨김 공고</a>
    </main>
  </body>
</html>
"""

DETAIL_PAGE = """
<!doctype html>
<html lang="ko">
  <body>
    <main>
      <header>
        <h2>테스트 회사</h2>
        <h1>서버 개발자</h1>
      </header>
      <section>
        <div><h3>주요업무</h3></div>
        <div>API를 설계하고 운영합니다.</div>
      </section>
      <section>
        <div><h3>자격요건</h3></div>
        <div>Python 개발 경험이 필요합니다.</div>
      </section>
      <section>
        <h2>비슷한 채용 공고</h2>
        <p>현재 공고와 무관한 텍스트</p>
      </section>
    </main>
  </body>
</html>
"""


def test_rallit_adapter_searches_lists_dynamic_urls_and_extracts_detail():
    adapter = RallitAdapter()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.route(
            "https://www.rallit.com/**",
            lambda route: route.fulfill(
                status=200,
                content_type="text/html; charset=utf-8",
                body=SEARCH_PAGE,
            ),
        )
        page.goto("https://www.rallit.com/")

        adapter.submit_search(page, "동적 검색어")
        assert parse_qs(urlsplit(page.url).query)["keyword"] == ["동적 검색어"]
        assert adapter.list_detail_urls(page, 2) == [
            "https://www.rallit.com/positions/dynamic-a/first",
            "https://www.rallit.com/positions/dynamic-b/second",
        ]

        page.set_content(DETAIL_PAGE)
        extraction = adapter.extract(page)
        browser.close()

    assert extraction["company_name"] == "테스트 회사"
    assert extraction["position"] == "서버 개발자"
    assert "API를 설계하고 운영합니다." in str(extraction["full_text"])
    assert "Python 개발 경험이 필요합니다." in str(extraction["full_text"])
    assert "현재 공고와 무관한 텍스트" not in str(extraction["full_text"])


def test_rallit_adapter_matches_only_official_job_site_hosts():
    adapter = RallitAdapter()

    assert adapter.matches("https://www.rallit.com/")
    assert adapter.matches("https://rallit.com/positions/example")
    assert not adapter.matches("https://business.rallit.com/")
    assert not adapter.matches("https://example.com/rallit.com")

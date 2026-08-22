def test_site_registry_loads_supported_navigation_profiles():
    from agent.sites import list_supported_sites, load_site_profile

    profiles = list_supported_sites()
    slugs = {profile.slug for profile in profiles}

    assert {
        "wanted",
        "jobkorea",
        "saramin",
        "worknet",
        "rocketpunch",
        "incruit",
    }.issubset(slugs)
    for profile in profiles:
        loaded = load_site_profile(profile.slug)
        for role in ("home", "search", "job_detail"):
            assert loaded.page_guidance[role].instructions, (profile.slug, role)
        assert loaded.page_guidance["job_detail"].reading_targets, profile.slug

    worknet_detail = load_site_profile("worknet").page_guidance["job_detail"]
    assert worknet_detail.reveal_controls == ("더보기",)


def test_official_site_urls_resolve_from_slug_name_and_domain():
    from agent.sites import get_official_site_url

    expected = {
        "wanted": "https://www.wanted.co.kr",
        "잡코리아": "https://www.jobkorea.co.kr",
        "https://www.saramin.co.kr": "https://www.saramin.co.kr",
        "고용24": "https://www.work24.go.kr",
        "rocketpunch.com": "https://www.rocketpunch.com",
        "인크루트": "https://job.incruit.com",
    }

    for requested_site, official_url in expected.items():
        assert get_official_site_url(requested_site) == official_url


def test_worker_preparation_opens_requested_site_instead_of_default(monkeypatch):
    from agent.application.worker_execution_service import prepare_worker_start_screen
    from agent.sites import load_site_profile
    from agent.tests.worker_test_support import worker_state

    calls = []
    warmed = []
    reasoning_warmed = []

    class FakeActionTools:
        def open_browser(self, url="", current_url="", site=""):
            calls.append({"url": url, "current_url": current_url, "site": site})
            return {
                "status": "success",
                "result": {"url": "https://www.jobkorea.co.kr"},
            }

    class FakeRuntime:
        def get_action_tools(self):
            return FakeActionTools()

        def ensure_ocr_worker_ready(self):
            warmed.append(True)

        def prepare_reasoning_models(self, _tool_schemas):
            reasoning_warmed.append(True)

    result = prepare_worker_start_screen(
        worker_state(),
        load_site_profile("잡코리아"),
        worker_runtime=FakeRuntime(),
    )

    assert calls == [{"url": "", "current_url": "", "site": "jobkorea"}]
    assert warmed == [True]
    assert reasoning_warmed == [True]
    assert result["observation"]["current_url"] == "https://www.jobkorea.co.kr"


def test_site_page_roles_use_declared_url_signals():
    from agent.runtime.site_context import (
        infer_site_page_role,
        looks_like_job_detail_url,
    )

    url_cases = [
        (
            "https://www.jobkorea.co.kr/Recruit/GI_Read/50000001",
            "job_detail",
        ),
        (
            "https://www.work24.go.kr/wk/a/b/1500/empDetailAuthView.do"
            "?wantedAuthNo=51078967&infoTypeCd=CJK",
            "job_detail",
        ),
        (
            "https://job.incruit.com/jobdb_info/jobpost.asp?job=2608040000080",
            "job_detail",
        ),
        ("https://www.work24.go.kr/cm/main.do", "home"),
        ("https://www.saramin.co.kr/zf_user/", "home"),
    ]

    for url, expected_role in url_cases:
        assert infer_site_page_role(url, []) == expected_role, url
        if expected_role == "job_detail":
            assert looks_like_job_detail_url(url) is True, url

    assert (
        infer_site_page_role(
            "https://www.wanted.co.kr/",
            ["검색 결과", "iOS 개발자 · 직무"],
        )
        == "search_overlay"
    )


def test_unregistered_site_does_not_use_generic_job_url_heuristic():
    from agent.runtime.site_context import infer_site_page_role

    assert infer_site_page_role("https://example.com/job/123", ["추천 검색어"]) == ""


def test_detail_context_does_not_require_a_detail_url_pattern():
    from agent.runtime.site_context import is_job_detail_context

    assert (
        is_job_detail_context(
            "https://www.rocketpunch.com/jobs",
            page_role="side_panel_detail",
        )
        is True
    )
    assert (
        is_job_detail_context(
            "https://www.work24.go.kr/search",
            marker_texts=["모집요강", "직무내용", "근무조건"],
        )
        is True
    )


def test_rocketpunch_list_and_selected_job_use_side_panel_contract():
    from agent.runtime.site_context import (
        infer_site_page_role,
        looks_like_job_detail_url,
        site_runtime_guidance,
    )

    url = "https://www.rocketpunch.com/jobs"
    list_markers = ["키워드", "직군", "숙련도", "기업 규모", "근무 방식"]

    home_guidance = site_runtime_guidance("https://www.rocketpunch.com", "home")
    assert "사이드바의 '채용' 메뉴" in home_guidance
    assert infer_site_page_role(url, list_markers) == "search"
    guidance = site_runtime_guidance(url, "search")
    assert "페이지 중앙 채용 검색 영역" in guidance
    assert "왼쪽 사이드바" in guidance
    assert "공고 카드를 큐에 넣기 전에 상단 필터" in guidance
    assert "'직군'에서 'SW 개발'" in guidance
    assert "넓은 상위 직군" in guidance
    assert "다른 직무를 대신 선택하지" in guidance
    assert "scroll.marker_id" in guidance
    assert (
        infer_site_page_role(
            url,
            list_markers + ["주요업무", "자격요건", "채용 상세"],
        )
        == "job_detail"
    )
    selected_url = (
        "https://www.rocketpunch.com/jobs?keyword=백엔드+개발자&selectedJobId=159079"
    )
    assert looks_like_job_detail_url(selected_url) is True
    assert infer_site_page_role(selected_url, ["주요업무"]) == "job_detail"
    detail_guidance = site_runtime_guidance(selected_url, "job_detail")
    assert "공고 카드 큐가 아직 비어 있으면" in detail_guidance
    assert "상단 '직군' 필터" in detail_guidance


def test_search_reasoning_receives_current_site_guidance(monkeypatch):
    from agent.graph import worker_reasoning_prompt
    from agent.tests.worker_test_support import worker_state
    from shared.schema.collection_intent import CollectionIntent

    expected_guidance = "SITE_GUIDANCE_SENTINEL"
    monkeypatch.setattr(
        worker_reasoning_prompt,
        "site_runtime_guidance",
        lambda url, role: (
            expected_guidance
            if url == "https://www.rocketpunch.com/jobs" and role == "search"
            else ""
        ),
    )

    messages = worker_reasoning_prompt.build_reasoning_messages(
        worker_state(
            request={
                "collection_intent": CollectionIntent(
                    search_keyword="백엔드 개발자",
                    target_count=1,
                ),
            },
            observation={
                "current_url": "https://www.rocketpunch.com/jobs",
                "current_page_role": "search",
                "current_markers": [{"id": 1, "type": "text", "text": "키워드"}],
            },
        ),
        "",
    )

    assert expected_guidance in str(messages[1].content)
    assert "apply required query or filters and verify the update" in str(
        messages[0].content
    )
    assert "필터를 적용하고 결과 화면이 갱신됐는지 확인" in str(messages[1].content)
    assert "그 전에는 set_job_card_queue를 호출하지 마십시오" in str(
        messages[1].content
    )


def test_runtime_guidance_contains_only_current_site_and_role():
    from agent.runtime.site_context import site_runtime_guidance

    guidance = site_runtime_guidance(
        "https://www.jobkorea.co.kr/Recruit/GI_Read/50000001",
        "job_detail",
    )

    assert "잡코리아 / 추정 화면 역할 job_detail" in guidance
    assert "실제 캡처와 마커를 우선" in guidance
    assert "급여" in guidance
    assert "미방문 카드 큐" not in guidance
    assert "원티드" not in guidance


def test_realtime_scraping_goal_uses_requested_site_profile():
    from agent.application.collection_request_builder import build_site_goal
    from agent.sites import load_site_profile
    from shared.schema.collection_intent import CollectionIntent

    goal = build_site_goal(
        CollectionIntent(site="saramin", search_keyword="AI 엔지니어"),
        load_site_profile("saramin"),
    )

    assert "사람인(" in goal
    assert "https://www.saramin.co.kr" in goal
    assert "AI 엔지니어" in goal
    assert "원티드(" not in goal


def test_site_goal_injects_selected_skill_without_duplicate_profile_sections():
    from agent.application.collection_request_builder import build_site_goal
    from agent.sites import load_site_profile
    from shared.schema.collection_intent import CollectionIntent

    goal = build_site_goal(
        CollectionIntent(
            site="wanted",
            search_keyword="AI 엔지니어",
            count_mode="visible_all",
        ),
        load_site_profile("wanted"),
    )

    assert "[선택된 사이트 스킬]" in goal
    assert "원티드 채용공고 수집" in goal
    assert "[사이트 공통 흐름]" not in goal
    assert "[허용 도구]" not in goal


def test_realtime_scraping_wanted_starts_from_home_without_query_url():
    from agent.application.collection_request_builder import build_site_goal
    from agent.sites import load_site_profile
    from shared.schema.collection_intent import CollectionIntent

    profile = load_site_profile("wanted")
    keyword = "android \uac1c\ubc1c\uc790"
    goal = build_site_goal(
        CollectionIntent(site="wanted", search_keyword=keyword),
        profile,
    )

    assert (
        "Open only the site home page with open_browser: https://www.wanted.co.kr"
        in goal
    )
    assert "Do not construct or open a search/query URL yourself" in goal


def test_worker_receives_structured_collection_intent():
    from agent.application.worker_execution_service import (
        _build_initial_state,
        _resolve_collection_intent,
    )
    from agent.sites import load_site_profile
    from shared.schema.collection_intent import CollectionIntent

    profile = load_site_profile("wanted")
    intent = _resolve_collection_intent(
        CollectionIntent(
            original_query="7월 서울 iOS 개발자 공고 2개",
            site="wanted",
            search_keyword="iOS 개발자",
            target_count=2,
            filters={"posted_from": "2026-07-01", "location": "서울"},
            freshness_required=True,
            task_category="탐색",
            required_fields=["posted_at"],
        ),
        profile,
    )
    state = _build_initial_state(intent, profile, "worker-test")

    assert intent.target_count == 2
    assert intent.task_category == "탐색"
    assert state["request"]["collection_intent"] == intent
    assert intent.search_keyword == "iOS 개발자"
    assert intent.task_category == "탐색"
    assert [field.value for field in intent.required_fields] == [
        "company_name",
        "position",
        "url",
        "main_tasks",
        "requirements",
        "posted_at",
    ]
    assert "required_record_shape" in state["request"]["goal"]
    assert "Collect up to 2 distinct job postings" in state["request"]["goal"]
    assert intent.filters.model_dump(mode="json") == {
        "posted_from": "2026-07-01",
        "posted_to": "",
        "experience": "",
        "location": "서울",
        "employment_type": "",
    }
    assert "posted_at" in [field.value for field in intent.required_fields]

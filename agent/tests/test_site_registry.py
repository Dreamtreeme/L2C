def test_site_registry_lists_existing_profiles():
    from agent.sites import list_supported_sites

    sites = list_supported_sites()
    slugs = {site.slug for site in sites}

    assert {"wanted", "jobkorea", "saramin", "worknet", "rocketpunch"}.issubset(slugs)


def test_official_site_urls_resolve_from_slug_name_and_domain():
    from agent.sites import get_official_site_url

    expected = {
        "wanted": "https://www.wanted.co.kr",
        "잡코리아": "https://www.jobkorea.co.kr",
        "https://www.saramin.co.kr": "https://www.saramin.co.kr",
        "고용24": "https://www.work24.go.kr",
        "rocketpunch.com": "https://www.rocketpunch.com",
    }

    for requested_site, official_url in expected.items():
        assert get_official_site_url(requested_site) == official_url


def test_open_browser_uses_requested_site_official_url(monkeypatch):
    from agent.tools.actions import ActionTools

    class FakePerception:
        _browser_window_id = None

    action_tools = object.__new__(ActionTools)
    action_tools.perception = FakePerception()
    opened = []
    monkeypatch.setattr(action_tools, "_bound_browser_window_exists", lambda: False)
    monkeypatch.setattr(
        action_tools,
        "_open_url_in_new_window",
        lambda url: opened.append(url) or {"opened": True, "url": url},
    )
    monkeypatch.setattr(action_tools, "_reset_browser_zoom", lambda: None)

    result = action_tools.open_browser(site="saramin")

    assert result["status"] == "success"
    assert opened == ["https://www.saramin.co.kr"]


def test_worker_preparation_opens_requested_site_instead_of_default(monkeypatch):
    from agent.application.worker_execution_service import prepare_worker_start_screen
    from agent.graph import worker_resources
    from agent.sites import load_site_profile

    calls = []
    warmed = []
    reasoning_warmed = []

    class FakeSomEngine:
        def ensure_ocr_worker_ready(self):
            warmed.append(True)

    class FakePerception:
        som_engine = FakeSomEngine()

    class FakeActionTools:
        perception = FakePerception()

        def open_browser(self, url="", current_url="", site=""):
            calls.append({"url": url, "current_url": current_url, "site": site})
            return {
                "status": "success",
                "result": {"url": "https://www.jobkorea.co.kr"},
            }

    monkeypatch.setenv("VISION_WORKER_PREOPEN_BROWSER", "1")
    monkeypatch.setattr(worker_resources, "get_action_tools", lambda: FakeActionTools())
    monkeypatch.setattr(
        worker_resources,
        "prepare_reasoning_models",
        lambda: reasoning_warmed.append(True),
    )
    monkeypatch.setattr(
        "agent.graph.worker_observation.capture_node",
        lambda state: {
            "current_url": "https://www.jobkorea.co.kr",
            "current_url_stale": False,
            "recent_images": ["screen.png"],
            "low_information_screen": False,
        },
    )
    monkeypatch.setattr(
        "agent.graph.worker_observation.ocr_node",
        lambda state: {"current_markers": [{"id": 1}]},
    )
    monkeypatch.setattr(
        "agent.graph.worker_transition.transition_node",
        lambda state: {},
    )
    monkeypatch.setattr(
        "agent.graph.worker_collection.collection_node",
        lambda state: {},
    )

    result = prepare_worker_start_screen(
        {"current_url": "", "action_history": []},
        load_site_profile("잡코리아"),
    )

    assert calls == [{"url": "", "current_url": "", "site": "jobkorea"}]
    assert warmed == [True]
    assert reasoning_warmed == [True]
    assert result["current_url"] == "https://www.jobkorea.co.kr"


def test_site_registry_profile_files_exist():
    from agent.sites import list_supported_sites
    from agent.sites.loader import SITES_DIR

    for profile in list_supported_sites(enabled_only=False):
        path = SITES_DIR / profile.slug / "profile.json"
        assert path.exists(), f"missing profile.json for {profile.slug}: {path}"


def test_load_site_profile_returns_typed_contract():
    from agent.sites import load_site_profile

    profile = load_site_profile("wanted.co.kr")

    assert profile.slug == "wanted"
    assert "원티드" in profile.guidance
    assert "click_marker" in profile.tools.allowed_tools
    assert "finish_detail_reading" in profile.tools.allowed_tools
    assert "set_job_card_queue" in profile.tools.allowed_tools
    assert profile.collection_policy.required_fields


def test_all_site_profiles_define_collection_and_tool_contracts():
    from agent.sites import list_supported_sites, load_site_profile

    for profile in list_supported_sites():
        loaded = load_site_profile(profile.slug)
        payload = loaded.model_dump()
        assert loaded.collection_policy.required_fields
        assert loaded.tools.allowed_tools
        assert "reflex_policy" not in payload
        assert "reflex" not in payload["tools"]


def test_all_site_profiles_define_role_scoped_guidance():
    from agent.sites import list_supported_sites, load_site_profile

    for profile in list_supported_sites():
        guidance = load_site_profile(profile.slug).page_guidance
        assert guidance["home"].instructions
        assert guidance["search"].instructions
        assert guidance["job_detail"].instructions
        assert guidance["job_detail"].reading_targets


def test_jobkorea_detail_role_uses_declared_url_signal():
    from agent.runtime.site_context import infer_site_page_role, looks_like_job_detail_url

    url = "https://www.jobkorea.co.kr/Recruit/GI_Read/50000001"

    assert looks_like_job_detail_url(url) is True
    assert infer_site_page_role(url, []) == "job_detail"


def test_work24_detail_role_uses_declared_url_signal():
    from agent.runtime.site_context import infer_site_page_role, looks_like_job_detail_url

    url = (
        "https://www.work24.go.kr/wk/a/b/1500/empDetailAuthView.do"
        "?wantedAuthNo=51078967&infoTypeCd=CJK"
    )

    assert looks_like_job_detail_url(url) is True
    assert infer_site_page_role(url, []) == "job_detail"


def test_work24_redirected_home_uses_declared_url_signal():
    from agent.runtime.site_context import infer_site_page_role

    assert infer_site_page_role("https://www.work24.go.kr/cm/main.do", []) == "home"


def test_saramin_redirected_home_uses_declared_url_signal():
    from agent.runtime.site_context import infer_site_page_role

    assert infer_site_page_role("https://www.saramin.co.kr/zf_user/", []) == "home"


def test_unregistered_site_does_not_use_generic_job_url_heuristic():
    from agent.runtime.site_context import infer_site_page_role

    assert infer_site_page_role("https://example.com/job/123", ["추천 검색어"]) == ""


def test_detail_context_does_not_require_a_detail_url_pattern():
    from agent.runtime.site_context import is_job_detail_context

    assert is_job_detail_context(
        "https://www.rocketpunch.com/jobs",
        page_role="side_panel_detail",
    ) is True
    assert is_job_detail_context(
        "https://www.work24.go.kr/search",
        marker_texts=["모집요강", "직무내용", "근무조건"],
    ) is True


def test_rocketpunch_jobs_list_uses_job_search_guidance_without_hiding_side_panel():
    from agent.runtime.site_context import infer_site_page_role, site_runtime_guidance

    url = "https://www.rocketpunch.com/jobs"
    list_markers = ["키워드", "직군", "숙련도", "기업 규모", "근무 방식"]

    home_guidance = site_runtime_guidance("https://www.rocketpunch.com", "home")
    assert "사이드바의 '채용' 메뉴" in home_guidance
    assert infer_site_page_role(url, list_markers) == "search"
    guidance = site_runtime_guidance(url, "search")
    assert "페이지 중앙 채용 검색 영역" in guidance
    assert "왼쪽 사이드바" in guidance
    assert infer_site_page_role(
        url,
        list_markers + ["주요업무", "자격요건", "채용 상세"],
    ) == "job_detail"


def test_job_card_selector_receives_current_site_guidance(monkeypatch, tmp_path):
    from agent.runtime import job_card_selector

    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"not-used")
    monkeypatch.setattr(
        job_card_selector,
        "image_to_base64_jpeg",
        lambda *_args, **_kwargs: "image",
    )

    messages = job_card_selector._selection_messages(
        {
            "current_url": "https://www.rocketpunch.com/jobs",
            "current_page_role": "search",
            "current_markers": [{"id": 1, "type": "text", "text": "키워드"}],
            "marked_image": str(image_path),
            "recipe_params": {"query": "백엔드 개발자", "target_count": 1},
        },
        1,
    )

    assert "기업명, 직무명, 기술 스택, 담당 업무" in messages[0].content
    assert "관심 주제, 공고, 사람, 기업" in messages[0].content


def test_runtime_guidance_contains_only_current_site_and_role():
    from agent.runtime.site_context import site_runtime_guidance

    guidance = site_runtime_guidance(
        "https://www.jobkorea.co.kr/Recruit/GI_Read/50000001",
        "job_detail",
    )

    assert "잡코리아 / job_detail" in guidance
    assert "급여" in guidance
    assert "미방문 카드 큐" not in guidance
    assert "원티드" not in guidance


def test_preprocessor_uses_registry_source_platform():
    from agent.utils.preprocessor import Preprocessor

    assert (
        Preprocessor.parse_source_platform(
            "https://www.jobkorea.co.kr/Recruit/GI_Read/50000001"
        )
        == "JobKorea"
    )


def test_realtime_scraping_goal_uses_requested_site_profile():
    from agent.sites import load_site_profile
    from agent.tools.realtime_scraping import _build_site_goal

    goal = _build_site_goal("AI 엔지니어", load_site_profile("saramin"))

    assert "사람인(" in goal
    assert "https://www.saramin.co.kr" in goal
    assert "AI 엔지니어" in goal
    assert "원티드(" not in goal

def test_realtime_scraping_goal_includes_task_context():
    from agent.sites import load_site_profile
    from agent.tools.realtime_scraping import _build_site_goal

    goal = _build_site_goal(
        "AI engineer",
        load_site_profile("wanted"),
        task_context={
            "triage": {"goal_type": "job_collection", "risk_level": "safe_navigation"},
            "research_report": {"status": "skipped"},
            "allowed_actions": ["read", "navigate", "search"],
            "blocked_actions": ["submit", "agree"],
        },
    )

    assert "[Task triage and safety context]" in goal
    assert '"goal_type": "job_collection"' in goal
    assert "blocked_actions" in goal


def test_site_goal_injects_selected_skill_without_duplicate_profile_sections():
    from agent.sites import load_site_profile
    from agent.tools.realtime_scraping import _build_site_goal

    goal = _build_site_goal(
        "AI 엔지니어",
        load_site_profile("wanted"),
        collection_intent={
            "site": "wanted",
            "search_keyword": "AI 엔지니어",
            "count_mode": "visible_all",
        },
    )

    assert "[선택된 사이트 스킬]" in goal
    assert "원티드 채용공고 수집" in goal
    assert "[사이트 공통 흐름]" not in goal
    assert "[허용 도구]" not in goal

def test_realtime_scraping_extracts_user_search_intent_with_llm(monkeypatch):
    from agent.sites import load_site_profile
    from agent.tools.realtime_scraping import _extract_search_intent
    from shared.schema.collection_intent import CollectionIntent

    class FakeStructuredLLM:
        def invoke(self, messages):
            return CollectionIntent(search_keyword="ai\uc751\uc6a9\uc5d4\uc9c0\ub2c8\uc5b4", target_count=8)

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, schema):
            return FakeStructuredLLM()

    profile = load_site_profile("wanted")
    query = "\uc6d0\ud2f0\ub4dc\uc5d0\uc11c ai\uc751\uc6a9\uc5d4\uc9c0\ub2c8\uc5b4 \ucc44\uc6a9\uacf5\uace0 \ucc3e\uc544\uc918"
    monkeypatch.setenv("VISION_SEARCH_INTENT_MODE", "llm")
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", FakeLLM)

    intent = _extract_search_intent(query, profile)
    assert intent["search_keyword"] == "ai\uc751\uc6a9\uc5d4\uc9c0\ub2c8\uc5b4"
    assert intent["target_count"] == 8
    assert intent["source"] == "llm"


def test_realtime_scraping_direct_search_url_encodes_keyword():
    from urllib.parse import parse_qs, urlparse

    from agent.tools.realtime_scraping import _build_direct_search_url, _build_site_goal

    from agent.sites import load_site_profile
    from agent.sites.profile import NavigationPolicy

    profile = load_site_profile("wanted").model_copy(
        update={
            "slug": "sample",
            "display_name": "Sample",
            "domains": ("example.com",),
            "base_url": "https://example.com",
            "navigation_policy": NavigationPolicy(
                allow_direct_search_url=True,
                search_url_template="https://example.com/search?query={query}&tab=position",
            ),
        }
    )
    keyword = "AI \uc751\uc6a9 \uc5d4\uc9c0\ub2c8\uc5b4"
    url = _build_direct_search_url(keyword, profile)
    parsed = urlparse(url)

    assert parsed.scheme == "https"
    assert parsed.netloc == "example.com"
    assert parsed.path == "/search"
    assert parse_qs(parsed.query)["query"] == [keyword]
    assert parse_qs(parsed.query)["tab"] == ["position"]
    assert "Code-generated search URL" in _build_site_goal(keyword, profile, url)


def test_realtime_scraping_wanted_starts_from_home_without_query_url():
    from agent.sites import load_site_profile
    from agent.tools.realtime_scraping import _build_direct_search_url, _build_site_goal

    profile = load_site_profile("wanted")
    keyword = "android \uac1c\ubc1c\uc790"
    url = _build_direct_search_url(keyword, profile)
    goal = _build_site_goal(keyword, profile, url)

    assert url == ""
    assert "Code-generated search URL" not in goal
    assert "Open only the site home page with open_browser: https://www.wanted.co.kr" in goal
    assert "Do not construct or open a search/query URL yourself" in goal


def test_realtime_scraping_target_count_argument_overrides_intent(monkeypatch):
    from agent.tools import realtime_scraping as rt

    captured = {}

    class FakeApp:
        pass

    def fake_extract_search_intent(raw_query, profile):
        return {"search_keyword": "iOS 개발자", "target_count": 0, "source": "test"}

    def fake_run_graph_with_last_state(app, initial_state, recursion_limit):
        captured["recipe_params"] = dict(initial_state.get("recipe_params") or {})
        captured["goal"] = initial_state.get("goal", "")
        return {**initial_state, "is_finished": True, "extracted_jd": {}}, False

    monkeypatch.setattr("agent.graph.workflow.build_graph", lambda: FakeApp())
    monkeypatch.setattr(rt, "_extract_search_intent", fake_extract_search_intent)
    monkeypatch.setattr(rt, "_prepare_worker_start_screen", lambda initial_state, profile: initial_state)
    monkeypatch.setattr(rt, "_run_graph_with_last_state", fake_run_graph_with_last_state)
    monkeypatch.setattr(rt, "_commit_feedback_episodes", lambda *args, **kwargs: 0)

    result = rt.run_worker_once("iOS 개발자", site="wanted", target_count=2, task_category="검색")

    assert result["target_count"] == 2
    assert result["task_category"] == "검색"
    assert result["submission"]["target_count"] == 2
    assert result["submission"]["task_category"] == "검색"
    assert captured["recipe_params"]["target_count"] == 2
    assert captured["recipe_params"]["task_category"] == "검색"
    assert "Collect up to 2 distinct job postings" in captured["goal"]


def test_realtime_scraping_reuses_structured_search_intent(monkeypatch):
    from agent.tools import realtime_scraping as rt

    class FakeApp:
        pass

    monkeypatch.setattr("agent.graph.workflow.build_graph", lambda: FakeApp())
    monkeypatch.setattr(
        rt,
        "_extract_search_intent",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("구조화된 검색 인자를 다시 해석함")),
    )
    monkeypatch.setattr(rt, "_prepare_worker_start_screen", lambda initial_state, profile: initial_state)
    monkeypatch.setattr(
        rt,
        "_run_graph_with_last_state",
        lambda app, initial_state, recursion_limit: ({**initial_state, "is_finished": True, "extracted_jd": {}}, False),
    )
    monkeypatch.setattr(rt, "_commit_feedback_episodes", lambda *args, **kwargs: 0)

    result = rt.run_worker_once(
        "iOS 개발자",
        site="wanted",
        target_count=2,
        task_category="검색",
        search_intent_resolved=True,
    )

    assert result["keyword"] == "iOS 개발자"
    assert result["target_count"] == 2
    assert result["search_intent"]["source"] == "structured_arguments"


def test_realtime_scraping_passes_confirmed_collection_constraints_to_worker(monkeypatch):
    from agent.tools import realtime_scraping as rt

    captured = {}

    def fake_run_graph_with_last_state(app, initial_state, recursion_limit):
        captured["recipe_params"] = dict(initial_state.get("recipe_params") or {})
        captured["goal"] = initial_state.get("goal", "")
        return {**initial_state, "is_finished": True, "extracted_jd": {}}, False

    monkeypatch.setattr("agent.graph.workflow.build_graph", lambda: object())
    monkeypatch.setattr(rt, "_prepare_worker_start_screen", lambda initial_state, profile: initial_state)
    monkeypatch.setattr(rt, "_run_graph_with_last_state", fake_run_graph_with_last_state)
    monkeypatch.setattr(rt, "_commit_feedback_episodes", lambda *args, **kwargs: 0)

    intent = {
        "original_query": "지난달 서울 AI 공고 전부 비교해줘",
        "site": "wanted",
        "search_keyword": "AI 개발자",
        "count_mode": "visible_all",
        "target_count": 0,
        "filters": {"posted_date_expression": "지난달", "location": "서울"},
        "freshness_required": True,
        "purpose": "compare",
        "analysis_goal": "회사별 요구 기술 비교",
    }
    result = rt.run_worker_once(
        "AI 개발자",
        site="wanted",
        search_intent_resolved=True,
        collection_intent=intent,
    )

    assert result["collection_intent"]["filters"]["posted_date_expression"] == "지난달"
    assert captured["recipe_params"]["count_mode"] == "visible_all"
    assert captured["recipe_params"]["collection_intent"]["purpose"] == "compare"
    assert "Collect every relevant job card visible" in captured["goal"]
    assert "[Confirmed collection constraints]" in captured["goal"]
    assert '"posted_date_expression": "지난달"' in captured["goal"]
    assert '"location": "서울"' in captured["goal"]
    assert "purpose=compare" in captured["goal"]
    assert "analysis_goal=회사별 요구 기술 비교" in captured["goal"]


def test_worker_review_retries_are_disabled_by_default(monkeypatch):
    from agent.tools import realtime_scraping as rt

    monkeypatch.delenv("VISION_WORKER_REVIEW_RETRIES", raising=False)

    assert rt._worker_review_retries() == 0


def test_realtime_scraping_target_count_falls_back_to_intent(monkeypatch):
    from agent.tools import realtime_scraping as rt

    class FakeApp:
        pass

    def fake_extract_search_intent(raw_query, profile):
        return {"search_keyword": "iOS 개발자", "target_count": 3, "source": "test"}

    monkeypatch.setattr("agent.graph.workflow.build_graph", lambda: FakeApp())
    monkeypatch.setattr(rt, "_extract_search_intent", fake_extract_search_intent)
    monkeypatch.setattr(rt, "_prepare_worker_start_screen", lambda initial_state, profile: initial_state)
    monkeypatch.setattr(
        rt,
        "_run_graph_with_last_state",
        lambda app, initial_state, recursion_limit: ({**initial_state, "is_finished": True, "extracted_jd": {}}, False),
    )
    monkeypatch.setattr(rt, "_commit_feedback_episodes", lambda *args, **kwargs: 0)

    result = rt.run_worker_once("원티드에서 iOS 개발자 공고 3개", site="wanted")

    assert result["target_count"] == 3
    assert result["task_category"] == "검색"
    assert result["submission"]["target_count"] == 3
    assert result["submission"]["task_category"] == "검색"

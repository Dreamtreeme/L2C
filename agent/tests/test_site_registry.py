def test_site_registry_lists_existing_profiles():
    from agent.sites import list_supported_sites

    sites = list_supported_sites()
    slugs = {site["slug"] for site in sites}

    assert {"wanted", "jobkorea", "saramin", "worknet", "rocketpunch"}.issubset(slugs)
    assert all(site["runner"] == "vision_react" for site in sites)


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
        "_open_url_after_window_ready",
        lambda url: opened.append(url) or {"opened": True, "url": url},
    )

    result = action_tools.open_browser(site="saramin")

    assert result["status"] == "success"
    assert opened == ["https://www.saramin.co.kr"]


def test_worker_preparation_opens_requested_site_instead_of_default(monkeypatch):
    from agent.application.worker_execution_service import prepare_worker_start_screen
    from agent.graph import nodes
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
    monkeypatch.setattr(nodes, "_get_action_tools", lambda: FakeActionTools())
    monkeypatch.setattr(nodes, "prepare_reasoning_models", lambda: reasoning_warmed.append(True))
    monkeypatch.setattr(
        nodes,
        "perception_node",
        lambda state, **_kwargs: {
            "current_url": "https://www.jobkorea.co.kr",
            "current_url_stale": False,
            "current_markers": [{"id": 1}],
            "recent_images": ["screen.png"],
        },
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

    for entry in list_supported_sites(enabled_only=False):
        for key in ("manual_path", "prompt_path", "tools_path"):
            path = SITES_DIR / entry[key]
            assert path.exists(), f"missing {key} for {entry['slug']}: {path}"


def test_load_site_profile_returns_manual_prompt_and_tools():
    from agent.sites import load_site_profile

    profile = load_site_profile("wanted.co.kr")

    assert profile["entry"]["slug"] == "wanted"
    assert profile["manual"]["site"] == "wanted"
    assert "원티드" in profile["prompt"]
    assert "click_marker" in profile["tools"]["allowed_tools"]
    assert profile["manual"]["reflex_policy"]["reason_after_hit"] is True


def test_all_site_manuals_define_reflex_boundaries():
    from agent.sites import list_supported_sites, load_site_profile

    for entry in list_supported_sites():
        manual = load_site_profile(entry["slug"])["manual"]
        assert manual["stable_controls"]
        assert manual["variable_entities"]
        assert manual["reflex_policy"]["safe_actions"]
        assert manual["reflex_policy"]["unsafe_actions"]


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


def test_commander_site_tools_expose_classic_sites():
    import json
    from agent.tools.site_registry import list_collection_sites

    result = list_collection_sites.invoke({"enabled_only": True})
    data = json.loads(result)
    sites = data["sites"]
    slugs = [site["slug"] for site in sites]

    assert slugs == ["wanted", "jobkorea", "saramin", "worknet", "rocketpunch"]
    assert all(site["runner"] == "vision_react" for site in sites)
    assert all(site["classic_adapter"] == site["slug"] for site in sites)


def test_commander_site_profile_tool_returns_manual_prompt_and_tools():
    import json
    from agent.tools.site_registry import get_collection_site_profile

    result = get_collection_site_profile.invoke({"site": "jobkorea"})
    data = json.loads(result)

    assert data["site"]["slug"] == "jobkorea"
    assert data["site"]["classic_adapter"] == "jobkorea"
    assert data["manual"]["site"] == "jobkorea"
    assert "click_marker" in data["tools"]["allowed_tools"]
    assert "JobKorea" in data["prompt"]

def test_realtime_scraping_extracts_user_search_intent_with_llm(monkeypatch):
    from agent.sites import load_site_profile
    from agent.tools.realtime_scraping import SearchIntent, _extract_search_intent

    class FakeStructuredLLM:
        def invoke(self, messages):
            return SearchIntent(search_keyword="ai\uc751\uc6a9\uc5d4\uc9c0\ub2c8\uc5b4", target_count=8)

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

    profile = {
        "entry": {"slug": "sample", "display_name": "Sample", "base_url": "https://example.com"},
        "manual": {
            "base_url": "https://example.com",
            "navigation_policy": {
                "search_url_template": "https://example.com/search?query={query}&tab=position",
            },
            "common_flow": [],
            "stable_controls": [],
            "variable_entities": [],
            "ignore_elements": [],
            "collection_policy": {"required_fields": []},
            "reflex_policy": {"safe_actions": [], "unsafe_actions": []},
        },
        "tools": {"allowed_tools": []},
        "prompt": "",
    }
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


def test_realtime_scraping_keeps_collection_intent_out_of_worker_prompt(monkeypatch):
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
    assert "Structured user request" not in captured["goal"]
    assert "analysis_goal=회사별 요구 기술 비교" not in captured["goal"]


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

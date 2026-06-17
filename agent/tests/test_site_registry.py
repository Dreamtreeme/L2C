def test_site_registry_lists_existing_profiles():
    from agent.sites import list_supported_sites

    sites = list_supported_sites()
    slugs = {site["slug"] for site in sites}

    assert {"wanted", "jobkorea", "saramin", "worknet", "rocketpunch"}.issubset(slugs)
    assert all(site["runner"] == "vision_react" for site in sites)


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

def test_realtime_scraping_normalizes_user_query_for_search_keyword():
    from agent.sites import load_site_profile
    from agent.tools.realtime_scraping import _normalize_search_keyword

    profile = load_site_profile("wanted")
    query = "\uc6d0\ud2f0\ub4dc\uc5d0\uc11c ai\uc751\uc6a9\uc5d4\uc9c0\ub2c8\uc5b4 \ucc44\uc6a9\uacf5\uace0 \ucc3e\uc544\uc918"

    assert _normalize_search_keyword(query, profile) == "ai\uc751\uc6a9\uc5d4\uc9c0\ub2c8\uc5b4"


def test_realtime_scraping_direct_search_url_encodes_keyword():
    from urllib.parse import parse_qs, urlparse

    from agent.sites import load_site_profile
    from agent.tools.realtime_scraping import _build_direct_search_url, _build_site_goal

    profile = load_site_profile("wanted")
    keyword = "AI \uc751\uc6a9 \uc5d4\uc9c0\ub2c8\uc5b4"
    url = _build_direct_search_url(keyword, profile)
    parsed = urlparse(url)

    assert parsed.scheme == "https"
    assert parsed.netloc == "www.wanted.co.kr"
    assert parsed.path == "/search"
    assert parse_qs(parsed.query)["query"] == [keyword]
    assert parse_qs(parsed.query)["tab"] == ["position"]
    assert "Code-generated search URL" in _build_site_goal(keyword, profile, url)

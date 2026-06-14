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

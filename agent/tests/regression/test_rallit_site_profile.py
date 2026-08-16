from agent.runtime.site_context import infer_site_page_role
from agent.sites import get_official_site_url, load_site_profile


def test_rallit_profile_registers_official_homepage() -> None:
    profile = load_site_profile("rallit")

    assert profile.display_name == "랠릿"
    assert profile.base_url == "https://www.rallit.com/"
    assert profile.domains == ("www.rallit.com", "rallit.com")
    assert get_official_site_url("랠릿") == "https://www.rallit.com"


def test_rallit_profile_uses_visible_detail_cues() -> None:
    role = infer_site_page_role(
        "https://www.rallit.com/",
        ["주요업무", "자격요건"],
    )

    assert role == "job_detail"


def test_rallit_profile_keeps_detail_role_below_body_sections() -> None:
    role = infer_site_page_role(
        "https://www.rallit.com/positions/example-posting",
        ["랠릿", "회사/채용 공고로 검색"],
    )

    assert role == "job_detail"

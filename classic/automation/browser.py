"""Classic Playwright 브라우저 생명주기."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from playwright.sync_api import Page, sync_playwright

from agent.config import get_settings


@contextmanager
def open_browser_page() -> Iterator[Page]:
    """설정된 viewport로 Playwright 페이지 하나를 열고 종료한다."""

    browser_settings = get_settings().browser
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=browser_settings.playwright_headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={
                "width": browser_settings.chrome_window_width,
                "height": browser_settings.chrome_window_height,
            },
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        try:
            yield context.new_page()
        finally:
            context.close()
            browser.close()

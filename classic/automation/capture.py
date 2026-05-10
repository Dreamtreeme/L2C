"""Playwright 기반 본문 추출 디스패처.

URL을 받아 적절한 사이트 어댑터를 선택하고, 어댑터에 페이지를 위임합니다.
브라우저 lifecycle(launch, context, goto, close)은 이 모듈이 책임집니다.
사이트별 추출 로직은 classic/automation/sites/{wanted, jobkorea, ...}.py 참조.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

from shared.config import (
    CHROME_WINDOW_HEIGHT,
    CHROME_WINDOW_WIDTH,
    PAGE_LOAD_WAIT_SEC,
    PLAYWRIGHT_HEADLESS,
    PLAYWRIGHT_TIMEOUT_MS,
)

from .sites import resolve_adapter

logger = logging.getLogger(__name__)


def capture_and_extract_dom(
    url: str, save_name: str | None = None
) -> tuple[Path | None, dict]:
    """URL에서 채용공고 본문을 추출.

    1. URL → 사이트 어댑터 매칭
    2. Playwright 브라우저 생성 + URL 진입
    3. 어댑터.extract(page)에 위임
    4. {company_name, position, full_text} dict 반환
    """
    from zoneinfo import ZoneInfo

    logger.info(f"[capture_and_extract_dom] URL={url}")

    if save_name is None:
        kst_now = datetime.now(ZoneInfo("Asia/Seoul"))
        save_name = f"classic_{kst_now.strftime('%Y%m%d_%H%M%S')}"

    adapter = resolve_adapter(url)
    logger.info(f"사이트 어댑터: {adapter.name}")

    dom_data: dict = {
        "company_name": None,
        "position": None,
        "full_text": None,
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=PLAYWRIGHT_HEADLESS,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={"width": CHROME_WINDOW_WIDTH, "height": CHROME_WINDOW_HEIGHT},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        try:
            logger.info(f"페이지 접속 시도 중... (timeout={PLAYWRIGHT_TIMEOUT_MS}ms)")
            try:
                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=PLAYWRIGHT_TIMEOUT_MS,
                )
            except Exception as e:
                logger.warning(f"페이지 로딩 중 타임아웃 발생 (무시하고 진행): {e}")

            time.sleep(PAGE_LOAD_WAIT_SEC)

            # 사이트 어댑터에 추출 위임
            dom_data = adapter.extract(page)

        except Exception as e:
            logger.error(f"DOM 추출 중 오류 발생: {e}")
        finally:
            browser.close()

    # 스크린샷 저장은 현재 미구현. 시그니처 호환을 위해 None 반환.
    return None, dom_data

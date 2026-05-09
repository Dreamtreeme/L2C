"""
Playwright 기반 풀스크린 캡처 자동화.
불필요한 동작(스크롤, 더보기 클릭 등) 없이 DOM Bounding Box 기반으로 핵심 영역만 크롭하여 캡처합니다.
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
    SCREENSHOTS_DIR,
)

logger = logging.getLogger(__name__)

def capture_and_extract_dom(url: str, save_name: str | None = None) -> tuple[Path, dict]:
    from zoneinfo import ZoneInfo
    """
    URL에 접속하여:
    1. 전체 스크린샷 캡처
    2. 주요 섹션의 DOM Bounding Box를 기반으로 텍스트 추출
    """
    logger.info(f"[capture_and_extract_dom] URL={url}")
    
    if save_name is None:
        kst_now = datetime.now(ZoneInfo("Asia/Seoul"))
        save_name = f"classic_{kst_now.strftime('%Y%m%d_%H%M%S')}"
    out_path = SCREENSHOTS_DIR / f"{save_name}.png"

    dom_data = {
        "company_name": None,
        "position": None,
        "main_tasks": [],
        "requirements": [],
        "preferred": [],
        "benefits": []
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=PLAYWRIGHT_HEADLESS,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            viewport={'width': CHROME_WINDOW_WIDTH, 'height': CHROME_WINDOW_HEIGHT},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        try:
            page.goto(url, wait_until="networkidle", timeout=15000)
            time.sleep(PAGE_LOAD_WAIT_SEC)

            # 1. '더보기' 클릭
            more_btn = page.get_by_text("상세 정보 더 보기", exact=False).first
            if more_btn.is_visible(timeout=2000):
                more_btn.click()
                time.sleep(1.0)

            # 2. DOM Bounding Box 기반 데이터 추출 (고전적 방식)
            # 회사명 및 포지션 (Wanted 기준 특정 셀렉터 활용 가능하나, 범용성을 위해 텍스트 탐색)
            dom_data["company_name"] = _get_inner_text_safe(page.locator("section.JobHeader_className__W_7n9 h4").first)
            dom_data["position"] = _get_inner_text_safe(page.locator("section.JobHeader_className__W_7n9 h2").first)

            # 섹션별 키워드 매핑
            sections = {
                "주요업무": "main_tasks",
                "자격요건": "requirements",
                "우대사항": "preferred",
                "혜택 및 복지": "benefits"
            }

            for keyword, key in sections.items():
                try:
                    # 키워드 헤더를 찾음
                    header = page.get_by_text(keyword, exact=True).first
                    if header.is_visible():
                        # 헤더 부모의 다음 형제 요소에서 텍스트 추출 (고전적 스크래핑 전략)
                        text = page.evaluate("""
                            (keyword) => {
                                const el = Array.from(document.querySelectorAll('*')).find(e => e.textContent.trim() === keyword);
                                if (el && el.parentElement) {
                                    const next = el.parentElement.nextElementSibling;
                                    return next ? next.innerText : null;
                                }
                                return null;
                            }
                        """, keyword)
                        if text:
                            dom_data[key] = [line.strip() for line in text.split('\n') if line.strip()]
                except Exception as e:
                    logger.debug(f"DOM 추출 실패 ({keyword}): {e}")

            # 3. 전체 스크린샷 저장
            page.screenshot(path=str(out_path), full_page=True)

        except Exception as e:
            logger.error(f"DOM 추출 중 오류 발생: {e}")
        finally:
            browser.close()
        
    return out_path, dom_data

def _get_inner_text_safe(locator) -> str | None:
    try:
        if locator.is_visible():
            return locator.inner_text().strip()
    except:
        pass
    return None

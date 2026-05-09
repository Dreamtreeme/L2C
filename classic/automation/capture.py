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
    PLAYWRIGHT_TIMEOUT_MS,
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
        "full_text": None
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
            # 1. 페이지 접속 (타임아웃 완화 및 대기 전략 변경)
            logger.info(f"페이지 접속 시도 중... (timeout={PLAYWRIGHT_TIMEOUT_MS}ms)")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT_MS)
            except Exception as e:
                logger.warning(f"페이지 로딩 중 타임아웃 발생 (무시하고 진행): {e}")

            # 필수 요소가 나타날 때까지 대기 (최대 10초)
            try:
                page.wait_for_selector("section.JobHeader_className__W_7n9", timeout=10000)
            except:
                logger.warning("필수 섹션 로딩 대기 시간 초과")

            time.sleep(PAGE_LOAD_WAIT_SEC)

            # 2. '더보기' 클릭
            try:
                # 텍스트 기반으로 더보기 버튼 탐색
                more_btn = page.get_by_text("상세 정보 더 보기", exact=False).first
                if more_btn.is_visible(timeout=3000):
                    more_btn.click()
                    time.sleep(1.0)
                    logger.info("'상세 정보 더 보기' 버튼 클릭 완료")
            except Exception as e:
                logger.debug(f"더보기 버튼 클릭 실패 또는 불필요: {e}")

            # 2. 본문 텍스트 통째로 추출 (Wanted 본문 컨테이너 타겟팅)
            # 원티드 본문의 실제 클래스명 반영 및 범용 셀렉터
            content_locator = page.locator("div.JobDescription_JobDescription__b9_L3, .JobDescription_JobDescription__b9_L3, section.job-description").first
            
            if not content_locator.is_visible():
                # 백업: 본문으로 추정되는 가장 큰 섹션 탐색 (주요업무 키워드 포함 섹션)
                content_locator = page.locator("section").filter(has_text="주요업무").first
            
            if not content_locator.is_visible():
                # 최종 백업: 본문 전체
                content_locator = page.locator("article, main").first
            
            full_text = _get_inner_text_safe(content_locator)
            
            if full_text:
                logger.info(f"본문 텍스트 추출 완료 ({len(full_text)}자)")
                dom_data["full_text"] = full_text
            else:
                logger.warning("본문 텍스트를 찾지 못했습니다.")
                dom_data["full_text"] = ""

            # 기본 메타데이터 추출
            dom_data["company_name"] = _get_inner_text_safe(page.locator("section.JobHeader_className__W_7n9 h4, h4").first)
            dom_data["position"] = _get_inner_text_safe(page.locator("section.JobHeader_className__W_7n9 h2, h2").first)

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

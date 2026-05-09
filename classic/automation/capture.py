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
    SCREENSHOTS_DIR,
)

logger = logging.getLogger(__name__)

def capture_full_page(url: str, save_name: str | None = None) -> Path:
    from zoneinfo import ZoneInfo
    """URL에 접속하여 Playwright로 핵심 영역 스크린샷을 찍습니다."""
    logger.info(f"[capture_full_page] URL={url}")
    
    if save_name is None:
        kst_now = datetime.now(ZoneInfo("Asia/Seoul"))
        save_name = f"capture_{kst_now.strftime('%Y%m%d_%H%M%S')}"
    out_path = SCREENSHOTS_DIR / f"{save_name}.png"

    with sync_playwright() as p:
        # 봇 탐지 우회를 위해 headless=False 적용
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            viewport={'width': CHROME_WINDOW_WIDTH, 'height': CHROME_WINDOW_HEIGHT},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        logger.info("페이지 로딩 중...")
        try:
            page.goto(url, wait_until="networkidle", timeout=15000)
        except Exception as e:
            logger.warning(f"네트워크 안정화 대기 시간 초과(무시하고 진행): {e}")
        
        # 추가 대기 (비동기 렌더링 완료)
        time.sleep(PAGE_LOAD_WAIT_SEC)
        
        # --- 핵심 영역 좌표 탐색 (Zero-Cost OCR) ---
        logger.info("핵심 영역 좌표 탐색 중...")
        
        # '더보기' 버튼이 있다면 먼저 클릭해야 할 수도 있음.
        # Wanted의 경우 상세 내용을 숨겨두는 버튼이 있다면 눌러줘야 전체 Y 좌표를 알 수 있음.
        try:
            more_btn = page.get_by_text("상세 정보 더 보기", exact=False).first
            if more_btn.is_visible(timeout=1000):
                more_btn.click()
                time.sleep(0.5)
                logger.info("'상세 정보 더 보기' 버튼 클릭 완료")
        except Exception:
            pass

        # 시작 y는 무조건 0부터 시작하여 상단 네비게이션, 회사명, 포지션, 경력 정보가 잘리지 않게 함
        end_y = 0
        for keyword in ["태그", "근무지역", "마감일", "채용 전형", "혜택 및 복지"]:
            try:
                # 페이지 맨 아래쪽에서 찾는 것이 정확하므로 역순으로 탐색하거나 마지막 요소를 사용
                locs = page.get_by_text(keyword, exact=False).all()
                if locs:
                    # 보통 본문 아래쪽에 위치한 섹션을 찾음
                    last_loc = locs[-1]
                    box = last_loc.bounding_box()
                    # 헤더 영역(약 200px)보다 아래에 있는 진짜 본문 끝을 찾아야 함
                    if box and box["y"] > 200:
                        end_y = box["y"] + box["height"] + 200 # 아래쪽 여백 200px
                        logger.info(f"종료 키워드 '{keyword}' 찾음: y={box['y']}")
                        break
            except Exception:
                continue

        # 좌표 기반 크롭 캡처 (클립 영역 지정)
        if end_y > 200:
            clip_region = {
                "x": 0,
                "y": 0, # 무조건 0부터 캡처
                "width": CHROME_WINDOW_WIDTH,
                "height": end_y
            }
            logger.info(f"핵심 영역 크롭 캡처 진행 중... clip={clip_region}")
            page.screenshot(path=str(out_path), clip=clip_region, full_page=False)
        else:
            # 실패 시 다운샘플링 휴리스틱 크롭 혹은 풀페이지
            logger.warning("종료 키워드 좌표를 찾지 못했습니다. 전체 페이지 캡처로 대체합니다.")
            page.screenshot(path=str(out_path), full_page=True)
            
        browser.close()
        
    logger.info(f"스크린샷 저장 완료: {out_path}")
    return out_path

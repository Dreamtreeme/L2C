"""원티드(Wanted) 어댑터.

이 어댑터는 토큰 절약을 위해 사이트 특정 셀렉터를 사용합니다.
관련 한계는 루트 README의 'Classic - 전통 자동화' 섹션 참조.
"""

from __future__ import annotations

import logging
import time

from .base import SiteAdapter, get_inner_text_safe

logger = logging.getLogger(__name__)


class WantedAdapter(SiteAdapter):
    name = "wanted"

    def matches(self, url: str) -> bool:
        return "wanted.co.kr" in url

    def extract(self, page) -> dict:
        dom_data: dict = {
            "company_name": None,
            "position": None,
            "full_text": None,
        }

        # 1. 필수 요소가 나타날 때까지 대기 (최대 10초)
        try:
            page.wait_for_selector(
                "section.JobHeader_className__W_7n9", timeout=10000
            )
        except Exception:
            logger.warning("[wanted] 필수 섹션 로딩 대기 시간 초과")

        # 2. '상세 정보 더 보기' 클릭 (truncate 우회 — 필수)
        try:
            more_btn = page.get_by_text("상세 정보 더 보기", exact=False).first
            if more_btn.is_visible(timeout=3000):
                more_btn.click()
                time.sleep(1.0)
                logger.info("[wanted] '상세 정보 더 보기' 클릭 완료")
        except Exception as e:
            logger.debug(f"[wanted] 더보기 버튼 없음/실패: {e}")

        # 3. 본문 컨테이너 (사이트 특정 클래스 → 키워드 → 시맨틱 폴백)
        content_locator = page.locator(
            "div.JobDescription_JobDescription__b9_L3, "
            ".JobDescription_JobDescription__b9_L3, "
            "section.job-description"
        ).first

        if not content_locator.is_visible():
            content_locator = (
                page.locator("section").filter(has_text="주요업무").first
            )
        if not content_locator.is_visible():
            content_locator = page.locator("article, main").first

        full_text = get_inner_text_safe(content_locator)
        if full_text:
            logger.info(f"[wanted] 본문 텍스트 추출 완료 ({len(full_text)}자)")
            dom_data["full_text"] = full_text
        else:
            logger.warning("[wanted] 본문 텍스트를 찾지 못했습니다.")
            dom_data["full_text"] = ""

        # 4. 메타 (LLM이 놓쳤을 때 폴백용)
        dom_data["company_name"] = get_inner_text_safe(
            page.locator("section.JobHeader_className__W_7n9 h4, h4").first
        )
        dom_data["position"] = get_inner_text_safe(
            page.locator("section.JobHeader_className__W_7n9 h2, h2").first
        )

        return dom_data

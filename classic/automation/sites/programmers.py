"""프로그래머스(Programmers) 어댑터.

React 기반이라 클래스명 난독화 가능성이 있습니다.
초기 구현은 시맨틱 태그 폴백 기반이며, 첫 테스트 후 본문 셀렉터를 추가하세요.
"""

from __future__ import annotations

import logging
import time

from .base import SiteAdapter, get_inner_text_safe

logger = logging.getLogger(__name__)


class ProgrammersAdapter(SiteAdapter):
    name = "programmers"

    def matches(self, url: str) -> bool:
        # career.programmers.co.kr 또는 programmers.co.kr/job
        return "programmers.co.kr" in url

    def extract(self, page) -> dict:
        dom_data: dict = {
            "company_name": None,
            "position": None,
            "full_text": None,
        }

        # TODO(programmers): 첫 테스트 후 필수 요소를 wait_for_selector로 잡기
        time.sleep(1.0)

        # 1차 본문: 시맨틱 → body 폴백
        content_locator = page.locator(
            "div.position-detail, "       # TODO: 첫 테스트 후 실제 셀렉터로 교체
            "section.job-detail, "        # TODO: 후보 셀렉터
            "article, "
            "main, "
            "body"
        ).first

        full_text = get_inner_text_safe(content_locator)
        if full_text:
            logger.info(f"[programmers] 본문 추출 완료 ({len(full_text)}자)")
            dom_data["full_text"] = full_text
        else:
            logger.warning("[programmers] 본문 추출 실패")
            dom_data["full_text"] = ""

        return dom_data

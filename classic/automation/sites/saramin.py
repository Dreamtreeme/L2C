"""사람인(Saramin) 어댑터.

초기 구현은 시맨틱 태그 폴백 기반의 범용 추출입니다.
실제 URL 테스트 후 본문 컨테이너 셀렉터를 TODO 자리에 추가해 토큰을 절약하세요.
"""

from __future__ import annotations

import logging
import time

from .base import SiteAdapter, get_inner_text_safe

logger = logging.getLogger(__name__)


class SaraminAdapter(SiteAdapter):
    name = "saramin"

    def matches(self, url: str) -> bool:
        return "saramin.co.kr" in url

    def extract(self, page) -> dict:
        dom_data: dict = {
            "company_name": None,
            "position": None,
            "full_text": None,
        }

        # TODO(saramin): 첫 테스트 후 필수 요소를 wait_for_selector로 잡기
        # 예) page.wait_for_selector(".wrap_jv_cont", timeout=10000)
        time.sleep(1.0)

        # TODO(saramin): 본문이 iframe 안에 있는 케이스가 있음. 필요시 page.frame() 사용

        # 1차 본문: 시맨틱 → body 폴백
        content_locator = page.locator(
            "div.wrap_jv_cont, "         # TODO: 첫 테스트 후 실제 셀렉터로 교체
            "div.recruitment-detail, "    # TODO: 후보 셀렉터
            "article, "
            "main, "
            "body"
        ).first

        full_text = get_inner_text_safe(content_locator)
        if full_text:
            logger.info(f"[saramin] 본문 추출 완료 ({len(full_text)}자)")
            dom_data["full_text"] = full_text
        else:
            logger.warning("[saramin] 본문 추출 실패")
            dom_data["full_text"] = ""

        return dom_data

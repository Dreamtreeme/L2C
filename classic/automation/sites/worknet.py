"""워크넷(Worknet) 어댑터.

정부 운영 사이트로 비교적 단순한 HTML 구조가 예상됩니다.
초기 구현은 시맨틱 태그 폴백 기반이며, 실제 URL 테스트 후 본문 셀렉터를 추가하세요.
"""

from __future__ import annotations

import logging
import time

from .base import SiteAdapter, get_inner_text_safe

logger = logging.getLogger(__name__)


class WorknetAdapter(SiteAdapter):
    name = "worknet"

    def matches(self, url: str) -> bool:
        # work.go.kr / www.work.go.kr / work24.go.kr 등 도메인 변동 가능
        return "work.go.kr" in url or "work24.go.kr" in url

    def extract(self, page) -> dict:
        dom_data: dict = {
            "company_name": None,
            "position": None,
            "full_text": None,
        }

        # TODO(worknet): 첫 테스트 후 필수 요소를 wait_for_selector로 잡기
        time.sleep(1.0)

        # 1차 본문: 시맨틱 → body 폴백
        content_locator = page.locator(
            "div.detail_view, "          # TODO: 첫 테스트 후 실제 셀렉터로 교체
            "div.recruit-content, "       # TODO: 후보 셀렉터
            "article, "
            "main, "
            "body"
        ).first

        full_text = get_inner_text_safe(content_locator)
        if full_text:
            logger.info(f"[worknet] 본문 추출 완료 ({len(full_text)}자)")
            dom_data["full_text"] = full_text
        else:
            logger.warning("[worknet] 본문 추출 실패")
            dom_data["full_text"] = ""

        return dom_data

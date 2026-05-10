"""잡코리아(JobKorea) 어댑터.

초기 구현은 시맨틱 태그 폴백(article/main/body)에 의존하는 범용 추출입니다.
실제 URL로 1차 테스트 후, 토큰 최적화를 위해 본문 컨테이너 셀렉터를
TODO 주석 자리에 추가하세요.
"""

from __future__ import annotations

import logging
import time

from .base import SiteAdapter, get_inner_text_safe

logger = logging.getLogger(__name__)


class JobKoreaAdapter(SiteAdapter):
    name = "jobkorea"

    def matches(self, url: str) -> bool:
        return "jobkorea.co.kr" in url

    def extract(self, page) -> dict:
        dom_data: dict = {
            "company_name": None,
            "position": None,
            "full_text": None,
        }

        # TODO(jobkorea): 첫 테스트 후 필수 요소(직무 헤더 등)를 wait_for_selector로 잡기
        # 예) page.wait_for_selector(".tbList, .secTit", timeout=10000)
        time.sleep(1.0)

        # TODO(jobkorea): truncate UI가 있다면 텍스트 기반으로 클릭
        # 예) page.get_by_text("상세보기", exact=False).first.click()

        # 1차 본문: 시맨틱 → body 폴백 (사이트 특정 클래스는 1차 테스트 후 추가)
        content_locator = page.locator(
            "article.tbList, "          # TODO: 첫 테스트 후 실제 클래스로 교체
            "div.tbCol, "                # TODO: 후보 셀렉터
            "article, "
            "main, "
            "body"
        ).first

        full_text = get_inner_text_safe(content_locator)
        if full_text:
            logger.info(f"[jobkorea] 본문 추출 완료 ({len(full_text)}자)")
            dom_data["full_text"] = full_text
        else:
            logger.warning("[jobkorea] 본문 추출 실패")
            dom_data["full_text"] = ""

        # company_name, position은 LLM에 위임 (사이트별 추출 코드는 TODO)
        return dom_data

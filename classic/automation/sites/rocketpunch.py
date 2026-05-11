"""로켓펀치(Rocketpunch) 어댑터.

스타트업 중심 채용 플랫폼. 채용공고 페이지는 보통 다음 URL 패턴:
  https://www.rocketpunch.com/jobs/{id}
  https://www.rocketpunch.com/companies/{slug}/jobs/{id}

초기 구현은 시맨틱 폴백(article/main/body)에 의존하는 범용 추출입니다.
실제 URL 1차 테스트 후, 본문 컨테이너 클래스(.job-detail, .position-detail 등
실제 마크업)를 TODO 자리에 채워 토큰을 절약하세요.
"""

from __future__ import annotations

import logging
import time

from .base import SiteAdapter, get_inner_text_safe

logger = logging.getLogger(__name__)


class RocketpunchAdapter(SiteAdapter):
    name = "rocketpunch"

    def matches(self, url: str) -> bool:
        return "rocketpunch.com" in url

    def extract(self, page) -> dict:
        dom_data: dict = {
            "company_name": None,
            "position": None,
            "full_text": None,
        }

        # TODO(rocketpunch): 첫 테스트 후 필수 요소(직무 헤더·본문 컨테이너)를
        # wait_for_selector로 잡기. 예) page.wait_for_selector(".job-detail", ...)
        time.sleep(1.5)

        # 본문 컨테이너 후보:
        # TODO: 실 페이지 마크업 확인 후 정확한 클래스로 교체
        content_locator = page.locator(
            ".job-detail, "
            ".position-detail, "
            ".content-container, "
            "article, "
            "main, "
            "body"
        ).first

        full_text = get_inner_text_safe(content_locator)
        if full_text:
            logger.info(f"[rocketpunch] 본문 추출 완료 ({len(full_text)}자)")
            dom_data["full_text"] = full_text
        else:
            logger.warning("[rocketpunch] 본문 추출 실패")
            dom_data["full_text"] = ""

        # company_name, position은 LLM에 위임
        return dom_data

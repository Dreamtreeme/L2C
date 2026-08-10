"""로켓펀치(Rocketpunch) 어댑터.

URL 패턴:
  https://www.rocketpunch.com/jobs/{id}                       — dedicated 상세
  https://www.rocketpunch.com/jobs?selectedJobId={id}         — listing + 우측 패널

listing+selectedJobId URL은 canonical `/jobs/{id}`로 이동한 뒤 상세 본문을
가져온다. 회사명과 직무명은 다른 Classic 어댑터와 마찬가지로 본문 정제 단계가
추출한다.

본문 컨테이너는 로켓펀치가 PandaCSS atomic class를 쓰므로 시맨틱 태그(article,
main, body) cascade에 의존한다 — .h1/.position-title 같은 의미 셀렉터는 없음.
"""

from __future__ import annotations

import logging
import re
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

        selected = re.search(r"selectedJobId=(\d+)", page.url)
        if selected:
            canonical = f"https://www.rocketpunch.com/jobs/{selected.group(1)}"
            logger.info(f"[rocketpunch] canonical로 이동: {canonical}")
            try:
                page.goto(canonical, wait_until="domcontentloaded", timeout=15000)
            except Exception as e:
                logger.warning(f"[rocketpunch] canonical 이동 실패 (계속 진행): {e}")

        time.sleep(1.5)
        content_locator = page.locator(
            ".job-detail, .position-detail, .content-container, article, main, body"
        ).first
        body_text = get_inner_text_safe(content_locator) or ""
        if body_text:
            logger.info(f"[rocketpunch] 본문 추출 완료 ({len(body_text)}자)")
            dom_data["full_text"] = body_text
        else:
            logger.warning("[rocketpunch] 본문 추출 실패")
            dom_data["full_text"] = ""

        return dom_data

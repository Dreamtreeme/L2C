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
        import time

        dom_data: dict = {
            "company_name": None,
            "position": None,
            "full_text": None,
        }

        # 1. 고용24/워크넷 공고 컨테이너 로딩 대기
        try:
            # 고용24 통합 이후 주요 클래스인 .detail-tit 또는 .job-detail-cont 대기
            page.wait_for_selector(".detail-tit, .job-detail-cont, #contents", timeout=10000)
        except Exception:
            logger.warning("[worknet] 컨테이너 로딩 대기 시간 초과")

        time.sleep(1.0)

        # 2. 메타데이터 추출 (회사명, 직무명)
        try:
            dom_data["company_name"] = get_inner_text_safe(page.locator(".company-name, .info-comp").first)
            dom_data["position"] = get_inner_text_safe(page.locator(".detail-tit, h3.tit").first)
        except Exception as e:
            logger.debug(f"[worknet] 메타데이터 추출 중 일부 실패: {e}")

        # 3. 본문 텍스트 추출 (고용24/워크넷은 보통 상세내용이 펼쳐져 있음)
        # .job-detail-cont 가 핵심 본문 영역
        content_locator = page.locator(
            ".job-detail-cont, "
            ".detail-view, "
            "#contents, "
            "article, "
            "main"
        ).first
        
        full_text = get_inner_text_safe(content_locator)
        
        if full_text:
            logger.info(f"[worknet] DOM 추출 완료 ({len(full_text)}자)")
            dom_data["full_text"] = full_text
        else:
            logger.warning("[worknet] 본문 추출 실패")
            dom_data["full_text"] = ""

        return dom_data

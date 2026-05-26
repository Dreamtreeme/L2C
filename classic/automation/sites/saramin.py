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

        # 사람인 공고 컨테이너 로딩 대기
        try:
            # 메인 프레임 대기
            page.wait_for_selector(".wrap_jv_cont, #iframe_content_0", timeout=10000)
        except Exception:
            logger.warning("[saramin] 메인 컨테이너 로딩 대기 시간 초과")

        time.sleep(1.0)

        full_text = ""

        # 1) 메인 페이지에서 먼저 탐색
        content_locator = page.locator(".jv_cont.jv_detail").first
        full_text = get_inner_text_safe(content_locator) or ""

        # 2) frame_locator를 통한 iframe 탐색 (jview 방어)
        if not full_text.strip():
            logger.debug("[saramin] 메인 페이지에서 본문을 찾지 못함. iframe(#iframe_content_0) 탐색 시작...")
            try:
                frame_loc = page.frame_locator("#iframe_content_0")
                t = frame_loc.locator(".jv_cont.jv_detail").first.inner_text(timeout=3000).strip()
                if t:
                    full_text = t
                    logger.info("[saramin] iframe_content_0 에서 본문 발견")
            except Exception as e:
                logger.debug(f"[saramin] frame_locator 탐색 실패 (CORS 문제일 수 있음): {e}")

        # 3) 최후의 보루: iframe CORS(교차 출처) 제한이나 Playwright 로딩 타임아웃으로 접근이 불가능한 경우,
        # URL에서 rec_idx를 추출해 순수 상세내용 페이지(view-detail)를 HTTP GET으로 직접 가져옴 (제일 빠르고 확실함)
        if not full_text.strip():
            import re
            m = re.search(r"rec_idx=(\d+)", page.url)
            if m:
                rec_idx = m.group(1)
                detail_url = f"https://www.saramin.co.kr/zf_user/jobs/relay/view-detail?rec_idx={rec_idx}"
                logger.info(f"[saramin] iframe 접근 완전 실패. HTTP GET으로 우회 추출: {detail_url}")
                try:
                    import requests as _requests
                    from bs4 import BeautifulSoup
                    resp = _requests.get(
                        detail_url,
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                        timeout=10
                    )
                    resp.raise_for_status()
                    html = resp.text

                    # BeautifulSoup으로 HTML 파싱 (script/style 제거 후 텍스트 추출)
                    soup = BeautifulSoup(html, "html.parser")
                    for tag in soup(["script", "style"]):
                        tag.decompose()
                    text = soup.get_text(separator=" ", strip=True)

                    if text:
                        full_text = text
                        logger.info("[saramin] HTTP GET 우회 상세 페이지에서 본문 추출 완료")
                except Exception as e:
                    logger.warning(f"[saramin] HTTP GET 우회 상세 페이지 추출 실패: {e}")

        # 4) 폴백
        if not full_text.strip():
            logger.warning("[saramin] 모든 수단 실패. 범용 태그(body) 폴백 사용.")
            fallback = page.locator("article, main, body").first
            try:
                full_text = get_inner_text_safe(fallback) or ""
            except Exception:
                pass

        if full_text.strip():
            logger.info(f"[saramin] 본문 추출 완료 ({len(full_text)}자)")
            dom_data["full_text"] = full_text
        else:
            logger.warning("[saramin] 본문 추출 완전 실패")
            dom_data["full_text"] = ""

        return dom_data

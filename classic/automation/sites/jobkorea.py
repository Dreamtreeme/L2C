"""잡코리아(JobKorea) 어댑터.

잡코리아는 상세 모집요강(담당업무·자격요건·우대사항)을 별도 iframe에 담는
케이스가 많습니다. 그래서 메인 페이지 본문(.readWrap 등)만 긁으면
메타데이터(모집요강, 기업정보, 접수기간)만 잡히고 정작 핵심 본문이 빠집니다.

이 어댑터는 메인 페이지 본문 + 모든 iframe의 body 텍스트를 합쳐 LLM에 넘깁니다.
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

        # 공고 컨테이너 로딩 대기
        try:
            page.wait_for_selector(".readWrap", timeout=10000)
        except Exception:
            logger.warning("[jobkorea] 본문(.readWrap) 로딩 대기 시간 초과")

        time.sleep(1.0)

        # 다른 탭이 눌려있을 가능성에 대비해 '상세요강' 탭을 명시 클릭
        try:
            tab_btn = page.get_by_role("tab", name="상세요강")
            if tab_btn.is_visible(timeout=1000):
                tab_btn.click()
                time.sleep(0.5)
        except Exception as e:
            logger.debug(f"[jobkorea] 상세요강 탭 클릭 스킵: {e}")

        # 1) 메인 페이지 본문 (모집요강·기업정보·접수기간 등 메타데이터)
        content_locator = page.locator(
            ".readWrap, article, main, body"
        ).first
        main_text = get_inner_text_safe(content_locator) or ""

        # 2) iframe 내부 본문 (담당업무·자격요건·우대사항 등 실 모집내용)
        # 잡코리아는 상세 설명을 iframe에 박아두는 경우가 많아,
        # 메인만 가져오면 핵심을 놓침.
        iframe_texts: list[str] = []
        for frame in page.frames:
            if frame == page.main_frame:
                continue
            try:
                try:
                    frame.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                t = frame.locator("body").inner_text().strip()
                if t:
                    iframe_texts.append(t)
                    src = (frame.url or frame.name or "?")[:80]
                    logger.info(
                        f"[jobkorea] iframe 본문 수집 ({len(t)}자) - {src}"
                    )
            except Exception as e:
                logger.debug(f"[jobkorea] iframe 추출 실패 ({frame.url}): {e}")

        # 3) 합치기
        parts = [p for p in [main_text, *iframe_texts] if p]
        full_text = "\n\n".join(parts)

        if full_text:
            iframe_total = sum(len(t) for t in iframe_texts)
            logger.info(
                f"[jobkorea] 본문 추출 완료 "
                f"(메인 {len(main_text)}자 + iframe {len(iframe_texts)}개={iframe_total}자)"
            )
            dom_data["full_text"] = full_text
        else:
            logger.warning("[jobkorea] 본문 추출 실패 (메인·iframe 모두 비어 있음)")
            dom_data["full_text"] = ""

        # company_name, position은 LLM이 본문에서 추출
        return dom_data

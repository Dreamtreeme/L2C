"""로켓펀치(Rocketpunch) 어댑터.

URL 패턴:
  https://www.rocketpunch.com/jobs/{id}                       — dedicated 상세
  https://www.rocketpunch.com/jobs?selectedJobId={id}         — listing + 우측 패널

추출 전략:
  1) listing+selectedJobId로 들어오면, 좌측 카드(<a href='/jobs/{id}'>) 안의
     <p> 텍스트에서 회사명·직무명을 먼저 따 둔다. 로켓펀치 카드 구조는
     PandaCSS atomic class 기반이지만 <p> 등장 순서는 일관적
     (1: 회사명, 2: 직무명, 3: 직군 부제).
  2) canonical `/jobs/{id}`로 이동해 detail 본문을 가져온다.
  3) <title>도 백업으로 파싱한다 ("직무명 채용공고 | 회사명 | 로켓펀치" 형식).
  4) 회사명·직무명을 본문 앞에 prepend해서 LLM이 놓치지 않도록 한다.

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

        # 1) listing+selectedJobId 페이지에 있다면, 좌측 카드에서 메타를 선추출.
        #    redirect 전에 따야 함 — 페이지를 떠나면 사라짐.
        selected_id: str | None = None
        if "selectedJobId=" in page.url:
            m = re.search(r"selectedJobId=(\d+)", page.url)
            if m:
                selected_id = m.group(1)
                self._extract_from_listing_card(page, selected_id, dom_data)

        # 2) canonical 상세 페이지로 이동
        if selected_id:
            canonical = f"https://www.rocketpunch.com/jobs/{selected_id}"
            logger.info(f"[rocketpunch] canonical로 이동: {canonical}")
            try:
                page.goto(canonical, wait_until="domcontentloaded", timeout=15000)
            except Exception as e:
                logger.warning(f"[rocketpunch] canonical 이동 실패 (계속 진행): {e}")

        time.sleep(1.5)

        # 3) <title> 파싱 (주로 "회사명 - 직무명 채용 | 로켓펀치" 형식)
        header_parts: list[str] = []
        try:
            page_title = page.title() or ""
            if page_title:
                header_parts.append(f"페이지 타이틀: {page_title}")
                segments = [s.strip() for s in page_title.split("|")]
                if segments:
                    # "페이타랩 - Windows Developer 채용" -> "페이타랩 - Windows Developer"
                    base_str = re.sub(r"\s*채용\s*공고\s*$", "", segments[0]).strip()
                    base_str = re.sub(r"\s*채용\s*$", "", base_str).strip()
                    
                    if " - " in base_str:
                        comp, pos = base_str.split(" - ", 1)
                        if not dom_data["company_name"]:
                            dom_data["company_name"] = comp.strip()
                        if not dom_data["position"]:
                            dom_data["position"] = pos.strip()
                    else:
                        if not dom_data["position"]:
                            dom_data["position"] = base_str

                # 혹시 "직무명 | 회사명 | 로켓펀치" 형식일 경우 대비
                if len(segments) >= 2 and segments[1] and segments[1] != "로켓펀치":
                    if not dom_data["company_name"]:
                        dom_data["company_name"] = segments[1]
        except Exception as e:
            logger.debug(f"[rocketpunch] page.title() 파싱 실패: {e}")

        # 4) 추출한 메타를 본문 앞에 명시적으로 prepend (LLM hallucination 방지)
        if dom_data["company_name"]:
            header_parts.append(f"회사명: {dom_data['company_name']}")
        if dom_data["position"]:
            header_parts.append(f"직무명: {dom_data['position']}")

        # 5) 본문 cascade (시맨틱 → body 폴백)
        content_locator = page.locator(
            ".job-detail, .position-detail, .content-container, article, main, body"
        ).first
        body_text = get_inner_text_safe(content_locator) or ""

        # 6) 합치기
        full_text = "\n".join(header_parts) + ("\n\n" + body_text if body_text else "")

        if full_text.strip():
            logger.info(
                f"[rocketpunch] 본문 추출 완료 "
                f"(헤더 {len(header_parts)}줄 + 본문 {len(body_text)}자, "
                f"position={dom_data['position']}, company={dom_data['company_name']})"
            )
            dom_data["full_text"] = full_text
        else:
            logger.warning("[rocketpunch] 본문 추출 실패")
            dom_data["full_text"] = ""

        return dom_data

    @staticmethod
    def _extract_from_listing_card(page, job_id: str, dom_data: dict) -> None:
        """리스트 페이지의 선택된 카드에서 회사명·직무명을 추출해 dom_data에 채운다.

        카드 구조:
            <a href='/jobs/{id}?list=true'>
              ...
              <p>{회사명}</p>      ← nth(0)
              <p>{직무명}</p>      ← nth(1) (BodyM_Bold)
              <p>{직군 부제}</p>   ← nth(2)
              ...                  ← 4번 이후는 "직군"/"숙련도"/"규모"/"근무 방식" 라벨
        """
        try:
            card = page.locator(f"a[href*='/jobs/{job_id}']").first
            if not card.is_visible(timeout=10000):
                logger.debug(f"[rocketpunch] 리스트 카드(job_id={job_id}) 보이지 않음")
                return

            ps = card.locator("p")
            count = ps.count()
            texts: list[str] = []
            for i in range(min(count, 3)):
                try:
                    t = ps.nth(i).inner_text(timeout=1500).strip()
                    if t:
                        texts.append(t)
                except Exception:
                    pass

            if len(texts) >= 1:
                dom_data["company_name"] = texts[0]
            if len(texts) >= 2:
                dom_data["position"] = texts[1]

            logger.info(
                f"[rocketpunch] 리스트 카드에서 메타 추출: "
                f"company={dom_data['company_name']}, position={dom_data['position']}"
            )
        except Exception as e:
            logger.debug(f"[rocketpunch] 리스트 카드 추출 실패 (계속 진행): {e}")

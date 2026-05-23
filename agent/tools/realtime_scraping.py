import logging
import urllib.parse
import numpy as np
from pathlib import Path
from playwright.sync_api import sync_playwright
from langchain_core.tools import tool

from shared.db.database import Database
from agent.utils.preprocessor import Preprocessor
from classic.automation.sites.wanted import WantedAdapter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

@tool
def realtime_scraping(company: str = None, tech_stack: str = None) -> str:
    """
    실시간으로 브라우저를 구동하여 특정 기업(company)이나 기술 스택(tech_stack)에 맞는
    최신 채용 공고를 수집하고 데이터베이스에 전처리 및 임베딩과 함께 적재(UPSERT)하는 도구입니다.
    데이터베이스에 정보가 없거나 부족할 때 호출되어 RAG 지식베이스를 동적으로 보강합니다.
    """
    search_keyword = ""
    if company and tech_stack:
        search_keyword = f"{company} {tech_stack}"
    elif company:
        search_keyword = company
    elif tech_stack:
        search_keyword = tech_stack
    else:
        return "수집 실패: 검색 키워드(company, tech_stack)가 모두 누락되었습니다."

    logger.info(f"[realtime_scraping] Starting Playwright crawling for keyword: '{search_keyword}'")
    
    from shared.config import DB_PATH
    db = Database(DB_PATH)
    
    # 1. Playwright 기동 및 검색 결과 URL 파싱
    encoded_keyword = urllib.parse.quote(search_keyword)
    search_url = f"https://www.wanted.co.kr/search?query={encoded_keyword}&tab=position"
    
    job_urls = []
    
    try:
        with sync_playwright() as p:
            # headless=True 권장 (가볍고 빠름)
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            page = context.new_page()
            
            logger.info(f"[realtime_scraping] Navigating to search URL: {search_url}")
            page.goto(search_url, wait_until="networkidle")
            
            # 카드 리스트 로딩 대기
            try:
                page.wait_for_selector("div[data-cy='job-card'] a, a[href*='/wd/']", timeout=10000)
            except Exception:
                logger.info(f"[realtime_scraping] No job cards found for search query '{search_keyword}'")
                browser.close()
                return f"실시간 수집 완료: '{search_keyword}'에 매칭되는 채용 공고 카드가 존재하지 않습니다."
            
            # 상위 3개 채용 카드 URL 긁기
            locators = page.locator("div[data-cy='job-card'] a, a[href*='/wd/']").all()
            for loc in locators:
                href = loc.get_attribute("href")
                if href and "/wd/" in href:
                    full_url = f"https://www.wanted.co.kr{href}" if href.startswith("/") else href
                    if full_url not in job_urls:
                        job_urls.append(full_url)
                    if len(job_urls) >= 3:
                        break
            
            logger.info(f"[realtime_scraping] Found {len(job_urls)} job URLs for scraping")
            
            # 2. 각 URL별 상세 내용 크롤링
            adapter = WantedAdapter()
            
            success_count = 0
            for url in job_urls:
                try:
                    logger.info(f"[realtime_scraping] Scraping detail URL: {url}")
                    page.goto(url, wait_until="networkidle")
                    
                    # 어댑터를 재사용해 본문 텍스트 추출
                    dom_data = adapter.extract(page)
                    full_text = dom_data.get("full_text", "")
                    company_name = dom_data.get("company_name", company or "")
                    position = dom_data.get("position", "")
                    
                    if not full_text:
                        logger.warning(f"[realtime_scraping] Skip {url}: No text body extracted")
                        continue
                    
                    # 3. 전처리 적용
                    cleaned_text = Preprocessor.clean_text(full_text)
                    normalized_stack = Preprocessor.normalize_tech_stack(position + " " + cleaned_text)
                    exp_min, exp_max, exp_text = Preprocessor.extract_experience(position, cleaned_text)
                    content_hash = Preprocessor.generate_content_hash(company_name, position, cleaned_text)
                    
                    # 4. 임베딩 계산 (768차원)
                    try:
                        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                        raw_vector = embeddings_model.embed_query(cleaned_text)
                    except Exception as embed_err:
                        logger.warning(f"[realtime_scraping] Embedding generation failed: {embed_err}. Fallback to zero vector.")
                        raw_vector = [0.0] * 768
                    embedding_bytes = np.array(raw_vector, dtype=np.float32).tobytes()
                    
                    # 5. DB Upsert
                    db_payload = {
                        "company_name": company_name,
                        "position": position,
                        "tech_stack": normalized_stack,
                        "raw_ocr_text": cleaned_text,
                        "source_platform": "Wanted",
                        "experience_min": exp_min,
                        "experience_max": exp_max,
                        "experience_text": exp_text,
                        "content_hash": content_hash
                    }
                    db.upsert(url=url, data=db_payload, embedding=embedding_bytes)
                    success_count += 1
                    
                except Exception as detail_err:
                    logger.error(f"[realtime_scraping] Error scraping detail {url}: {detail_err}")
                    continue
                    
            browser.close()
            
        if success_count > 0:
            return f"실시간 채용 공고 수집 및 적재 완료: 총 {success_count}건의 새로운 공고가 데이터베이스에 성공적으로 동적 업데이트되었습니다."
        else:
            return "실시간 수집 실패: 추출 가능한 유효한 채용 정보를 찾지 못했습니다."
            
    except Exception as e:
        logger.error(f"[realtime_scraping] Scraping error: {e}")
        return f"실시간 수집 오류: {e}"

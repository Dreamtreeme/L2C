"""
중앙 설정 관리 모듈
.env 파일에서 자동화/LLM/DB 설정을 로드합니다.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── 프로젝트 경로 ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / os.getenv("DB_PATH", "data/jobs.db")
SCREENSHOTS_DIR = BASE_DIR / os.getenv("SCREENSHOT_DIR", "data/screenshots")
JSON_DIR = BASE_DIR / os.getenv("JSON_OUTPUT_DIR", "data/json")
LOGS_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")

for d in (SCREENSHOTS_DIR, JSON_DIR, LOGS_DIR, DB_PATH.parent):
    d.mkdir(parents=True, exist_ok=True)

# ── 캡처 파라미터 (Playwright) ──────────────────────────────────────────
PLAYWRIGHT_HEADLESS = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true"
PLAYWRIGHT_TIMEOUT_MS = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "30000"))
CHROME_WINDOW_WIDTH = int(os.getenv("CHROME_WINDOW_WIDTH", "1024"))
CHROME_WINDOW_HEIGHT = int(os.getenv("CHROME_WINDOW_HEIGHT", "768"))
PAGE_LOAD_WAIT_SEC = float(os.getenv("PAGE_LOAD_WAIT_SEC", "4"))

# ── LLM (Ollama) ───────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "2048"))

# ── LLM 프롬프트 ───────────────────────────────────────────
EXTRACTION_PROMPT = """당신은 한국어 채용공고 데이터 추출기입니다.

제공된 웹 페이지 스크린샷 이미지를 분석하여 채용공고 정보를 추출하세요.
이미지에 보이는 모든 텍스트를 꼼꼼히 확인하고 다음 필드들에 해당하는 정보를 찾아야 합니다.

엄격한 규칙:
1. 출력은 반드시 한국어. 한자(汉字/漢字) 절대 사용 금지.
2. 이미지에 실제로 등장한 정보만 추출. 추측·환각 금지.
3. 정보가 없으면 해당 필드는 null. 문자열 "null"이 아니라 진짜 null.
4. 본문이 아닌 광고/푸터/사이드바 텍스트는 무시.
5. 마크다운, 설명, 코드블록 없이 순수 JSON만 반환.

추출할 필드 (스키마는 그대로 따를 것):
{
  "company_name": "회사명 (string|null)",
  "position": "포지션/직무명 (string|null)",
  "job_category": "직군 — 개발/디자인/마케팅 등 (string|null)",
  "experience_level": "경력 요건 — 신입/3년 이상/경력무관 등 (string|null)",
  "education": "학력 요건 (string|null)",
  "employment_type": "정규직/계약직/인턴 등 (string|null)",
  "location": "근무 위치 (string|null)",
  "deadline": "마감일 (string|null)",
  "tech_stack": ["기술스택 배열, 없으면 []"],
  "main_tasks": ["주요업무 섹션의 항목 배열"],
  "requirements": ["자격요건 섹션의 항목 배열"],
  "preferred": ["우대사항 섹션의 항목 배열"],
  "benefits": ["혜택 및 복지 섹션의 항목 배열"],
  "salary": "연봉 정보 (string|null)"
}

JSON만 출력하세요."""

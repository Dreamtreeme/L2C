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



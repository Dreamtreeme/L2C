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
JSON_DIR = BASE_DIR / os.getenv("JSON_OUTPUT_DIR", "data/json")
LOGS_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")

for d in (JSON_DIR, LOGS_DIR, DB_PATH.parent):
    d.mkdir(parents=True, exist_ok=True)

# ── 캡처 파라미터 (Playwright) ──────────────────────────────────────────
PLAYWRIGHT_HEADLESS = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true"
PLAYWRIGHT_TIMEOUT_MS = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "30000"))
CHROME_WINDOW_WIDTH = int(os.getenv("CHROME_WINDOW_WIDTH", "1024"))
CHROME_WINDOW_HEIGHT = int(os.getenv("CHROME_WINDOW_HEIGHT", "768"))
PAGE_LOAD_WAIT_SEC = float(os.getenv("PAGE_LOAD_WAIT_SEC", "4"))

# ── LLM (Ollama) ───────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b") # 시스템 내 가벼운 모델 선택
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "2048"))

# ── LLM 프롬프트 ───────────────────────────────────────────
EXTRACTION_PROMPT = """당신은 채용공고 텍스트 정제 전문가입니다.
제공된 채용공고 전문을 읽고, 아래 JSON 구조에 맞춰 정보를 추출하세요.

엄격한 규칙:
1. 반드시 JSON 형식만 출력하세요. (마크다운 코드 블록 제외)
2. 정보가 없는 필드는 반드시 null로 채우세요.
3. 모든 리스트 필드는 내용이 없으면 []로 채우세요.

추출할 JSON 구조:
{{
  "company_name": "회사명",
  "position": "직무명",
  "job_category": "직군",
  "experience_level": "경력요건",
  "education": "학력요건",
  "employment_type": "고용형태",
  "location": "근무지",
  "deadline": "마감일",
  "tech_stack": ["기술스택 리스트"],
  "main_tasks": ["주요업무 리스트"],
  "requirements": ["자격요건 리스트"],
  "preferred": ["우대사항 리스트"],
  "benefits": ["복리후생 리스트"],
  "salary": "연봉정보"
}}

채용공고 전문:
{text}"""



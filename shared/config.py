"""
중앙 설정 관리 모듈
.env 파일에서 자동화/LLM/DB 설정을 로드합니다.
"""

from __future__ import annotations

from agent.config import get_settings

_SETTINGS = get_settings()

# ── 프로젝트 경로 ──────────────────────────────────────────
DB_PATH = _SETTINGS.paths.db_path
JSON_DIR = _SETTINGS.paths.json_dir
LOGS_DIR = _SETTINGS.paths.log_dir
SCREENSHOT_DIR = _SETTINGS.paths.screenshot_dir

for d in (JSON_DIR, LOGS_DIR, SCREENSHOT_DIR, DB_PATH.parent):
    d.mkdir(parents=True, exist_ok=True)

# ── 캡처 파라미터 (Playwright) ──────────────────────────────────────────
PLAYWRIGHT_HEADLESS = _SETTINGS.browser.playwright_headless
PLAYWRIGHT_TIMEOUT_MS = _SETTINGS.browser.playwright_timeout_ms
CHROME_WINDOW_WIDTH = _SETTINGS.browser.chrome_window_width
CHROME_WINDOW_HEIGHT = _SETTINGS.browser.chrome_window_height
PAGE_LOAD_WAIT_SEC = _SETTINGS.browser.page_load_wait_sec

# ── LLM (Ollama) ───────────────────────────────────────────
OLLAMA_MODEL = _SETTINGS.models.ollama_model
OLLAMA_HOST = _SETTINGS.models.ollama_host
LLM_TEMPERATURE = _SETTINGS.models.llm_temperature

# ── LLM 프롬프트 ───────────────────────────────────────────
EXTRACTION_PROMPT = """당신은 채용공고 텍스트 정제 전문가입니다.
제공된 [채용공고 전문]을 꼼꼼히 읽고, 아래 [추출할 JSON 구조]에 맞춰 실제 데이터를 추출하세요.

엄격한 규칙:
1. 반드시 ```json 으로 시작하고 ```로 끝나는 마크다운 코드 블록 안에 JSON을 작성하세요.
2. 예시 구조를 그대로 복사하지 말고, [채용공고 전문]에서 찾은 실제 값을 각 필드에 채워넣으세요.
3. 정보가 명확히 없는 필드만 null 또는 빈 배열([])로 두세요.
4. 요약하지 말고 최대한 구체적으로 추출하세요.

[추출할 JSON 구조]:
{{
  "company_name": "회사명",
  "position": "직무명",
  "job_category": "직군",
  "experience_level": "경력요건",
  "education": "학력요건",
  "employment_type": "고용형태",
  "location": "근무지",
  "posted_at": "게시일(YYYY-MM-DD)",
  "posted_at_text": "게시일 원문",
  "deadline": "마감일",
  "tech_stack": ["기술스택 리스트"],
  "main_tasks": ["주요업무 리스트"],
  "requirements": ["자격요건 리스트"],
  "preferred": ["우대사항 리스트"],
  "benefits": ["복리후생 리스트"],
  "salary": "연봉정보"
}}

[채용공고 전문]:
{text}"""

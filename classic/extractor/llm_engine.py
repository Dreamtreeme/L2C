"""
텍스트 기반 LLM 정제 엔진.
추출된 본문 텍스트를 구조화된 JSON으로 변환합니다.
"""

from __future__ import annotations

import os
import json
import logging
import re
import time
from pathlib import Path

import ollama

from shared.config import (
    EXTRACTION_PROMPT,
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)
from shared.schema.jd_schema import JobPosting

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self):
        self.use_gemini = bool(os.getenv("GEMINI_API_KEY"))
        if not self.use_gemini:
            self._client = ollama.Client(host=OLLAMA_HOST)
            self._ensure_model(OLLAMA_MODEL)
        else:
            logger.info("GEMINI_API_KEY detected. LLMEngine will use Gemini 3.5 Flash.")

    def _ensure_model(self, model_name: str):
        try:
            res = self._client.list()
            # ollama 파이썬 클라이언트 버전에 따라 응답이 객체일 수도, 딕셔너리일 수도 있음
            models = getattr(res, 'models', [])
            if not models and isinstance(res, dict):
                models = res.get('models', [])
                
            available = []
            for m in models:
                if isinstance(m, dict):
                    available.append(m.get('model') or m.get('name'))
                else:
                    available.append(getattr(m, 'model', getattr(m, 'name', '')))

            if model_name not in available and f"{model_name}:latest" not in available:
                logger.info(f"모델 '{model_name}'이 존재하지 않습니다. 다운로드를 시작합니다...")
                self._client.pull(model_name)
                logger.info("모델 다운로드 완료.")
        except Exception as e:
            logger.warning(f"모델 확인 중 오류 발생: {e}")

    def extract_from_text(self, text: str) -> dict:
        """텍스트 전문 -> 채용공고 JSON."""
        if not text:
            return {}

        prompt = EXTRACTION_PROMPT.format(text=text)

        if getattr(self, "use_gemini", False):
            logger.info("[LLMEngine] 텍스트 정제 시작 (모델: gemini-3.5-flash)")
            t0 = time.time()
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=LLM_TEMPERATURE)
                response = llm.invoke(prompt)
                output = response.content
                if isinstance(output, list):
                    output = "\n".join(item if isinstance(item, str) else item.get("text", "") if isinstance(item, dict) else str(item) for item in output)
                elapsed = time.time() - t0
                logger.info(f"[LLMEngine] Gemini 정제 완료 ({elapsed:.1f}s)")
                logger.debug(f"LLM 원본 응답: {output[:300]}")
                parsed = self._parse_json(output)
                return self._validate(parsed)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.warning(f"Gemini extraction failed: {e}. Falling back to Ollama.")
                self.use_gemini = False
                self._client = ollama.Client(host=OLLAMA_HOST)
                self._ensure_model(OLLAMA_MODEL)

        # Qwen3 thinking 모드 비활성화 토큰.
        # user message 끝에 /no_think를 두면 Qwen3가 <think>...</think> 추론
        # 단계를 건너뛰고 바로 답한다. 로켓펀치 케이스에서 thinking + JSON 모드
        # 결합으로 응답이 170초까지 걸리고 모든 필드가 null로 빠진 적이 있어서
        # 안정성·속도 양쪽을 위해 끔.
        prompt += "\n\n/no_think"

        logger.info(f"[LLMEngine] 텍스트 정제 시작 (모델: {OLLAMA_MODEL})")
        t0 = time.time()

        # format="json"으로 Ollama JSON 모드 강제.
        # 토큰 디코딩 단계에서 유효한 JSON만 나오도록 제약을 걸어,
        # 모델이 마크다운 요약 모드로 빠지는 것을 방지한다.
        # (로켓펀치 첫 실행에서 Qwen3:8b가 프롬프트의 JSON 규칙을 무시하고
        #  '### 회사 소개' 같은 마크다운으로 응답해 모든 필드가 null이 된 케이스 대응)
        response = self._client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": LLM_TEMPERATURE},
        )
        
        elapsed = time.time() - t0
        output = response.get("message", {}).get("content", "")
        logger.info(f"[LLMEngine] 정제 완료 ({elapsed:.1f}s)")
        logger.debug(f"LLM 원본 응답: {output[:300]}")
        
        parsed = self._parse_json(output)
        return self._validate(parsed)

    @staticmethod
    def _parse_json(text: str) -> dict:
        if not text or not text.strip():
            logger.error("LLM 응답이 비어있습니다.")
            return {}

        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if m:
            json_str = m.group(1).strip()
        else:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = m.group(0) if m else text.strip()
        
        try:
            if not json_str:
                raise ValueError("추출된 JSON 문자열이 비어있습니다.")
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"JSON 파싱 실패: {e}\n원본 텍스트 일부: {text[:300]}")
            return {}

    # 스키마의 단일 string vs list 필드 분류 (LLM이 혼동하는 케이스를 흡수)
    _STRING_FIELDS = frozenset({
        "company_name", "position", "job_category", "experience_level",
        "education", "employment_type", "location", "deadline", "salary",
    })
    _LIST_FIELDS = frozenset({
        "tech_stack", "main_tasks", "requirements", "preferred", "benefits",
    })

    @classmethod
    def _normalize_types(cls, data: dict) -> dict:
        """LLM 출력의 타입 불일치를 스키마에 맞춰 강제 normalize.

        - String 필드가 list로 오면 ", ".join(...)
          예: experience_level=['신입', '미들', '시니어'] → '신입, 미들, 시니어'
        - List 필드가 string으로 오면 쉼표/세미콜론으로 split 후 list화
          예: tech_stack='Java, Spring' → ['Java', 'Spring']

        DB에는 string은 그대로, list는 JSON 직렬화돼 들어가야 하므로
        타입이 어긋나면 sqlite InterfaceError로 파이프라인이 죽는다.
        """
        for f in cls._STRING_FIELDS:
            v = data.get(f)
            if isinstance(v, list):
                joined = ", ".join(str(x).strip() for x in v if str(x).strip())
                data[f] = joined or None

        for f in cls._LIST_FIELDS:
            v = data.get(f)
            if isinstance(v, str):
                parts = [s.strip() for s in re.split(r"[,;]\s*", v) if s.strip()]
                data[f] = parts

        return data

    @classmethod
    def _validate(cls, data: dict) -> dict:
        normalized = cls._normalize_types(data)
        try:
            # 스키마에 맞춰 검증 및 정제
            return JobPosting(**normalized).model_dump()
        except Exception as e:
            logger.warning(f"스키마 검증 경고: {e}")
            return normalized

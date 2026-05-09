"""
텍스트 기반 LLM 정제 엔진.
추출된 본문 텍스트를 구조화된 JSON으로 변환합니다.
"""

from __future__ import annotations

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
        self._client = ollama.Client(host=OLLAMA_HOST)
        self._ensure_model(OLLAMA_MODEL)

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
        
        logger.info(f"[LLMEngine] 텍스트 정제 시작 (모델: {OLLAMA_MODEL})")
        t0 = time.time()
        
        response = self._client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
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

    @staticmethod
    def _validate(data: dict) -> dict:
        try:
            # 스키마에 맞춰 검증 및 정제
            return JobPosting(**data).model_dump()
        except Exception as e:
            logger.warning(f"스키마 검증 경고: {e}")
            return data

"""
Ollama 기반 LLM 정제 엔진.
(Vision LLM 활용하여 스크린샷 이미지에서 JSON 직접 추출)
"""

from __future__ import annotations

import base64
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
    _instance: "LLMEngine | None" = None

    def __new__(cls) -> "LLMEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
        return cls._instance

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        logger.debug(f"Ollama Client 생성 host={OLLAMA_HOST}")
        self._client = ollama.Client(host=OLLAMA_HOST)
        try:
            available = [m.model for m in self._client.list().models]
            logger.debug(f"Ollama 사용 가능 모델: {available}")
            found = any(
                OLLAMA_MODEL in name or name.startswith(OLLAMA_MODEL.split(":")[0])
                for name in available
            )
            if not found:
                logger.warning(f"모델 '{OLLAMA_MODEL}' 없음 → pull 시도")
                self._client.pull(OLLAMA_MODEL)
        except Exception as e:
            raise RuntimeError(
                f"Ollama 연결 실패: {e}\nOllama가 실행 중인지 확인: ollama serve"
            ) from e

    def extract(self, image_path: Path | str) -> dict:
        """스크린샷 이미지 → 채용공고 JSON."""
        self._ensure_client()
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        logger.info(
            f"[extract] model={OLLAMA_MODEL} 이미지={image_path.name} "
            f"prompt_length={len(EXTRACTION_PROMPT)} num_predict={LLM_NUM_PREDICT}"
        )
        
        # Read image to base64
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        t0 = time.time()
        response = self._client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT,
                    "images": [image_b64]
                }
            ],
            options={"num_predict": LLM_NUM_PREDICT, "temperature": LLM_TEMPERATURE},
        )
        elapsed = time.time() - t0
        output = response["message"]["content"]
        logger.info(f"[extract] LLM 응답 {len(output)}자 ({elapsed:.1f}s)")
        logger.debug(f"[extract] 응답 미리보기: {output[:300]}")
        parsed = self._parse_json(output)
        return self._validate(parsed)

    @staticmethod
    def _parse_json(text: str) -> dict:
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if m:
            json_str = m.group(1).strip()
        else:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = m.group(0) if m else text.strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}\n원본: {text}")
            raise RuntimeError(f"LLM 출력을 JSON으로 파싱할 수 없습니다: {e}") from e

    @staticmethod
    def _validate(data: dict) -> dict:
        nulls_normalized = 0
        for k, v in list(data.items()):
            if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
                data[k] = None
                nulls_normalized += 1
        if nulls_normalized:
            logger.debug(f"[_validate] {nulls_normalized}개 'null'/'none' 문자열을 None으로 변환")
        try:
            return JobPosting(**data).model_dump()
        except Exception as e:
            logger.warning(f"스키마 검증 경고 (원본 반환): {e}")
            return data

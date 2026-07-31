"""누적한 상세 OCR을 채용공고 스키마로 정제한다."""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.config import get_settings
from agent.prompts.detail_extraction import build_detail_extraction_system_prompt
from agent.runtime.detail_runtime import detail_buffer_text
from agent.utils.logger import logger
from agent.utils.model_dump import dump_model


class ExternalModelHTTPError(RuntimeError):
    """직접 HTTP 모델 호출의 상태 코드를 재시도 정책에 전달한다."""

    def __init__(self, status_code: int, payload: str):
        self.status_code = int(status_code)
        super().__init__(f"HTTP {self.status_code}: {payload}")


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


class OllamaDetailExtractionLLM:
    """Ollama JSON 모드로 누적 OCR을 정제한다."""

    def __init__(self, model_name: str):
        import ollama
        self.model_name = model_name
        self.client = ollama.Client(
            host=get_settings().models.ollama_host,
            timeout=get_settings().models.detail_openai_timeout_sec,
        )

    @staticmethod
    def _response_content(response: Any) -> str:
        if isinstance(response, dict):
            message = response.get("message") or {}
            return str(message.get("content") or "")
        message = getattr(response, "message", None)
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(getattr(message, "content", "") or "")

    def invoke(self, messages: list[Any]) -> Any:
        from json_repair import repair_json
        from shared.schema.jd_schema import JobPosting
        from agent.application.run_context import observe_external_llm_call

        system_text = "\n".join(
            _message_content(message)
            for message in messages
            if isinstance(message, SystemMessage)
        )
        user_text = "\n\n".join(
            _message_content(message)
            for message in messages
            if not isinstance(message, SystemMessage)
        )
        prompt = (
            f"{system_text}\n\n"
            "반드시 아래 JSON 스키마와 호환되는 JSON 객체 하나만 출력하십시오.\n"
            f"{json.dumps(JobPosting.model_json_schema(), ensure_ascii=False)}\n\n"
            f"{user_text}"
        )
        with observe_external_llm_call(
            component="detail_extraction",
            provider="ollama",
            model=self.model_name,
        ) as observation:
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "temperature": 0,
                    "num_predict": get_settings().models.detail_ollama_num_predict,
                },
            )
            observation.set_usage(
                {
                    "input_tokens": (
                        response.get("prompt_eval_count", 0)
                        if isinstance(response, dict)
                        else getattr(response, "prompt_eval_count", 0)
                    ),
                    "output_tokens": (
                        response.get("eval_count", 0)
                        if isinstance(response, dict)
                        else getattr(response, "eval_count", 0)
                    ),
                }
            )
        parsed = repair_json(self._response_content(response), return_objects=True)
        return JobPosting.model_validate(parsed if isinstance(parsed, dict) else {})


class OpenAIDetailExtractionLLM:
    """OpenAI Responses API로 누적 OCR을 정제한다."""

    def __init__(self, model_name: str):
        import requests
        configured_key = get_settings().models.openai_api_key
        api_key = configured_key.get_secret_value().strip() if configured_key else ""
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model_name = model_name
        self.api_key = api_key
        self.requests = requests

    @staticmethod
    def _output_text(response_json: dict[str, Any]) -> str:
        if isinstance(response_json.get("output_text"), str):
            return response_json["output_text"]
        parts: list[str] = []
        for item in response_json.get("output") or []:
            for content in item.get("content") or []:
                if content.get("type") == "output_text":
                    parts.append(str(content.get("text") or ""))
        return "\n".join(parts)

    @staticmethod
    def _job_posting_schema() -> dict[str, Any]:
        from shared.schema.jd_schema import JobPosting

        schema = JobPosting.model_json_schema()
        properties = dict(schema.get("properties") or {})
        for noisy_field in (
            "url",
            "source_platform",
            "raw_ocr_text",
            "content_hash",
            "evidence_hash",
        ):
            properties.pop(noisy_field, None)
        schema["properties"] = properties
        if isinstance(schema.get("required"), list):
            schema["required"] = [key for key in schema["required"] if key in properties]
        schema["additionalProperties"] = False
        return schema

    def invoke(self, messages: list[Any]) -> Any:
        from json_repair import repair_json
        from shared.schema.jd_schema import JobPosting
        from agent.application.run_context import observe_external_llm_call

        system_text = "\n".join(
            _message_content(message)
            for message in messages
            if isinstance(message, SystemMessage)
        )
        user_text = "\n\n".join(
            _message_content(message)
            for message in messages
            if not isinstance(message, SystemMessage)
        )
        system_prompt = build_detail_extraction_system_prompt(system_text)
        payload = {
            "model": self.model_name,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "max_output_tokens": get_settings().models.detail_openai_max_output_tokens,
            "temperature": 0,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "job_posting",
                    "schema": self._job_posting_schema(),
                    "strict": False,
                }
            },
            "store": False,
        }
        with observe_external_llm_call(
            component="detail_extraction",
            provider="openai",
            model=self.model_name,
        ) as observation:
            response = self.requests.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=get_settings().models.detail_openai_timeout_sec,
            )
            try:
                response_json = response.json()
            except ValueError:
                response_json = {"raw_text": response.text}
            observation.set_usage(response_json.get("usage") or {})
            observation.set_output(status_code=int(response.status_code))
            if response.status_code >= 400:
                raise ExternalModelHTTPError(
                    response.status_code,
                    json.dumps(response_json, ensure_ascii=False)[:2000],
                )
        parsed = repair_json(self._output_text(response_json), return_objects=True)
        return JobPosting.model_validate(parsed if isinstance(parsed, dict) else {})


def detail_extraction_model_spec() -> str:
    from agent.application.model_policy import lightweight_model_name

    return lightweight_model_name("VISION_DETAIL_FINAL_EXTRACTION_MODEL")


_detail_extraction_llm: Any = None
_detail_extraction_llm_key: str | None = None


def get_detail_extraction_llm() -> Any:
    """설정한 제공자의 상세 OCR 정제 모델을 지연 초기화한다."""

    global _detail_extraction_llm, _detail_extraction_llm_key
    model_spec = detail_extraction_model_spec()
    if _detail_extraction_llm is None or _detail_extraction_llm_key != model_spec:
        from agent.application.model_clients import get_structured_google_model
        from shared.schema.jd_schema import JobPosting

        if model_spec.startswith("ollama:"):
            model_name = model_spec.removeprefix("ollama:").strip()
            _detail_extraction_llm = OllamaDetailExtractionLLM(model_name)
        elif model_spec.startswith("openai:"):
            model_name = model_spec.removeprefix("openai:").strip()
            _detail_extraction_llm = OpenAIDetailExtractionLLM(model_name)
        else:
            _detail_extraction_llm = get_structured_google_model(
                model_spec,
                JobPosting,
                temperature=0.0,
                execution_role="detail",
            )
        _detail_extraction_llm_key = model_spec
    return _detail_extraction_llm


def clear_detail_extraction_model_cache() -> None:
    """애플리케이션 런타임 종료 시 상세 정제 모델 참조를 해제한다."""

    global _detail_extraction_llm, _detail_extraction_llm_key
    _detail_extraction_llm = None
    _detail_extraction_llm_key = None


def extract_job_from_job_detail_buffer(state: dict, current_url: str) -> dict[str, Any]:
    """상태의 OCR 버퍼를 공고 한 건으로 정제한다."""

    from agent.runtime.job_field_contract import (
        detail_coverage_status,
        required_fields_from_state,
    )

    buffer = dict(state.get("job_detail_buffer", {}) or {})
    ocr_text = detail_buffer_text(buffer)
    if not ocr_text.strip():
        return {}
    required_fields = required_fields_from_state(state)
    coverage = detail_coverage_status(
        dict(state.get("job_detail_coverage", {}) or {}),
        required_fields,
    )
    messages = [
        SystemMessage(
            content=(
                "누적 OCR 본문에서 채용공고 1건을 JobPosting 스키마로 정리하십시오. "
                "OCR에 없는 사실은 만들지 말고, 알 수 없는 필드는 비우십시오. "
                "required_fields는 반드시 OCR과 field_evidence를 확인해 채우십시오. "
                "카드 목록에서 추정한 회사명이나 직무명은 사실 근거로 사용하지 않습니다. "
                "company_name과 position은 상세 OCR의 헤더 또는 본문 근거를 우선하십시오. "
                "unavailable_fields는 공고가 제공하지 않는 것으로 이미 판정된 필드이므로 "
                "추측해서 채우지 마십시오. 현재 상세 URL은 보존하십시오."
            )
        ),
        HumanMessage(
            content=json.dumps(
                {
                    "current_url": current_url,
                    "required_fields": required_fields,
                    "field_evidence": coverage["field_evidence"],
                    "unavailable_fields": coverage[
                        "unavailable_fields"
                    ],
                    "ocr_text": ocr_text,
                },
                ensure_ascii=False,
                indent=2,
            )
        ),
    ]
    started = time.perf_counter()
    llm = get_detail_extraction_llm()
    if isinstance(llm, (OllamaDetailExtractionLLM, OpenAIDetailExtractionLLM)):
        from agent.application.run_context import (
            invoke_direct_model_with_policy,
        )

        response = invoke_direct_model_with_policy(
            llm,
            messages,
            "detail_extraction",
            execution_role="detail",
        )
    else:
        from agent.application.run_context import invoke_with_metrics

        response = invoke_with_metrics(
            llm,
            messages,
            "detail_extraction",
            stream=True,
        )
    extracted = dump_model(response)
    duration = time.perf_counter() - started
    logger.info(
        "Detail OCR final extraction completed",
        duration_sec=round(duration, 6),
        model=detail_extraction_model_spec(),
        ocr_chars=len(ocr_text),
        ocr_lines=len(buffer.get("lines") or []),
    )
    extracted = {key: value for key, value in extracted.items() if value not in (None, "", [], {})}
    if current_url and not extracted.get("url"):
        extracted["url"] = current_url
    # 구조화 결과와 별개로 실제 누적 OCR과 대표 화면을 출처 증거로 보존합니다.
    extracted["raw_ocr_text"] = ocr_text
    screens = [str(item) for item in (buffer.get("screens") or []) if str(item)]
    screen_evidence = [
        dict(item)
        for item in (buffer.get("screen_evidence") or [])
        if isinstance(item, dict) and str(item.get("path") or "").strip()
    ]
    representative = next(
        (
            str(item["path"])
            for item in screen_evidence
            if int(item.get("added_lines") or 0) > 0
        ),
        screens[0] if screens else "",
    )
    if representative:
        extracted["_evidence_screenshot_path"] = representative
    return extracted


__all__ = [
    "clear_detail_extraction_model_cache",
    "OllamaDetailExtractionLLM",
    "OpenAIDetailExtractionLLM",
    "detail_extraction_model_spec",
    "extract_job_from_job_detail_buffer",
    "get_detail_extraction_llm",
]

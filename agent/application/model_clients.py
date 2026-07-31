"""외부 모델 클라이언트를 프로세스 수명 동안 재사용한다."""

from __future__ import annotations

import threading
from typing import Any

from agent.config import get_settings


_LOCK = threading.RLock()
_GOOGLE_CLIENTS: dict[
    tuple[str, float, float | None, int | None, int | None, str | None], Any
] = {}
_GOOGLE_STRUCTURED_CLIENTS: dict[
    tuple[str, float, float | None, int | None, int | None, str | None, type], Any
] = {}


_MODELS_WITHOUT_SAMPLING_PARAMETERS = {
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
}
_MODEL_THINKING_LEVELS = {
    "gemini-3.6-flash": "medium",
    "gemini-3.5-flash-lite": "minimal",
}
_VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
_DEPRECATED_LATEST_MODEL_PARAMETERS = {
    "candidate_count",
    "temperature",
    "top_k",
    "top_p",
}


def _supports_sampling_parameters(model: str) -> bool:
    """신규 Gemini 모델의 폐기된 샘플링 인자를 전송하지 않는다."""

    return str(model).strip() not in _MODELS_WITHOUT_SAMPLING_PARAMETERS


def _normalized_temperature(model: str, temperature: float) -> float:
    """전송하지 않는 샘플링 값이 클라이언트 캐시를 분리하지 않게 한다."""

    return float(temperature) if _supports_sampling_parameters(model) else 0.0


def _resolved_max_output_tokens(
    model: str,
    max_output_tokens: int | None,
) -> int | None:
    """3.5 Lite의 과도한 생성만 기본 제한하고 명시 설정을 우선한다."""

    if max_output_tokens is not None:
        value = int(max_output_tokens)
        return value if value > 0 else None
    if model != "gemini-3.5-flash-lite":
        return None
    value = get_settings().models.lightweight_max_output_tokens
    return value if value > 0 else None


def _resolved_thinking_level(model: str, thinking_level: str | None) -> str | None:
    """명시한 사고 수준을 우선하고 모델별 공식 기본 정책을 적용한다."""

    value = str(thinking_level or _MODEL_THINKING_LEVELS.get(model) or "").strip().lower()
    if not value:
        return None
    if value not in _VALID_THINKING_LEVELS:
        raise ValueError(f"Unsupported thinking level: {value}")
    return value


def _latest_model_client_class(base_client_class: type) -> type:
    """LangChain이 다시 넣는 신규 모델 폐기 인자를 최종 요청에서 제거한다."""

    class LatestGeminiChatModel(base_client_class):
        def _build_base_generation_config(self, stop: Any, **kwargs: Any) -> dict[str, Any]:
            config = super()._build_base_generation_config(stop, **kwargs)
            for key in _DEPRECATED_LATEST_MODEL_PARAMETERS:
                config.pop(key, None)
            return config

    return LatestGeminiChatModel


def get_google_chat_model(
    model: str,
    *,
    temperature: float = 0.0,
    request_timeout: float | None = None,
    retries: int | None = None,
    max_output_tokens: int | None = None,
    thinking_level: str | None = None,
    execution_role: str | None = None,
) -> Any:
    """동일 설정의 Gemini 클라이언트를 한 번만 생성한다."""

    if execution_role:
        from agent.application.model_policy import model_execution_policy

        execution_policy = model_execution_policy(execution_role)
        if request_timeout is None:
            request_timeout = execution_policy.request_timeout_sec
        if retries is None:
            retries = execution_policy.retries
    normalized_timeout = None if request_timeout is None else float(request_timeout)
    normalized_retries = None if retries is None else max(0, int(retries))
    model_name = str(model).strip()
    normalized_max_output_tokens = _resolved_max_output_tokens(
        model_name,
        max_output_tokens,
    )
    normalized_thinking_level = _resolved_thinking_level(
        model_name,
        thinking_level,
    )
    key = (
        model_name,
        _normalized_temperature(model_name, temperature),
        normalized_timeout,
        normalized_retries,
        normalized_max_output_tokens,
        normalized_thinking_level,
    )
    with _LOCK:
        client = _GOOGLE_CLIENTS.get(key)
        if client is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            kwargs: dict[str, Any] = {"model": key[0]}
            configured_api_key = get_settings().models.gemini_api_key
            if configured_api_key is not None:
                kwargs["api_key"] = configured_api_key.get_secret_value()
            if _supports_sampling_parameters(key[0]):
                kwargs["temperature"] = key[1]
            if normalized_thinking_level:
                kwargs["thinking_level"] = normalized_thinking_level
            if normalized_timeout is not None:
                kwargs["request_timeout"] = normalized_timeout
            if normalized_retries is not None:
                kwargs["retries"] = normalized_retries
            if normalized_max_output_tokens is not None:
                # LangChain의 max_tokens 필드가 Gemini max_output_tokens로 변환됩니다.
                kwargs["max_tokens"] = normalized_max_output_tokens
            client_class = (
                _latest_model_client_class(ChatGoogleGenerativeAI)
                if key[0] in _MODELS_WITHOUT_SAMPLING_PARAMETERS
                else ChatGoogleGenerativeAI
            )
            client = client_class(**kwargs)
            _GOOGLE_CLIENTS[key] = client
        return client


def get_structured_google_model(
    model: str,
    schema: type,
    *,
    temperature: float = 0.0,
    request_timeout: float | None = None,
    retries: int | None = None,
    max_output_tokens: int | None = None,
    thinking_level: str | None = None,
    execution_role: str | None = None,
) -> Any:
    """동일 모델과 출력 스키마의 구조화 클라이언트를 재사용한다."""

    if execution_role:
        from agent.application.model_policy import model_execution_policy

        execution_policy = model_execution_policy(execution_role)
        if request_timeout is None:
            request_timeout = execution_policy.request_timeout_sec
        if retries is None:
            retries = execution_policy.retries
    normalized_timeout = None if request_timeout is None else float(request_timeout)
    normalized_retries = None if retries is None else max(0, int(retries))
    model_name = str(model).strip()
    normalized_max_output_tokens = _resolved_max_output_tokens(
        model_name,
        max_output_tokens,
    )
    normalized_thinking_level = _resolved_thinking_level(
        model_name,
        thinking_level,
    )
    key = (
        model_name,
        _normalized_temperature(model_name, temperature),
        normalized_timeout,
        normalized_retries,
        normalized_max_output_tokens,
        normalized_thinking_level,
        schema,
    )
    with _LOCK:
        client = _GOOGLE_STRUCTURED_CLIENTS.get(key)
        if client is None:
            client = get_google_chat_model(
                model_name,
                temperature=key[1],
                request_timeout=normalized_timeout,
                retries=normalized_retries,
                max_output_tokens=normalized_max_output_tokens,
                thinking_level=normalized_thinking_level,
                execution_role=execution_role,
            ).with_structured_output(schema)
            _GOOGLE_STRUCTURED_CLIENTS[key] = client
        return client


def clear_model_client_cache() -> None:
    """테스트와 명시적 런타임 재시작에서 모델 클라이언트 캐시를 비운다."""

    with _LOCK:
        _GOOGLE_STRUCTURED_CLIENTS.clear()
        _GOOGLE_CLIENTS.clear()


__all__ = [
    "clear_model_client_cache",
    "get_google_chat_model",
    "get_structured_google_model",
]

"""외부 모델 클라이언트를 프로세스 수명 동안 재사용한다."""

from __future__ import annotations

import threading
from typing import Any


_LOCK = threading.RLock()
_GOOGLE_CLIENTS: dict[tuple[str, float], Any] = {}
_GOOGLE_STRUCTURED_CLIENTS: dict[tuple[str, float, type], Any] = {}


def get_google_chat_model(model: str, *, temperature: float = 0.0) -> Any:
    """동일 설정의 Gemini 클라이언트를 한 번만 생성한다."""

    key = (str(model), float(temperature))
    with _LOCK:
        client = _GOOGLE_CLIENTS.get(key)
        if client is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            client = ChatGoogleGenerativeAI(model=key[0], temperature=key[1])
            _GOOGLE_CLIENTS[key] = client
        return client


def get_structured_google_model(
    model: str,
    schema: type,
    *,
    temperature: float = 0.0,
) -> Any:
    """동일 모델과 출력 스키마의 구조화 클라이언트를 재사용한다."""

    key = (str(model), float(temperature), schema)
    with _LOCK:
        client = _GOOGLE_STRUCTURED_CLIENTS.get(key)
        if client is None:
            client = get_google_chat_model(model, temperature=temperature).with_structured_output(schema)
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

"""구조화 모델 응답을 지정한 Pydantic 모델로 변환한다."""

from __future__ import annotations

from typing import Any


def parse_model_payload(value: Any, model_type: type) -> Any:
    if isinstance(value, model_type):
        return value
    if isinstance(value, dict):
        return model_type.model_validate(value)
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return model_type.model_validate_json(content)
    return model_type.model_validate(content)


__all__ = ["parse_model_payload"]

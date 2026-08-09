"""Pydantic 모델과 구조화 모델 응답의 변환 함수."""

from __future__ import annotations

from typing import Any, TypeVar


ModelT = TypeVar("ModelT")


def dump_model(model: Any) -> dict[str, Any]:
    """Pydantic 모델이나 dict를 일반 dict로 변환한다."""

    if model is None:
        return {}
    if isinstance(model, dict):
        return dict(model)
    return dict(model.model_dump())


def parse_model_payload(value: Any, model_type: type[ModelT]) -> ModelT:
    """구조화 모델 응답을 지정한 Pydantic 모델로 변환한다."""

    if isinstance(value, model_type):
        return value
    if isinstance(value, dict):
        return model_type.model_validate(value)
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return model_type.model_validate_json(content)
    return model_type.model_validate(content)


__all__ = ["dump_model", "parse_model_payload"]

"""Pydantic 모델과 dict 입력을 일관된 dict로 변환하는 헬퍼."""

from __future__ import annotations

from typing import Any


def dump_model(model: Any) -> dict[str, Any]:
    """Pydantic v1/v2 모델이나 dict를 일반 dict로 변환한다."""
    if model is None:
        return {}
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "model_dump"):
        return dict(model.model_dump())
    if hasattr(model, "dict"):
        return dict(model.dict())
    try:
        return dict(model)
    except (TypeError, ValueError):
        return {}

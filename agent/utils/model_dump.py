"""Pydantic 모델과 dict 입력을 일관된 dict로 변환하는 헬퍼."""

from __future__ import annotations

from typing import Any


def dump_model(model: Any) -> dict[str, Any]:
    """Pydantic v2 모델이나 dict를 일반 dict로 변환한다."""
    if model is None:
        return {}
    if isinstance(model, dict):
        return dict(model)
    return dict(model.model_dump())

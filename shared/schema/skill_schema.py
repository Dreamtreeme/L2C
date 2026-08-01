"""Reflex 경로 조회와 입력 치환에 필요한 메타데이터."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ReplayMode = Literal["fixed", "parameterized", "reasoning"]


class SkillInputSlot(BaseModel):
    """실행마다 바뀔 수 있는 입력 슬롯(input slot)."""

    name: str = Field("", description="슬롯 이름(slot name)")
    description: str = Field("", description="입력값 설명(input description)")
    observed_value: Any = Field(None, description="탐색 중 관찰된 값(observed value)")
    required: bool = Field(False, description="재생 시 필수 입력 여부(required)")
    source: str = Field("", description="값의 출처(source)")


class RecipeSkillMetadata(BaseModel):
    """활성 Reflex 경로를 찾고 입력값을 확인하는 메타데이터."""

    when_to_use: str = Field("", description="사용 조건(when to use)")
    site: str = Field("", description="대상 사이트(site)")
    task_category: str = Field("", description="작업 카테고리(task category)")
    inputs: list[SkillInputSlot] = Field(default_factory=list, description="가변 입력 목록(input slots)")

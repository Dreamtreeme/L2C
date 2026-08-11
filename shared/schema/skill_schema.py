"""Reflex 경로 조회와 입력 치환에 필요한 메타데이터."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ReplayMode = Literal["fixed", "parameterized", "reasoning"]
RecipeInputName = Literal["search_keyword"]
RECIPE_INPUT_NAMES = frozenset({"search_keyword"})


class SkillInputSlot(BaseModel):
    """재생 시 반드시 제공해야 하는 입력 이름."""

    model_config = ConfigDict(extra="forbid")

    name: RecipeInputName = Field(..., description="수집 요청에서 가져올 입력 이름")


class RecipeSkillMetadata(BaseModel):
    """활성 Reflex 경로를 찾고 입력값을 확인하는 메타데이터."""

    model_config = ConfigDict(extra="forbid")

    task_category: str = Field("", description="작업 카테고리(task category)")
    inputs: list[SkillInputSlot] = Field(default_factory=list, description="가변 입력 목록(input slots)")

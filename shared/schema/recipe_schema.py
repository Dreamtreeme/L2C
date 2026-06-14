"""
Reflex Recipe 스키마.
비전 ReAct 런에서 학습한 (화면-상태 -> 타깃) 매핑을 정형화한다.
DOM/Playwright 셀렉터/절대좌표가 아니라 OCR 마커 텍스트 공간에 머문다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecipeTarget(BaseModel):
    """클릭/입력 대상 마커의 디스크립터 (재생 시 현재 마커와 매칭하는 기준)."""

    text: str = Field("", description="정규화된 OCR 텍스트")
    semantic_label: Optional[str] = Field(None, description="LLM supplied title or label for the selected card/list item")
    region: Optional[str] = Field(None, description="화면 영역 힌트(타이브레이크용)")
    ordinal: Optional[int] = Field(None, description="동일 텍스트 다수일 때 순서 타이브레이크")
    evidence_texts: List[str] = Field(default_factory=list, description="Nearby OCR texts used as generic replay evidence")


class RecipeStep(BaseModel):
    """한 화면-상태에서 수행한 단일 액션 기록."""

    seq: int = Field(..., description="런 내 스텝 순번")
    state_key: str = Field(..., description="이 액션을 수행한 화면-상태 키")
    url_template: str = Field("", description="상태키의 URL 템플릿 성분")
    action: str = Field(..., description="click_marker/type_in_marker/scroll/press_key/open_browser/go_back")
    target: Optional[RecipeTarget] = Field(None, description="클릭/입력 대상(해당 시)")
    value: Optional[str] = Field(None, description="입력값/키/URL 등 부가 인자")
    param: Dict[str, Any] = Field(default_factory=dict, description="action_node에 다시 전달할 부가 인자")
    is_param: bool = Field(False, description="value가 사용자 목표에서 온 가변 파라미터인지")
    expected_next_state: Optional[str] = Field(None, description="이 액션 직후 기대 상태키(검증용)")


class SiteRecipe(BaseModel):
    """한 사이트에서 목표를 달성한 액션 시퀀스."""

    site: str = Field(..., description="netloc 기반 사이트 식별자")
    goal: str = Field("", description="이 런의 사용자 목표 원문")
    steps: List[RecipeStep] = Field(default_factory=list)
    success_count: int = Field(1, description="동일 경로 성공 누적 횟수")
    updated_at: str = Field("", description="마지막 학습 시각")

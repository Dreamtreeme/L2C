"""스킬형 메타데이터(Skill-style metadata) 스키마.

자율탐색(worker exploration)에서 나온 행동 기록을 재사용 가능한 작업 설명으로
검토할 수 있게 만드는 구조다. 이 스키마는 실행 판단을 직접 하지 않고,
지휘자/비평가 모델(Commander/Critic LLM)이 판단할 근거를 담는다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


ReplayMode = Literal["fixed", "parameterized", "reasoning"]


class SkillInputSlot(BaseModel):
    """실행마다 바뀔 수 있는 입력 슬롯(input slot)."""

    name: str = Field("", description="슬롯 이름(slot name)")
    description: str = Field("", description="입력값 설명(input description)")
    observed_value: Any = Field(None, description="탐색 중 관찰된 값(observed value)")
    required: bool = Field(False, description="재생 시 필수 입력 여부(required)")
    source: str = Field("", description="값의 출처(source)")


class SkillVerification(BaseModel):
    """재사용 성공/실패를 확인하는 검증 기준(verification criteria)."""

    success_signals: List[str] = Field(default_factory=list, description="성공 신호(success signals)")
    failure_signals: List[str] = Field(default_factory=list, description="실패 신호(failure signals)")
    fallback_conditions: List[str] = Field(default_factory=list, description="추론으로 넘길 조건(fallback conditions)")


class SkillStepIntent(BaseModel):
    """각 행동 단계가 가진 의도(step intent)."""

    seq: int = Field(0, description="행동 순서(step index)")
    action: str = Field("", description="도구 이름(tool action)")
    intent: str = Field("", description="행동 의도(intent)")
    target_role: str = Field("", description="대상 역할(target role)")
    component: str = Field("", description="화면 구성요소(component)")
    expected_after: str = Field("", description="행동 이후 기대 화면 변화(expected after)")
    fixed: bool | None = Field(None, description="고정 행동 여부(fixed step)")
    replay_mode: ReplayMode = Field(
        "reasoning",
        description="고정 재생, 입력 치환 재생, 매 실행 추론 중 하나(replay mode)",
    )
    slot_refs: List[str] = Field(default_factory=list, description="참조 입력 슬롯(slot references)")


class RecipeSkillMetadata(BaseModel):
    """반사 레시피(Reflex Recipe)에 붙는 스킬형 메타데이터."""

    when_to_use: str = Field("", description="사용 조건(when to use)")
    goal_pattern: str = Field("", description="목표 패턴(goal pattern)")
    site: str = Field("", description="대상 사이트(site)")
    page_type: str = Field("", description="화면 유형(page type)")
    inputs: List[SkillInputSlot] = Field(default_factory=list, description="가변 입력 목록(input slots)")
    step_intents: List[SkillStepIntent] = Field(default_factory=list, description="단계별 의도(step intents)")
    fixed_steps_summary: str = Field("", description="고정 단계 요약(fixed steps summary)")
    decision_points: List[str] = Field(default_factory=list, description="판단 지점(decision points)")
    verification: SkillVerification = Field(default_factory=SkillVerification, description="검증 기준(verification)")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="메타데이터 신뢰도(confidence)")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="검토 근거(evidence)")

"""반사 레시피(Reflex Recipe) 스키마.

타깃 주변 ROI pHash와 OCR 마커 비율을 기준으로 재생할 행동을 저장한다.
DOM, Playwright selector, 절대좌표를 저장하지 않고 화면 텍스트/마커 공간에 머문다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from shared.schema.skill_schema import RecipeSkillMetadata, ReplayMode


TransitionCueKind = Literal["text_any", "text_all", "slot_text", "min_text_markers"]


class TransitionCue(BaseModel):
    """OCR 결과에서 기계적으로 검사할 수 있는 화면 전환 단서."""

    kind: TransitionCueKind
    values: List[str] = Field(default_factory=list, description="검사할 OCR 텍스트 후보(values)")
    slot: str = Field("", description="실행 파라미터에서 가져올 입력 슬롯(slot)")
    min_count: int = Field(0, ge=0, description="최소 텍스트 마커 수(min count)")


class TransitionOutcome(BaseModel):
    """같은 행동 뒤에 발생할 수 있는 정상 결과 분기."""

    name: str
    cues: List[TransitionCue] = Field(default_factory=list)


class TransitionContract(BaseModel):
    """행동 이후 화면이 준비됐는지 판정하기 위한 구조화 계약."""

    common_ready_cues: List[TransitionCue] = Field(default_factory=list)
    outcomes: List[TransitionOutcome] = Field(default_factory=list)
    loading_cues: List[TransitionCue] = Field(default_factory=list)
    timeout_sec: float = Field(12.0, gt=0.0, le=60.0)


class RecipeTarget(BaseModel):
    """클릭/입력 대상 마커(target marker)를 다시 찾기 위한 정보."""

    text: str = Field("", description="정규화된 OCR 텍스트(OCR text)")
    semantic_label: Optional[str] = Field(None, description="LLM이 보정한 대상 라벨(semantic label)")
    region: Optional[str] = Field(None, description="화면 내 대략적 위치(region)")
    ordinal: Optional[int] = Field(None, description="동일 텍스트 중 화면 순서(ordinal)")
    evidence_texts: List[str] = Field(default_factory=list, description="주변 OCR 근거 텍스트(evidence texts)")
    marker_type: str = Field("", description="OCR/아이콘 마커 유형(marker type)")
    bbox_ratio: List[float] = Field(default_factory=list, description="화면 크기 대비 대상 bbox 비율(bbox ratio)")
    center_ratio: List[float] = Field(default_factory=list, description="화면 크기 대비 대상 중심 비율(center ratio)")


class RecipeStep(BaseModel):
    """ROI 검증 후 실행할 단일 행동 기록(recipe step)."""

    seq: int = Field(..., description="단계 순서(step index)")
    roi_signature: Dict[str, Any] = Field(default_factory=dict, description="타깃 주변 ROI pHash 서명(ROI signature)")
    url_template: str = Field("", description="URL 템플릿(url template)")
    page_role: str = Field("", description="행동을 기록한 화면 역할(page role)")
    observed_page_role: str = Field("", description="코드가 관찰한 화면 역할(observed page role)")
    declared_page_role: str = Field("", description="행동 계획이 선언한 화면 역할(declared page role)")
    action: str = Field(..., description="도구 이름(tool action)")
    target: Optional[RecipeTarget] = Field(None, description="클릭/입력 대상(target)")
    value: Optional[str] = Field(None, description="입력값 또는 부가 값(value)")
    param: Dict[str, Any] = Field(default_factory=dict, description="재생 시 넘길 인자(param)")
    is_param: bool = Field(False, description="값이 실행마다 바뀌는 파라미터인지 여부(is_param)")
    expected_after: str = Field("", description="행동 직후 기대 화면 변화(expected_after)")
    transition_contract: Optional[TransitionContract] = Field(
        None,
        description="행동 직후 OCR로 검사할 전환 계약(transition contract)",
    )
    intent: str = Field("", description="LLM이 남긴 행동 의도(intent)")
    target_role: str = Field("", description="대상 역할(target_role)")
    component: str = Field("", description="화면 구성요소(component)")
    slot_refs: List[str] = Field(default_factory=list, description="참조하는 입력 슬롯(slot_refs)")
    fixed: Optional[bool] = Field(None, description="고정 행동 여부(fixed)")
    replay_mode: ReplayMode = Field(
        "reasoning",
        description="이 단계를 그대로 재생할지, 파라미터화할지, 추론할지(replay mode)",
    )


class FollowupActionTrigger(BaseModel):
    """대상 좌표가 없는 후속 행동을 선택할 수 있는 직전 성공 문맥."""

    action: str = Field(..., description="직전에 성공한 행동(trigger action)")
    component: str = Field("", description="직전 행동의 화면 구성요소(trigger component)")
    page_role: str = Field("", description="직전 행동을 수행한 화면 역할(trigger page role)")


class FollowupActionStrategy(BaseModel):
    """직전 성공 문맥에 이어 실행할 좌표 없는 원자 행동 전략."""

    site: str = Field(..., description="사이트 식별자(site)")
    task_category: str = Field("", description="작업 카테고리(task category)")
    trigger: FollowupActionTrigger
    page_role: str = Field("", description="후속 행동을 수행할 화면 역할(page role)")
    url_template: str = Field("", description="후속 행동을 기록한 URL 템플릿(url template)")
    action: str = Field(..., description="후속 도구 이름(follow-up action)")
    param: Dict[str, Any] = Field(default_factory=dict, description="도구 실행 인자(action parameters)")
    expected_after: str = Field("", description="행동 뒤 기대 화면 변화(expected after)")
    transition_contract: TransitionContract
    success_count: int = Field(1, ge=0, description="성공 누적 횟수(success count)")


class SiteRecipe(BaseModel):
    """특정 사이트(site)와 목표(goal)에 대해 재사용할 원자 행동 레시피."""

    site: str = Field(..., description="사이트 식별자(site)")
    goal: str = Field("", description="학습 당시 사용자 목표(goal)")
    steps: List[RecipeStep] = Field(default_factory=list)
    skill_metadata: RecipeSkillMetadata = Field(default_factory=RecipeSkillMetadata)
    success_count: int = Field(1, description="성공 누적 횟수(success_count)")
    updated_at: str = Field("", description="마지막 갱신 시각(updated_at)")

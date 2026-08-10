"""경험 기반 탐색 레시피 스키마.

레시피는 단일 행동 목록이 아니라 검증 가능한 화면 상태 전이의 연속으로 저장한다.
DOM, Playwright selector, 절대좌표는 저장하지 않는다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from shared.schema.skill_schema import RecipeSkillMetadata, ReplayMode


class RecipeTarget(BaseModel):
    """클릭/입력 대상 마커(target marker)를 다시 찾기 위한 정보."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field("", description="정규화된 OCR 텍스트(OCR text)")
    semantic_label: Optional[str] = Field(None, description="LLM이 보정한 대상 라벨(semantic label)")
    region: Optional[str] = Field(None, description="화면 내 대략적 위치(region)")
    marker_type: str = Field("", description="OCR/아이콘 마커 유형(marker type)")
    bbox_ratio: List[float] = Field(default_factory=list, description="화면 크기 대비 대상 bbox 비율(bbox ratio)")
    center_ratio: List[float] = Field(default_factory=list, description="화면 크기 대비 대상 중심 비율(center ratio)")


class RecipeCheckpoint(BaseModel):
    """전이 전후에 다시 확인할 화면 상태."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(
        "",
        description="자율탐색 당시 화면 관찰 식별자(observation id)",
    )
    url_template: str = Field("", description="URL 템플릿(url template)")
    page_role: str = Field("", description="관찰된 화면 역할(page role)")
    screen_context_signature: Dict[str, Any] = Field(
        default_factory=dict,
        description="타깃이 없는 상태를 확인하는 화면 서명(screen context signature)",
    )
    anchor_target: Optional[RecipeTarget] = Field(
        None,
        description="해당 상태에서 확인할 다음 행동의 타깃(anchor target)",
    )
    anchor_roi_signature: Dict[str, Any] = Field(
        default_factory=dict,
        description="앵커 타깃 주변 ROI pHash 서명(anchor ROI signature)",
    )


class RecipeAction(BaseModel):
    """한 화면 관찰을 근거로 연속 실행할 수 있는 물리 행동."""

    model_config = ConfigDict(extra="forbid")

    source_seq: int = Field(..., description="자율탐색 원본 행동 번호(source sequence)")
    action: str = Field(..., description="도구 이름(tool action)")
    target: Optional[RecipeTarget] = Field(None, description="클릭/입력 대상(target)")
    roi_signature: Dict[str, Any] = Field(
        default_factory=dict,
        description="타깃 주변 ROI pHash 서명(ROI signature)",
    )
    value: Optional[str] = Field(None, description="입력값 또는 부가 값(value)")
    param: Dict[str, Any] = Field(default_factory=dict, description="재생 시 넘길 인자(param)")
    is_param: bool = Field(False, description="값이 실행마다 바뀌는 파라미터인지 여부(is_param)")
    intent: str = Field("", description="LLM이 남긴 행동 의도(intent)")
    target_role: str = Field("", description="대상 역할(target_role)")
    component: str = Field("", description="화면 구성요소(component)")
    slot_refs: List[str] = Field(default_factory=list, description="참조하는 입력 슬롯(slot_refs)")
    risk_level: str = Field(
        "",
        description="자율 탐색 실행 당시 위험도 선언(risk level)",
    )
    replay_mode: ReplayMode = Field(
        "reasoning",
        description="이 단계를 그대로 재생할지, 파라미터화할지, 추론할지(replay mode)",
    )


class RecipeTransition(BaseModel):
    """검증된 이전 상태에서 행동 묶음을 수행해 다음 상태로 이동하는 단위."""

    model_config = ConfigDict(extra="forbid")

    seq: int = Field(..., description="레시피 전이 순서(transition index)")
    before: RecipeCheckpoint
    actions: List[RecipeAction] = Field(min_length=1)
    after: RecipeCheckpoint
    expected_after: str = Field("", description="행동 후 기대 결과(expected after)")
    intent: str = Field("", description="전이 목적(transition intent)")


class RecipePath(BaseModel):
    """검증 가능한 시작·전이·완료 상태로 구성된 실행 경로."""

    model_config = ConfigDict(extra="forbid")

    start_state: RecipeCheckpoint
    transitions: List[RecipeTransition] = Field(min_length=1)
    completion_state: RecipeCheckpoint


class SiteRecipe(RecipePath):
    """특정 사이트와 목표에서 순서대로 재사용할 검증된 상태 전이 경로."""

    site: str = Field(..., description="사이트 식별자(site)")
    goal: str = Field("", description="학습 당시 사용자 목표(goal)")
    skill_metadata: RecipeSkillMetadata = Field(default_factory=RecipeSkillMetadata)
    support_count: int = Field(1, description="같은 경로를 지지한 자율탐색 후보 수")
    replay_success_count: int = Field(0, description="실제 재생 성공 횟수")
    replay_failure_count: int = Field(0, description="실제 재생 실패 횟수")
    updated_at: str = Field("", description="마지막 갱신 시각(updated_at)")

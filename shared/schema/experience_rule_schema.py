"""정제된 경험 규칙과 현재 화면에 해석된 실행 계약."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.schema.execution_record_schema import (
    ActionParameters,
    ActionTarget,
    PhysicalActionName,
)
from shared.schema.skill_schema import RecipeSkillMetadata


EffectKind = Literal[
    "url_change",
    "page_change",
    "target_region_change",
    "screen_change",
]


class RuleScreen(BaseModel):
    """규칙을 적용할 수 있는 화면의 저장된 기준."""

    model_config = ConfigDict(extra="forbid")

    url_template: str = ""
    page_role: str = ""
    reference_signature: dict[str, Any] = Field(default_factory=dict)


class RuleTarget(BaseModel):
    """현재 화면에서 다시 찾을 행동 대상의 원본 좌표와 ROI 근거."""

    model_config = ConfigDict(extra="forbid")

    reference: ActionTarget | None = None
    reference_roi_signature: dict[str, Any] = Field(default_factory=dict)


class RuleAction(BaseModel):
    """원본 행동에 출처를 둔 재사용 가능 행동."""

    model_config = ConfigDict(extra="forbid")

    source_seq: int
    action: PhysicalActionName
    target: RuleTarget | None = None
    param: ActionParameters = Field(default_factory=ActionParameters)
    input_slot: str = ""
    risk_level: str = ""


class ExpectedEffect(BaseModel):
    """규칙 행동이 성공했을 때 관찰되어야 하는 화면 변화."""

    model_config = ConfigDict(extra="forbid")

    kind: EffectKind
    description: str = ""
    expected_url_template: str = ""
    expected_page_role: str = ""
    reference_after_signature: dict[str, Any] = Field(default_factory=dict)
    target_region_ratio: list[float] = Field(default_factory=list)


class ExperienceRuleStep(BaseModel):
    """한 화면에서 연속 실행하고 결과를 검증할 경험 규칙 단계."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    source_transition_seqs: list[int] = Field(min_length=1)
    before: RuleScreen
    actions: list[RuleAction] = Field(min_length=1)
    intent: str
    expected_effect: ExpectedEffect
    source_node_id: str = ""


class ExperienceRuleNode(BaseModel):
    """하나의 의미 목적과 그 목적을 수행하는 물리 단계 목록."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    purpose: str
    source_event_seqs: list[int] = Field(min_length=1)
    step_ids: list[str] = Field(min_length=1)


class ExperienceRule(BaseModel):
    """사이트와 작업 목적에 맞춰 순서대로 적용하는 경험 규칙."""

    model_config = ConfigDict(extra="forbid")

    site: str
    goal: str = ""
    skill_metadata: RecipeSkillMetadata = Field(default_factory=RecipeSkillMetadata)
    steps: list[ExperienceRuleStep] = Field(min_length=1)
    nodes: list[ExperienceRuleNode] = Field(default_factory=list)
    support_count: int = 1
    replay_success_count: int = 0
    replay_failure_count: int = 0
    updated_at: str = ""


class InteractionRegionHandle(BaseModel):
    """현재 캡처에서만 유효한 물리 대상과 효과 검증 영역."""

    model_config = ConfigDict(extra="forbid")

    marker_id: int | None = None
    center_ratio: list[float] = Field(default_factory=list)
    bbox_ratio: list[float] = Field(default_factory=list)
    effect_region_ratio: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_physical_target(self) -> "InteractionRegionHandle":
        if self.marker_id is None and len(self.center_ratio) != 2:
            raise ValueError(
                "현재 대상에는 marker_id 또는 2차원 center_ratio가 필요합니다."
            )
        return self


class ResolvedRuleAction(BaseModel):
    """현재 캡처의 물리 대상으로 바인딩한 단일 행동."""

    model_config = ConfigDict(extra="forbid")

    source_seq: int
    action: PhysicalActionName
    target: InteractionRegionHandle | None = None
    param: ActionParameters = Field(default_factory=ActionParameters)
    risk_level: str = ""


class ResolvedRuleStep(BaseModel):
    """현재 관찰 한 장에서 실행할 수 있도록 해석된 경험 단계."""

    model_config = ConfigDict(extra="forbid")

    recipe_key: str
    step_index: int = Field(ge=0)
    observation_id: str
    actions: list[ResolvedRuleAction] = Field(min_length=1)
    expected_effect: ExpectedEffect


class ReplaySession(BaseModel):
    """현재 실행 중인 경험 규칙의 진행 위치."""

    model_config = ConfigDict(extra="forbid")

    recipe_key: str
    current_step_index: int = Field(ge=0)
    pending_step_index: int | None = Field(default=None, ge=0)
    step_count: int = Field(gt=0)

    def pending_is_current(self) -> bool:
        return self.pending_step_index == self.current_step_index

    def is_last_step(self) -> bool:
        return self.current_step_index + 1 >= self.step_count

    def advance(self) -> "ReplaySession | None":
        if not self.pending_is_current() or self.is_last_step():
            return None
        return self.model_copy(
            update={
                "current_step_index": self.current_step_index + 1,
                "pending_step_index": None,
            }
        )


__all__ = [
    "ExpectedEffect",
    "ExperienceRule",
    "ExperienceRuleNode",
    "ExperienceRuleStep",
    "InteractionRegionHandle",
    "ReplaySession",
    "ResolvedRuleAction",
    "ResolvedRuleStep",
    "RuleAction",
    "RuleScreen",
    "RuleTarget",
]

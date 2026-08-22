"""자율탐색 중 실제로 관찰하고 실행한 사실의 저장 계약."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.schema.skill_schema import RecipeInputName


PhysicalActionName = Literal[
    "click_marker",
    "type_in_marker",
    "press_key",
    "scroll",
    "go_back",
    "close_current_tab",
    "switch_tab",
    "open_browser",
]
ActionDirection = Literal[
    "down",
    "up",
    "left",
    "right",
    "next",
    "previous",
]
ScrollAmount = Literal["small", "page"]
ActionResultStatus = Literal["", "success", "error", "skipped"]
TransitionStatus = Literal[
    "",
    "idle",
    "waiting_capture",
    "pending",
    "needs_ocr",
    "ready",
    "unknown",
]

TARGET_ACTIONS = frozenset({"click_marker", "type_in_marker"})
COMMIT_ACTIONS = frozenset({"press_key"})
REVIEWABLE_ACTIONS = TARGET_ACTIONS | COMMIT_ACTIONS
NAVIGATION_ACTIONS = frozenset({"go_back", "close_current_tab", "switch_tab"})
TRAJECTORY_ACTIONS = REVIEWABLE_ACTIONS | NAVIGATION_ACTIONS | {"scroll"}
UI_ACTIONS = TRAJECTORY_ACTIONS | {"open_browser"}


class ActionParameters(BaseModel):
    """실제로 도구에 전달된 물리 행동 인자."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    slot_name: RecipeInputName | None = None
    key: str = ""
    direction: ActionDirection | None = None
    amount: ScrollAmount | None = None
    url: str = ""


class ActionTarget(BaseModel):
    """행동 당시 선택한 화면 대상의 스냅샷."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    semantic_label: str | None = None
    region: str | None = None
    marker_type: str = ""
    bbox_ratio: list[float] = Field(default_factory=list)
    center_ratio: list[float] = Field(default_factory=list)


class ScreenCheckpoint(BaseModel):
    """행동 전후에 실제로 캡처한 화면 상태."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = ""
    url_template: str = ""
    page_role: str = ""
    screen_context_signature: dict[str, Any] = Field(default_factory=dict)
    anchor_target: ActionTarget | None = None
    anchor_roi_signature: dict[str, Any] = Field(default_factory=dict)

    def has_anchor(self) -> bool:
        return self.anchor_target is not None and bool(self.anchor_roi_signature)

    def has_context_phash(self) -> bool:
        return bool(self.screen_context_signature.get("phash"))

    def same_observation_as(self, other: "ScreenCheckpoint") -> bool:
        return bool(
            self.observation_id
            and other.observation_id
            and self.observation_id == other.observation_id
        )


class ObservedAction(BaseModel):
    """한 화면 관찰을 근거로 실제 실행한 물리 행동."""

    model_config = ConfigDict(extra="forbid")

    source_seq: int
    action: PhysicalActionName
    target: ActionTarget | None = None
    roi_signature: dict[str, Any] = Field(default_factory=dict)
    param: ActionParameters = Field(default_factory=ActionParameters)
    intent: str = ""
    target_role: str = ""
    component: str = ""
    slot_refs: list[str] = Field(default_factory=list)
    risk_level: str = ""

    def parameter_slot(self) -> str:
        if self.param.slot_name:
            return self.param.slot_name
        return self.slot_refs[0] if self.slot_refs else ""


class ObservedTransitionEvidence(BaseModel):
    """전이가 실제로 일어났는지 판단할 때 사용한 실행·화면 근거."""

    model_config = ConfigDict(extra="forbid")

    source: str = ""
    result_status: ActionResultStatus = ""
    result_reason: str = ""
    status: TransitionStatus = ""
    outcome: str = ""
    reason: str = ""
    recipe_key: str = ""
    recipe_step_index: int | None = None
    recipe_step_count: int | None = None
    transition_actions: list[str] = Field(default_factory=list)
    after_state_match: dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    elapsed_sec: float = 0.0
    phash_distance: int | None = None
    visual_change_ratio: float | None = None
    ocr_skipped: bool = False
    before_marker_texts: list[str] = Field(default_factory=list)
    after_marker_texts: list[str] = Field(default_factory=list)
    screenshot: str = ""
    marked_image: str = ""


class ObservedTransition(BaseModel):
    """한 화면에서 행동 묶음을 수행해 다음 화면을 관찰한 원본 기록."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    before: ScreenCheckpoint
    actions: list[ObservedAction] = Field(min_length=1)
    after: ScreenCheckpoint
    expected_after: str = ""
    intent: str = ""
    evidence: ObservedTransitionEvidence | None = None


__all__ = [
    "ActionParameters",
    "ActionResultStatus",
    "ActionTarget",
    "COMMIT_ACTIONS",
    "NAVIGATION_ACTIONS",
    "ObservedAction",
    "ObservedTransition",
    "ObservedTransitionEvidence",
    "PhysicalActionName",
    "REVIEWABLE_ACTIONS",
    "ScreenCheckpoint",
    "TARGET_ACTIONS",
    "TRAJECTORY_ACTIONS",
    "TransitionStatus",
    "UI_ACTIONS",
]

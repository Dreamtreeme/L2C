"""자율탐색에서 기록하고 경험 기반 탐색에서 재사용하는 공통 계약."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from shared.schema.skill_schema import (
    RecipeInputName,
    RecipeSkillMetadata,
    ReplayMode,
)


PhysicalActionName = Literal[
    "click_marker",
    "focus_marker",
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

TARGET_REPLAY_ACTIONS = frozenset({"click_marker", "type_in_marker"})
RECIPE_COMMIT_ACTIONS = frozenset({"press_key"})
REVIEWABLE_REPLAY_ACTIONS = TARGET_REPLAY_ACTIONS | RECIPE_COMMIT_ACTIONS
NAVIGATION_ACTIONS = frozenset({"go_back", "close_current_tab", "switch_tab"})
TRAJECTORY_ACTIONS = REVIEWABLE_REPLAY_ACTIONS | NAVIGATION_ACTIONS | {"scroll"}
UI_ACTIONS = TRAJECTORY_ACTIONS | {"focus_marker", "open_browser"}


class ActionParameters(BaseModel):
    """기록 가능한 물리 행동의 제한된 인자 집합."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    slot_name: RecipeInputName | None = None
    key: str = ""
    direction: ActionDirection | None = None
    amount: ScrollAmount | None = None
    url: str = ""


class ActionTarget(BaseModel):
    """화면에서 물리 행동 대상을 다시 찾기 위한 정보."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    semantic_label: Optional[str] = None
    region: Optional[str] = None
    marker_type: str = ""
    bbox_ratio: List[float] = Field(default_factory=list)
    center_ratio: List[float] = Field(default_factory=list)


class ScreenCheckpoint(BaseModel):
    """행동 전후에 관찰한 화면 상태."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = ""
    url_template: str = ""
    page_role: str = ""
    screen_context_signature: Dict[str, Any] = Field(default_factory=dict)
    anchor_target: Optional[ActionTarget] = None
    anchor_roi_signature: Dict[str, Any] = Field(default_factory=dict)

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

class PhysicalAction(BaseModel):
    """한 화면 관찰을 근거로 실행한 물리 행동."""

    model_config = ConfigDict(extra="forbid")

    source_seq: int
    action: PhysicalActionName
    target: Optional[ActionTarget] = None
    roi_signature: Dict[str, Any] = Field(default_factory=dict)
    param: ActionParameters = Field(default_factory=ActionParameters)
    intent: str = ""
    target_role: str = ""
    component: str = ""
    slot_refs: List[str] = Field(default_factory=list)
    risk_level: str = ""
    replay_mode: ReplayMode = "reasoning"

    def has_replay_target(self) -> bool:
        return (
            self.action in TARGET_REPLAY_ACTIONS
            and self.target is not None
            and bool(self.roi_signature)
        )

    def parameter_slot(self) -> str:
        if self.param.slot_name:
            return self.param.slot_name
        return self.slot_refs[0] if self.slot_refs else ""

    def is_supported_replay_action(self) -> bool:
        if self.replay_mode == "reasoning":
            return False
        if self.action in TARGET_REPLAY_ACTIONS:
            if not self.has_replay_target():
                return False
            if self.action == "type_in_marker":
                if self.replay_mode == "parameterized":
                    return bool(
                        self.parameter_slot()
                        and self.parameter_slot() in self.slot_refs
                    )
                return bool(self.replay_mode == "fixed" and self.param.text)
            return self.replay_mode == "fixed"
        if self.action in RECIPE_COMMIT_ACTIONS:
            return bool(
                self.replay_mode == "fixed"
                and self.param.key.strip().casefold() in {"enter", "return"}
            )
        return False


class TransitionEvidence(BaseModel):
    """전이가 실제로 일어났는지 판단할 때 사용한 실행·화면 근거."""

    model_config = ConfigDict(extra="forbid")

    source: str = ""
    result_status: ActionResultStatus = ""
    result_reason: str = ""
    status: TransitionStatus = ""
    outcome: str = ""
    reason: str = ""
    recipe_key: str = ""
    recipe_transition_index: int | None = None
    recipe_transition_count: int | None = None
    transition_actions: List[str] = Field(default_factory=list)
    after_state_match: Dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    elapsed_sec: float = 0.0
    phash_distance: int | None = None
    visual_change_ratio: float | None = None
    ocr_skipped: bool = False
    before_marker_texts: List[str] = Field(default_factory=list)
    after_marker_texts: List[str] = Field(default_factory=list)
    screenshot: str = ""
    marked_image: str = ""


class ExperienceTransition(BaseModel):
    """한 화면에서 행동 묶음을 수행해 다음 화면으로 이동한 기록."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    before: ScreenCheckpoint
    actions: List[PhysicalAction] = Field(min_length=1)
    after: ScreenCheckpoint
    expected_after: str = ""
    intent: str = ""
    evidence: TransitionEvidence | None = None


class ExperiencePath(BaseModel):
    """검증 가능한 화면 전이를 순서대로 연결한 경험 경로."""

    model_config = ConfigDict(extra="forbid")

    transitions: List[ExperienceTransition] = Field(min_length=1)

    @property
    def start_state(self) -> ScreenCheckpoint:
        return self.transitions[0].before

    @property
    def completion_state(self) -> ScreenCheckpoint:
        return self.transitions[-1].after


class SiteExperience(ExperiencePath):
    """특정 사이트와 목표에서 재사용할 수 있도록 승격된 경험 경로."""

    site: str
    goal: str = ""
    skill_metadata: RecipeSkillMetadata = Field(default_factory=RecipeSkillMetadata)
    support_count: int = 1
    replay_success_count: int = 0
    replay_failure_count: int = 0
    updated_at: str = ""


class ReplaySession(BaseModel):
    """현재 실행 중인 경험 경로의 진행 위치."""

    model_config = ConfigDict(extra="forbid")

    recipe_key: str
    current_transition_index: int = Field(ge=0)
    pending_transition_index: int | None = Field(default=None, ge=0)
    transition_count: int = Field(gt=0)

    def pending_is_current(self) -> bool:
        return self.pending_transition_index == self.current_transition_index

    def is_last_transition(self) -> bool:
        return self.current_transition_index + 1 >= self.transition_count

    def advance(self) -> ReplaySession | None:
        if not self.pending_is_current() or self.is_last_transition():
            return None
        return self.model_copy(
            update={
                "current_transition_index": self.current_transition_index + 1,
                "pending_transition_index": None,
            }
        )


__all__ = [
    "ActionParameters",
    "ActionResultStatus",
    "ActionTarget",
    "NAVIGATION_ACTIONS",
    "ExperiencePath",
    "ExperienceTransition",
    "PhysicalAction",
    "PhysicalActionName",
    "RECIPE_COMMIT_ACTIONS",
    "REVIEWABLE_REPLAY_ACTIONS",
    "ReplaySession",
    "ScreenCheckpoint",
    "SiteExperience",
    "TARGET_REPLAY_ACTIONS",
    "TRAJECTORY_ACTIONS",
    "TransitionEvidence",
    "TransitionStatus",
    "UI_ACTIONS",
]

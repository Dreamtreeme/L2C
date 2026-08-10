"""피드백 루프(feedback loop) 스키마.

이 모델들은 자율 탐색 중 실제로 무슨 일이 있었는지를 기록한다.
행동이 영구적으로 재사용 가능한지는 코드가 결정하지 않고,
이후 지휘자/비평가(Commander/Critic)가 판단한다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from shared.schema.collection_intent import CollectionIntent

FeedbackLabel = Literal[
    "success",
    "partial",
    "no_effect",
    "error",
]


class ActionProposal(BaseModel):
    """실행 전 행위자가 제안한 행동(action proposal)."""

    model_config = ConfigDict(extra="forbid")

    action: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ActionObservation(BaseModel):
    """행동 실행 전후에 관찰된 사실(action observation)."""

    model_config = ConfigDict(extra="forbid")

    before: Dict[str, Any] = Field(default_factory=dict)
    after: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)


class ActionFeedback(BaseModel):
    """실행된 행동에 대한 1차 피드백(action feedback)."""

    model_config = ConfigDict(extra="forbid")

    label: FeedbackLabel
    reason: str = ""


class FeedbackEpisode(BaseModel):
    """제안 -> 실행 -> 관찰 -> 피드백 단위 기록(feedback episode)."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    proposal: ActionProposal
    observation: ActionObservation
    feedback: ActionFeedback


class RecordedRecipeStep(BaseModel):
    """자율탐색 중 재사용 후보로 기록한 물리 행동."""

    model_config = ConfigDict(extra="forbid")

    seq: int | None = None
    url_template: str = ""
    page_role: str = ""
    before_state: Dict[str, Any] = Field(default_factory=dict)
    action: str = ""
    target: Optional[Dict[str, Any]] = None
    roi_signature: Dict[str, Any] = Field(default_factory=dict)
    screen_context_signature: Dict[str, Any] = Field(default_factory=dict)
    value: Any = None
    param: Dict[str, Any] = Field(default_factory=dict)
    is_param: bool = False
    expected_after: str = ""
    intent: str = ""
    target_role: str = ""
    component: str = ""
    slot_refs: List[str] = Field(default_factory=list)
    risk_level: str = ""
    replay_mode: Literal["fixed", "parameterized", "reasoning"] = "reasoning"


class RecordedTransition(BaseModel):
    """행동 뒤 실제 화면 변화를 확인한 기록."""

    model_config = ConfigDict(extra="forbid")

    action_seq: int | None = None
    action: str = ""
    before_observation_id: str = ""
    after_observation_id: str = ""
    step: Dict[str, Any] = Field(default_factory=dict)
    expected_after: str = ""
    source: str = ""
    recipe_key: str = ""
    recipe_transition_index: int | None = None
    recipe_transition_count: int | None = None
    transition_actions: List[str] = Field(default_factory=list)
    after_state_match: Dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    elapsed_sec: float = 0.0
    status: str = ""
    outcome: str = ""
    reason: str = ""
    phash_distance: int | None = None
    visual_change_ratio: float | None = None
    ocr_skipped: bool = False
    marker_count: int = 0
    marker_texts: List[str] = Field(default_factory=list)
    screenshot: str = ""
    marked_image: str = ""
    current_url: str = ""
    page_role: str = ""
    after_state: Dict[str, Any] = Field(default_factory=dict)


class RecordedActionEvent(BaseModel):
    """행동 선택, 실행 결과와 화면 전환을 한 순번으로 묶은 기록."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    observation_id: str = ""
    result: Dict[str, Any] = Field(default_factory=dict)
    recipe_step: RecordedRecipeStep | None = None
    feedback_episode: FeedbackEpisode | None = None
    transition: RecordedTransition | None = None


ReviewDecision = Literal["accept", "revise", "reject"]


class WorkerSubmission(BaseModel):
    """하위 비전 작업자(child vision worker)가 지휘자에게 넘기는 제출물."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    goal: str = ""
    run_status: str = ""
    collected_count: int = 0
    observed_job_ids: List[int] = Field(default_factory=list)
    persisted_count: int = 0
    action_events: List[RecordedActionEvent] = Field(default_factory=list)
    collection_intent: CollectionIntent = Field(default_factory=CollectionIntent)
    extracted_summary: Dict[str, Any] = Field(default_factory=dict)

    @property
    def recorded_steps(self) -> List[RecordedRecipeStep]:
        return [event.recipe_step for event in self.action_events if event.recipe_step]

    @property
    def feedback_episodes(self) -> List[FeedbackEpisode]:
        return [
            event.feedback_episode
            for event in self.action_events
            if event.feedback_episode
        ]

    @property
    def transition_records(self) -> List[RecordedTransition]:
        return [event.transition for event in self.action_events if event.transition]


class StoredWorkerSubmission(BaseModel):
    """SQLite에 저장된 작업자 제출물과 조회 메타데이터."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    source: str
    payload: WorkerSubmission


class RecipeCandidate(BaseModel):
    """SQLite 검토 큐에서 복원한 자율탐색 레시피 후보."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str
    validation: Dict[str, Any] = Field(default_factory=dict)
    review_attempts: int = 0
    review_started_at: str | None = None
    next_review_at: str | None = None
    review_error: str = ""
    created_at: str = ""
    updated_at: str = ""
    run_id: str = ""
    source: str = ""
    submission: WorkerSubmission

    @property
    def site(self) -> str:
        return self.submission.collection_intent.site

    @property
    def goal(self) -> str:
        return self.submission.goal

    @property
    def keyword(self) -> str:
        return self.submission.collection_intent.search_keyword

    @property
    def steps(self) -> List[RecordedRecipeStep]:
        return self.submission.recorded_steps


class RecipeStepVerdict(BaseModel):
    """자율탐색이 제안한 재생 단계에 대한 비평가의 가지치기 판정."""

    seq: int
    keep: bool = False
    reason: str = ""


class RecipeCandidateReview(BaseModel):
    """반사 레시피 후보를 유지하거나 제거하는 비평가 판정."""

    decision: ReviewDecision
    reasons: List[str] = Field(default_factory=list)
    feedback_to_worker: str = ""
    step_verdicts: List[RecipeStepVerdict] = Field(default_factory=list)

"""자율탐색 실행 기록과 경험 경로 검토 계약."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.schema.collection_intent import CollectionIntent
from shared.schema.recipe_schema import (
    ExperienceTransition,
    PhysicalAction,
    ScreenCheckpoint,
)


class ExecutionEvent(BaseModel):
    """행동 선택부터 화면 전환 확인까지 한 순번으로 보관한 실행 기록."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    observation_id: str = ""
    result: Dict[str, Any] = Field(default_factory=dict)
    candidate_action: PhysicalAction | None = None
    before_checkpoint: ScreenCheckpoint | None = None
    before_marker_texts: List[str] = Field(default_factory=list)
    expected_after: str = ""
    intent: str = ""
    transition: ExperienceTransition | None = None


class WorkerSubmission(BaseModel):
    """하위 비전 작업자가 한 번의 실행을 마치고 제출한 기록."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    goal: str = ""
    run_status: str = ""
    collected_count: int = 0
    observed_job_ids: List[int] = Field(default_factory=list)
    persisted_count: int = 0
    action_events: List[ExecutionEvent] = Field(default_factory=list)
    collection_intent: CollectionIntent = Field(default_factory=CollectionIntent)
    extracted_summary: Dict[str, Any] = Field(default_factory=dict)

    @property
    def transitions(self) -> List[ExperienceTransition]:
        return [event.transition for event in self.action_events if event.transition]

    @property
    def actions(self) -> List[PhysicalAction]:
        return [
            action
            for transition in self.transitions
            for action in transition.actions
        ]


class StoredWorkerSubmission(BaseModel):
    """SQLite에 저장된 작업자 제출물과 조회 메타데이터."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    source: str
    payload: WorkerSubmission


class RecipeCandidate(BaseModel):
    """검토 큐에서 사용하는 자율탐색 경험 후보."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    contract_version: int = 3
    status: str
    validation: Dict[str, Any] = Field(default_factory=dict)
    review_attempts: int = 0
    review_started_at: str | None = None
    next_review_at: str | None = None
    review_error: str = ""
    created_at: str = ""
    updated_at: str = ""
    source: str = ""
    goal: str = ""
    run_status: str = ""
    collected_count: int = 0
    persisted_count: int = 0
    collection_intent: CollectionIntent = Field(default_factory=CollectionIntent)
    action_events: List[ExecutionEvent] = Field(default_factory=list)

    @classmethod
    def from_submission(
        cls,
        submission: WorkerSubmission,
        **metadata: Any,
    ) -> "RecipeCandidate":
        return cls(
            **metadata,
            goal=submission.goal,
            run_status=submission.run_status,
            collected_count=submission.collected_count,
            persisted_count=submission.persisted_count,
            collection_intent=submission.collection_intent,
            action_events=submission.action_events,
        )

    @property
    def site(self) -> str:
        return self.collection_intent.site

    @property
    def keyword(self) -> str:
        return self.collection_intent.search_keyword

    @property
    def transitions(self) -> List[ExperienceTransition]:
        return [event.transition for event in self.action_events if event.transition]

    @property
    def steps(self) -> List[PhysicalAction]:
        return [
            action
            for transition in self.transitions
            for action in transition.actions
        ]

    def transition_for_action(self, source_seq: int) -> ExperienceTransition | None:
        for transition in self.transitions:
            if any(action.source_seq == source_seq for action in transition.actions):
                return transition
        return None


ReviewDecision = Literal["accept", "reject"]


class RecipeTransitionVerdict(BaseModel):
    """기록된 행동 묶음 전이를 경험 경로에 남길지 결정한 판정."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    keep: bool = False
    reason: str = ""


class RecipeCandidateReview(BaseModel):
    """경험 후보의 전이를 유지하거나 제거하는 비평가 판정."""

    model_config = ConfigDict(extra="forbid")

    decision: ReviewDecision
    reasons: List[str] = Field(default_factory=list)
    feedback_to_worker: str = ""
    transition_verdicts: List[RecipeTransitionVerdict] = Field(
        default_factory=list
    )


__all__ = [
    "ExecutionEvent",
    "RecipeCandidate",
    "RecipeCandidateReview",
    "RecipeTransitionVerdict",
    "StoredWorkerSubmission",
    "WorkerSubmission",
]

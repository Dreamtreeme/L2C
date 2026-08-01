"""피드백 루프(feedback loop) 스키마.

이 모델들은 자율 탐색 중 실제로 무슨 일이 있었는지를 기록한다.
행동이 영구적으로 재사용 가능한지는 코드가 결정하지 않고,
이후 지휘자/비평가(Commander/Critic)가 판단한다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

FeedbackLabel = Literal[
    "success",
    "partial",
    "wrong_target",
    "no_effect",
    "loop_risk",
    "error",
]


class ActionProposal(BaseModel):
    """실행 전 행위자가 제안한 행동(action proposal)."""

    action: str
    args: Dict[str, Any] = Field(default_factory=dict)
    llm_thought: str = ""
    reason: str = ""
    target: Optional[Dict[str, Any]] = None
    target_label: Optional[str] = None
    component_candidate: Optional[str] = None
    target_role_candidate: Optional[str] = None
    expected_after: str = ""


class ActionObservation(BaseModel):
    """행동 실행 전후에 관찰된 사실(action observation)."""

    before: Dict[str, Any] = Field(default_factory=dict)
    after: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)


class ActionFeedback(BaseModel):
    """실행된 행동에 대한 1차 피드백(action feedback)."""

    label: FeedbackLabel
    reason: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class FeedbackEpisode(BaseModel):
    """제안 -> 실행 -> 관찰 -> 피드백 단위 기록(feedback episode)."""

    seq: int
    goal: str = ""
    site: str = ""
    proposal: ActionProposal
    observation: ActionObservation
    feedback: ActionFeedback


IssueSeverity = Literal["error", "warning"]
ReviewDecision = Literal["accept", "revise", "reject"]


class SubmissionIssue(BaseModel):
    """의미 판단 전 구조 검증에서 발견한 문제(submission issue)."""

    field: str
    reason: str
    severity: IssueSeverity = "error"


class WorkerSubmission(BaseModel):
    """하위 비전 작업자(child vision worker)가 지휘자에게 넘기는 제출물."""

    run_id: str = ""
    goal: str = ""
    site: str = ""
    task_category: str = ""
    keyword: str = ""
    run_status: str = ""
    is_finished: bool = False
    hit_recursion_limit: bool = False
    collected_count: int = 0
    observed_job_ids: List[int] = Field(default_factory=list)
    target_count: int = 0
    persisted_count: int = 0
    feedback_saved: int = 0
    recorded_steps: List[Dict[str, Any]] = Field(default_factory=list)
    feedback_episodes: List[Dict[str, Any]] = Field(default_factory=list)
    transition_records: List[Dict[str, Any]] = Field(default_factory=list)
    skill_metadata_evidence: Dict[str, Any] = Field(default_factory=dict)
    collection_intent: Dict[str, Any] = Field(default_factory=dict)
    semantic_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_summary: Dict[str, Any] = Field(default_factory=dict)
    worker_notes: str = ""


class CommanderReview(BaseModel):
    """작업 제출물에 대한 지휘자/비평가 판정(commander review)."""

    decision: ReviewDecision
    reasons: List[str] = Field(default_factory=list)
    feedback_to_worker: str = ""
    accept_collected_data: bool = False
    continue_collection: bool = False
    recipe_candidate: bool = False
    confidence: float = Field(0.0, ge=0.0, le=1.0)


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
    confidence: float = Field(0.0, ge=0.0, le=1.0)

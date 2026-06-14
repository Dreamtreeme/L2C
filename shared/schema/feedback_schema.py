"""Feedback-loop schemas for Reflex Recipe promotion.

These models describe what happened during exploration. They do not decide
whether an action is permanently replayable; later Critic/Memory code can use
these episodes to promote or reject candidate Reflex behavior.
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


class ParameterCandidate(BaseModel):
    """A value that may become a slot in a future recipe template."""

    slot_candidate: str = Field("", description="Suggested slot name, e.g. query/sample_count/site")
    value: Any = Field(None, description="Observed value for this candidate slot")
    reason: str = Field("", description="Why this value may be variable")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ActionProposal(BaseModel):
    """The actor's proposed action before execution."""

    action: str
    args: Dict[str, Any] = Field(default_factory=dict)
    llm_thought: str = ""
    target: Optional[Dict[str, Any]] = None
    target_label: Optional[str] = None
    component_candidate: Optional[str] = None
    target_role_candidate: Optional[str] = None
    parameter_candidates: List[ParameterCandidate] = Field(default_factory=list)
    fixed_candidate: Optional[bool] = None


class ActionObservation(BaseModel):
    """Facts observed around the action execution."""

    before: Dict[str, Any] = Field(default_factory=dict)
    after: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)


class ActionFeedback(BaseModel):
    """First-pass feedback label for an executed proposal."""

    label: FeedbackLabel
    reason: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class FeedbackEpisode(BaseModel):
    """One proposal -> action -> observation -> feedback record."""

    seq: int
    goal: str = ""
    site: str = ""
    page_state_key: str = ""
    proposal: ActionProposal
    observation: ActionObservation
    feedback: ActionFeedback

IssueSeverity = Literal["error", "warning"]
ReviewDecision = Literal["accept", "revise", "reject"]


class SubmissionIssue(BaseModel):
    """Shape-level issue found before semantic review."""

    field: str
    reason: str
    severity: IssueSeverity = "error"


class WorkerSubmission(BaseModel):
    """A child vision worker's structured handoff to the commander/critic layer."""

    run_id: str = ""
    goal: str = ""
    site: str = ""
    keyword: str = ""
    run_status: str = ""
    review_attempt: int = 0
    is_finished: bool = False
    hit_recursion_limit: bool = False
    collected_count: int = 0
    persisted_count: int = 0
    feedback_saved: int = 0
    recorded_steps: List[Dict[str, Any]] = Field(default_factory=list)
    feedback_episodes: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_summary: Dict[str, Any] = Field(default_factory=dict)
    worker_notes: str = ""


class CommanderReview(BaseModel):
    """Commander/Critic verdict for one worker submission."""

    decision: ReviewDecision
    reasons: List[str] = Field(default_factory=list)
    feedback_to_worker: str = ""
    recipe_candidate: bool = False
    confidence: float = Field(0.0, ge=0.0, le=1.0)
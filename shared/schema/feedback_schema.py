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
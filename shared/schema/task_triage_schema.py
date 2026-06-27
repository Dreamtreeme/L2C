"""Task triage and public research schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


GoalType = Literal["job_collection", "information_lookup", "account_action", "financial_action", "unknown"]
KnowledgeState = Literal["known", "unknown"]
RiskLevel = Literal["safe_read", "safe_navigation", "sensitive"]


class TaskTriage(BaseModel):
    """Top-level classification before an autonomous browser worker is allowed to act."""

    goal_type: GoalType = Field("unknown", description="High-level task category.")
    known_or_unknown: KnowledgeState = Field("unknown", description="Whether the system already has a known route.")
    risk_level: RiskLevel = Field("safe_read", description="Highest expected risk level.")
    requires_research: bool = Field(False, description="Whether public web research is required before execution.")
    sensitive_steps: list[str] = Field(default_factory=list, description="Steps requiring human confirmation.")
    reasons: list[str] = Field(default_factory=list, description="Short evidence for this triage decision.")


class ResearchPath(BaseModel):
    """One public route candidate found during pre-execution research."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    domain: str = ""
    official_hint: bool = False


class PublicResearchReport(BaseModel):
    """Public-web research summary used for route selection and worker constraints."""

    status: Literal["skipped", "completed", "failed"] = "skipped"
    query: str = ""
    meaning: str = ""
    possible_sites: list[ResearchPath] = Field(default_factory=list)
    official_paths: list[ResearchPath] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    sensitive_steps: list[str] = Field(default_factory=list)
    needs_user_choice: bool = False
    needs_user_confirmation: bool = False
    error: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


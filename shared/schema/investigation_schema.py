"""지휘자가 사용자 요청을 조사 가능한 계획으로 만드는 공통 계약."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class InvestigationPurpose(str, Enum):
    """사용자가 원하는 최종 결과의 종류."""

    LOOKUP = "lookup"
    COLLECT = "collect"
    COMPARE = "compare"
    TREND = "trend"


class InvestigationStatus(str, Enum):
    """조사 진행 상태."""

    UNDERSTANDING = "understanding"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    CHECKING_EVIDENCE = "checking_evidence"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    ANSWERING = "answering"
    COMPLETED = "completed"
    FAILED = "failed"


class ClarificationOption(BaseModel):
    """사용자가 선택할 수 있는 하나의 확정 값."""

    option_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    description: str = ""


class ClarificationQuestion(BaseModel):
    """한 가지 의미 단위를 확정하는 객관식 질문."""

    question_id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: list[ClarificationOption] = Field(default_factory=list)
    allow_custom: bool = True
    reason: str = ""

    @model_validator(mode="after")
    def validate_choices(self) -> "ClarificationQuestion":
        if not self.options and not self.allow_custom:
            raise ValueError("선택지 또는 직접 입력 중 하나는 허용해야 합니다.")
        option_ids = [item.option_id for item in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("선택지 식별자는 질문 안에서 중복될 수 없습니다.")
        return self


class ClarificationAnswer(BaseModel):
    """확인 질문에 대한 사용자의 구조화된 답변."""

    question_id: str = Field(min_length=1)
    selected_option_id: str = ""
    value: str = ""
    custom_value: str = ""

    @model_validator(mode="after")
    def validate_answer(self) -> "ClarificationAnswer":
        if not any((self.selected_option_id, self.value, self.custom_value)):
            raise ValueError("선택한 값 또는 직접 입력한 값이 필요합니다.")
        return self


class InvestigationConstraints(BaseModel):
    """사용자와의 대화를 거쳐 확정된 조사 조건."""

    search_keywords: list[str] = Field(default_factory=list)
    sites: list[str] = Field(default_factory=list)
    posted_from: str = ""
    posted_to: str = ""
    comparison_posted_from: str = ""
    comparison_posted_to: str = ""
    count_mode: str = "unspecified"
    target_count: int = Field(default=0, ge=0, le=100)
    location: str = ""
    experience: str = ""
    employment_type: str = ""
    analysis_dimensions: list[str] = Field(default_factory=list)


class EvidenceRequirement(BaseModel):
    """최종 답변을 뒷받침하기 위해 필요한 자료."""

    requirement_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    cohort: str = ""
    posted_from: str = ""
    posted_to: str = ""
    required_fields: list[str] = Field(default_factory=list)
    minimum_count: int = Field(default=1, ge=0, le=1000)
    required_sites: list[str] = Field(default_factory=list)
    reason: str = ""


class InvestigationPlanStep(BaseModel):
    """지휘자가 실행 전에 확정한 하나의 행동 단계."""

    step_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""
    expected_evidence: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    fallback: str = ""
    status: str = "pending"


class InvestigationRequest(BaseModel):
    """지휘자가 유지하는 사용자 요청의 전체 조사 상태."""

    investigation_id: str = Field(min_length=1)
    conversation_id: str = ""
    original_query: str = Field(min_length=1)
    objective: str = ""
    deliverable: str = ""
    purpose: InvestigationPurpose = InvestigationPurpose.LOOKUP
    status: InvestigationStatus = InvestigationStatus.UNDERSTANDING
    constraints: InvestigationConstraints = Field(default_factory=InvestigationConstraints)
    unresolved_fields: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    clarification_answers: list[ClarificationAnswer] = Field(default_factory=list)
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)
    evidence_snapshot: dict[str, Any] = Field(default_factory=dict)
    missing_evidence: list[str] = Field(default_factory=list)
    plan: list[InvestigationPlanStep] = Field(default_factory=list)
    executed_step_ids: list[str] = Field(default_factory=list)
    evidence_document_ids: list[int] = Field(default_factory=list)
    final_answer: str = ""


__all__ = [
    "ClarificationAnswer",
    "ClarificationOption",
    "ClarificationQuestion",
    "EvidenceRequirement",
    "InvestigationConstraints",
    "InvestigationPlanStep",
    "InvestigationPurpose",
    "InvestigationRequest",
    "InvestigationStatus",
]

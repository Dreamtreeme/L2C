"""지휘자가 사용자 요청을 조사 가능한 계획으로 만드는 공통 계약."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.schema.jd_schema import JobField
from shared.schema.collection_intent import CollectionIntent


class InvestigationPurpose(str, Enum):
    """사용자가 원하는 최종 결과의 종류."""

    LOOKUP = "lookup"
    COLLECT = "collect"
    COMPARE = "compare"
    TREND = "trend"


class EvidencePolicy(str, Enum):
    """질문에 답할 때 외부 근거를 확보하는 방식."""

    MODEL_KNOWLEDGE = "model_knowledge"
    DATABASE_FIRST = "database_first"
    WEB_REQUIRED = "web_required"
    DATABASE_ONLY = "database_only"


class InvestigationStatus(str, Enum):
    """조사 진행 상태."""

    UNDERSTANDING = "understanding"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    CHECKING_EVIDENCE = "checking_evidence"
    PLANNING = "planning"
    COLLECTING = "collecting"
    PERSISTING = "persisting"
    VALIDATING = "validating"
    ANSWERING = "answering"
    COMPLETED = "completed"
    FAILED = "failed"


class ConversationTurn(BaseModel):
    """조사 요청 해석에 참고할 이전 사용자·답변 한 쌍."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    user_query: str = ""
    assistant_answer: str = ""
    run_status: str = ""


class ClarificationOption(BaseModel):
    """사용자가 선택할 수 있는 하나의 확정 값."""

    model_config = ConfigDict(extra="forbid")

    option_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    collection_search_term: str = ""
    matching_count: int = Field(default=0, ge=0)
    concept_count: int = Field(
        default=0,
        ge=0,
        description="이 선택지 아래에 포함된 활성 직무 개념 수.",
    )
    description: str = ""


ClarificationField = Literal[
    "recent_period",
    "comparison_period",
    "site_scope",
    "target_count",
    "occupation_domain_concept_keys",
    "occupation_concept_keys",
    "occupation_query",
    "skill_queries",
    "analysis_dimensions",
    "sites",
    "posted_from",
    "posted_to",
    "location",
    "experience",
    "employment_type",
]


class ClarificationQuestion(BaseModel):
    """한 가지 의미 단위를 확정하는 객관식 질문."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    field: ClarificationField
    question: str = Field(min_length=1)
    options: list[ClarificationOption] = Field(default_factory=list)
    allow_custom: bool = True
    reason: str = ""
    candidate_count: int = Field(default=0, ge=0)
    concept_count: int = Field(default=0, ge=0)
    facet_type: str = ""

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

    model_config = ConfigDict(extra="forbid")

    occupation_scope_required: bool = Field(
        default=False,
        description=(
            "공고 조사 목표를 정하려면 사용자가 업무 영역이나 직무를 선택해야 하는지 여부."
        ),
    )
    occupation_domain_query: str = Field(
        default="",
        description=(
            "사용자가 명시한 업무 기능 기준의 넓은 직무 영역. 회사의 산업 분류와 구분한다."
        ),
    )
    occupation_domain_concept_keys: list[str] = Field(
        default_factory=list,
        description="검색 의미 사전에서 확정된 직무 영역 개념 키.",
    )
    occupation_query: str = Field(
        default="",
        description="사용자가 요청한 직무 표현. 사전 해석과 미분류 직무 판정에만 사용한다.",
    )
    collection_search_term: str = Field(
        default="",
        description="채용 사이트 검색창에 입력할 검색어. DB 후보 판정에는 사용하지 않는다.",
    )
    exact_text_groups: list[list[str]] = Field(
        default_factory=list,
        description=(
            "사용자가 문자열 자체를 조건으로 명시한 표현 묶음. 묶음 안은 OR, "
            "묶음 사이는 AND로 평가하며 직무 동의어 확장에는 사용하지 않는다."
        ),
    )
    occupation_concept_keys: list[str] = Field(
        default_factory=list,
        description="검색 의미 사전에서 확정된 직무 개념 키. 여러 값은 OR로 평가한다.",
    )
    occupation_resolution: Literal[
        "unresolved",
        "exact_alias",
        "user_selected",
        "reviewed_alias",
        "semantic_no_match",
    ] = Field(
        default="unresolved",
        description="직무 표현이 현재 개념 키로 확정된 경로.",
    )
    occupation_scope_mode: Literal["unspecified", "all", "selected"] = Field(
        default="unspecified",
        description=(
            "unspecified는 카디널리티 질문 가능, all은 사용자가 전체 하위 직무를 "
            "명시, selected는 객관식으로 범위를 확정한 상태다."
        ),
    )
    skill_queries: list[str] = Field(
        default_factory=list,
        description="사용자가 명시한 기술 표현. 활성 기술 사전으로 해석한다.",
    )
    skill_concept_keys: list[str] = Field(
        default_factory=list,
        description="검색 의미 사전에서 확정된 기술 개념 키.",
    )
    skill_match_mode: Literal["all", "any"] = "all"
    skill_requirement_type: Literal["any", "required", "preferred", "mentioned"] = "any"
    sites: list[str] = Field(default_factory=list)
    posted_from: str = ""
    posted_to: str = ""
    comparison_posted_from: str = ""
    comparison_posted_to: str = ""
    count_mode: Literal["unspecified", "explicit", "visible_all"] = "unspecified"
    target_count: int = Field(default=0, ge=0, le=100)
    location: str = ""
    experience: str = ""
    employment_type: str = ""
    analysis_dimensions: list[Annotated[str, Field(min_length=1, max_length=80)]] = (
        Field(default_factory=list, max_length=12)
    )

    @model_validator(mode="after")
    def normalize_occupation_scope_requirement(self) -> "InvestigationConstraints":
        has_occupation_scope = any(
            (
                self.occupation_domain_query.strip(),
                self.occupation_domain_concept_keys,
                self.occupation_query.strip(),
                self.occupation_concept_keys,
            )
        )
        if has_occupation_scope:
            self.occupation_scope_required = False
        return self

    @model_validator(mode="after")
    def validate_count_mode(self) -> "InvestigationConstraints":
        if self.count_mode == "explicit" and self.target_count == 0:
            raise ValueError("명시적 수집 개수에는 1 이상의 target_count가 필요합니다.")
        if self.count_mode != "explicit" and self.target_count > 0:
            raise ValueError(
                "target_count는 count_mode=explicit일 때만 사용할 수 있습니다."
            )
        return self


class EvidenceRequirement(BaseModel):
    """최종 답변을 뒷받침하기 위해 필요한 자료."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    occupation_domain_query: str = Field(
        default="",
        description="이 근거 집단이 속한 업무 기능 기준 직무 영역.",
    )
    occupation_domain_concept_keys: list[str] = Field(
        default_factory=list,
        description="이 근거 집단에 확정된 직무 영역 개념 키.",
    )
    occupation_query: str = Field(
        default="",
        description="이 근거 집단의 직무 표현. 코드가 직무 개념 키로 해석한다.",
    )
    occupation_concept_keys: list[str] = Field(
        default_factory=list,
        description="이 근거 집단에 확정된 직무 개념 키.",
    )
    collection_search_term: str = Field(
        default="",
        description="이 근거가 부족할 때 사이트 검색창에 입력할 검색어.",
    )
    skill_queries: list[str] = Field(default_factory=list)
    skill_concept_keys: list[str] = Field(default_factory=list)
    skill_match_mode: Literal["all", "any"] = "all"
    skill_requirement_type: Literal["any", "required", "preferred", "mentioned"] = "any"
    exact_text_groups: list[list[str]] = Field(
        default_factory=list,
        description=(
            "문자열 자체가 조건일 때만 사용하는 표현 묶음. 직무·기술 판정은 사전 키를 사용한다."
        ),
    )
    posted_from: str = ""
    posted_to: str = ""
    required_fields: list[JobField] = Field(default_factory=list)
    minimum_count: int = Field(default=1, ge=0, le=1000)
    required_sites: list[str] = Field(default_factory=list)
    reason: str = ""


class ToolCapability(BaseModel):
    """지휘자가 계획 전에 확인하는 도구 능력과 제약."""

    tool_name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    supported_operations: list[str] = Field(default_factory=list)
    supported_filters: dict[str, str] = Field(default_factory=dict)
    verifiable_fields: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    expected_latency: str = ""


class InvestigationPlanStep(BaseModel):
    """지휘자가 실행 전에 확정한 하나의 행동 단계."""

    step_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: CollectionIntent = Field(default_factory=CollectionIntent)
    purpose: str = ""
    expected_evidence: list[str] = Field(default_factory=list)


class InvestigationRequest(BaseModel):
    """지휘자가 유지하는 사용자 요청의 전체 조사 상태."""

    investigation_id: str = Field(min_length=1)
    conversation_id: str = ""
    original_query: str = Field(min_length=1)
    objective: str = ""
    deliverable: str = ""
    purpose: InvestigationPurpose = InvestigationPurpose.LOOKUP
    evidence_policy: EvidencePolicy = EvidencePolicy.DATABASE_FIRST
    status: InvestigationStatus = InvestigationStatus.UNDERSTANDING
    constraints: InvestigationConstraints = Field(
        default_factory=InvestigationConstraints
    )
    assumptions: list[str] = Field(default_factory=list)
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    clarification_answers: list[ClarificationAnswer] = Field(default_factory=list)
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)
    evidence_snapshot: dict[str, Any] = Field(default_factory=dict)
    missing_evidence: list[str] = Field(default_factory=list)
    plan: list[InvestigationPlanStep] = Field(default_factory=list)
    executed_step_ids: list[str] = Field(default_factory=list)
    evidence_document_ids: list[int] = Field(default_factory=list)
    collection_document_ids: list[int] = Field(default_factory=list)
    final_answer: str = ""


class RequestAnalysis(BaseModel):
    """도구를 사용하기 전에 수행하는 사용자 요청 해석 결과."""

    objective: str = Field(min_length=1)
    deliverable: str = Field(min_length=1)
    purpose: InvestigationPurpose = InvestigationPurpose.LOOKUP
    evidence_policy: EvidencePolicy = EvidencePolicy.DATABASE_FIRST
    constraints: InvestigationConstraints = Field(
        default_factory=InvestigationConstraints
    )
    assumptions: list[str] = Field(default_factory=list)
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)


class TaxonomyResolution(BaseModel):
    """선택된 직무 영역 안에서 사용자 표현을 사전 개념에 대응한 결과."""

    decision: Literal["match", "ambiguous", "no_match"] = "no_match"
    selected_concept_key: str = ""
    alternative_concept_keys: list[str] = Field(default_factory=list)
    reason: str = ""


class EvidencePlan(BaseModel):
    """확정된 요청에 답하기 위해 필요한 근거 목록."""

    requirements: list[EvidenceRequirement] = Field(default_factory=list)


class InvestigationActionPlan(BaseModel):
    """DB 근거가 부족할 때 실행할 도구 단계 목록."""

    steps: list[InvestigationPlanStep] = Field(default_factory=list)
    cannot_proceed_reason: str = ""


class RequirementEvidenceDecision(BaseModel):
    """한 근거 집단에 실제로 포함되는 DB 문서 판단."""

    requirement_id: str = Field(min_length=1)
    matching_document_ids: list[int] = Field(
        default_factory=list,
        description=(
            "사전으로 확정하지 못한 의미 조건을 만족하는 문서 ID. 제공된 후보 안에서만 반환한다."
        ),
    )


class EvidenceValidation(BaseModel):
    """의미 조건을 고려한 근거 집단 검증 결과."""

    decisions: list[RequirementEvidenceDecision] = Field(default_factory=list)


__all__ = [
    "ClarificationAnswer",
    "ClarificationField",
    "ClarificationOption",
    "ClarificationQuestion",
    "ConversationTurn",
    "EvidencePolicy",
    "EvidenceRequirement",
    "InvestigationConstraints",
    "InvestigationPlanStep",
    "InvestigationPurpose",
    "InvestigationRequest",
    "InvestigationStatus",
    "InvestigationActionPlan",
    "EvidencePlan",
    "RequestAnalysis",
    "TaxonomyResolution",
    "EvidenceValidation",
    "RequirementEvidenceDecision",
    "ToolCapability",
]

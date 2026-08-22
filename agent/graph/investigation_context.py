"""조사 그래프가 공유하는 상태, 모델과 요청 문맥 계약."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

from agent.config import get_settings
from agent.llm.clients import get_structured_google_model
from agent.llm.policy import commander_model_name
from agent.sites import list_supported_sites
from shared.schema.investigation_schema import (
    EvidencePlan,
    EvidenceRequirement,
    EvidenceValidation,
    GroundedAnswer,
    GroundedAnswerDraft,
    InvestigationActionPlan,
    InvestigationConstraints,
    InvestigationPlanStep,
    InvestigationRequest,
    RequestAnalysis,
)
from shared.schema.collection_intent import CollectionResult
from shared.schema.collection_run import CollectionBatch


class InvestigationRequestState(TypedDict, total=False):
    investigation: InvestigationRequest


class InvestigationEvidenceState(TypedDict, total=False):
    requirements: list[EvidenceRequirement]
    db_report: dict[str, Any]


class InvestigationExecutionState(TypedDict, total=False):
    plan: list[InvestigationPlanStep]
    executed_step_ids: list[str]
    collection_document_ids: list[int]
    collection_results: list[CollectionResult]
    pending_collection: CollectionBatch | None
    cannot_proceed_reason: str
    replan_attempted: bool


class InvestigationAnswerState(TypedDict, total=False):
    final_answer: str
    grounded_answer: GroundedAnswer | None
    run_status: str


def merge_investigation_section(
    current: Mapping[str, Any] | None,
    update: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {**dict(current or {}), **dict(update or {})}


class InvestigationState(TypedDict):
    request: Annotated[
        InvestigationRequestState,
        merge_investigation_section,
    ]
    evidence: Annotated[
        InvestigationEvidenceState,
        merge_investigation_section,
    ]
    execution: Annotated[
        InvestigationExecutionState,
        merge_investigation_section,
    ]
    answer: Annotated[
        InvestigationAnswerState,
        merge_investigation_section,
    ]


def create_investigation_state(
    investigation: InvestigationRequest,
) -> InvestigationState:
    """새 조사 실행에 필요한 네 책임 섹션을 초기화한다."""

    return {
        "request": {
            "investigation": investigation,
        },
        "evidence": {
            "requirements": [],
            "db_report": {},
        },
        "execution": {
            "plan": [],
            "executed_step_ids": [],
            "collection_document_ids": [],
            "collection_results": [],
            "pending_collection": None,
            "cannot_proceed_reason": "",
            "replan_attempted": False,
        },
        "answer": {
            "final_answer": "",
            "grounded_answer": None,
            "run_status": "running",
        },
    }


class InvestigationModels:
    """그래프 단계별 모델을 지연 생성하고 테스트에서 교체할 수 있게 한다."""

    def __init__(
        self,
        *,
        analysis_model: Any = None,
        evidence_model: Any = None,
        action_model: Any = None,
        validation_model: Any = None,
        answer_model: Any = None,
    ):
        self.analysis_model = analysis_model
        self.evidence_model = evidence_model
        self.action_model = action_model
        self.validation_model = validation_model
        self.answer_model = answer_model

    @staticmethod
    def _max_output_tokens() -> int | None:
        value = get_settings().models.commander_max_output_tokens
        return value if value > 0 else None

    def _structured(
        self,
        injected_model: Any,
        schema: type,
        *,
        thinking_level: str | None = None,
    ) -> Any:
        if injected_model is not None:
            return injected_model
        return get_structured_google_model(
            commander_model_name(),
            schema,
            temperature=0.0,
            max_output_tokens=self._max_output_tokens(),
            thinking_level=thinking_level,
            execution_role="commander",
        )

    def analysis(self) -> Any:
        return self._structured(self.analysis_model, RequestAnalysis)

    def evidence(self) -> Any:
        return self._structured(self.evidence_model, EvidencePlan)

    def action(self) -> Any:
        return self._structured(self.action_model, InvestigationActionPlan)

    def validation(self) -> Any:
        return self._structured(
            self.validation_model,
            EvidenceValidation,
            thinking_level="low",
        )

    def answer(self, *, thinking_level: str | None = None) -> Any:
        return self._structured(
            self.answer_model,
            GroundedAnswerDraft,
            thinking_level=thinking_level,
        )


def normalize_site_slugs(
    constraints: InvestigationConstraints,
) -> InvestigationConstraints:
    """모델이 표시명을 반환해도 사이트 레지스트리 slug로 정규화한다."""

    if not constraints.sites:
        return constraints
    aliases: dict[str, str] = {}
    for profile in list_supported_sites(enabled_only=False):
        values = [profile.slug, profile.display_name, *profile.domains]
        aliases.update(
            {
                value.strip().casefold(): profile.slug
                for value in values
                if value.strip()
            }
        )
    normalized = [
        aliases.get(str(value).strip().casefold(), str(value).strip())
        for value in constraints.sites
        if str(value).strip()
    ]
    return constraints.model_copy(update={"sites": list(dict.fromkeys(normalized))})


def collection_capabilities_for(
    catalog: list[dict[str, Any]],
    investigation: InvestigationRequest,
) -> list[dict[str, Any]]:
    """확정된 사이트가 있으면 해당 수집 능력만 LLM에 공개한다."""

    sites = {
        str(site).strip()
        for site in investigation.constraints.sites
        if str(site).strip()
    }
    if not sites:
        return list(catalog)
    return [
        item
        for item in catalog
        if str(item.get("tool_name") or "").partition(":")[2] in sites
    ]


def build_request_prompt_context(
    investigation: InvestigationRequest,
) -> dict[str, Any]:
    """LLM 단계에 전달할 요청 계약만 남기고 누적 실행 상태는 제외한다."""

    return {
        "investigation_id": investigation.investigation_id,
        "original_query": investigation.original_query,
        "objective": investigation.objective,
        "deliverable": investigation.deliverable,
        "purpose": investigation.purpose.value,
        "evidence_policy": investigation.evidence_policy.value,
        "constraints": investigation.constraints.model_dump(mode="json"),
        "assumptions": list(investigation.assumptions),
    }


__all__ = [
    "InvestigationState",
    "InvestigationModels",
    "build_request_prompt_context",
    "collection_capabilities_for",
    "create_investigation_state",
    "merge_investigation_section",
    "normalize_site_slugs",
]

"""조사 그래프가 공유하는 상태, 모델과 요청 문맥 계약."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

from agent.llm.clients import (
    get_google_chat_model,
    get_structured_google_model,
)
from agent.llm.policy import commander_model_name
from shared.schema.investigation_schema import (
    EvidencePlan,
    EvidenceValidation,
    InvestigationActionPlan,
    InvestigationConstraints,
    InvestigationRequest,
    RequestAnalysis,
    TaxonomyResolution,
)


class InvestigationRequestState(TypedDict, total=False):
    investigation: dict[str, Any]
    capability_catalog: list[dict[str, Any]]
    clarification: dict[str, Any]


class InvestigationEvidenceState(TypedDict, total=False):
    db_report: dict[str, Any]
    documents: list[dict[str, Any]]
    valid_ids: list[int]


class InvestigationExecutionState(TypedDict, total=False):
    collection_results: list[dict[str, Any]]
    run_status: str
    cannot_proceed_reason: str


class InvestigationAnswerState(TypedDict, total=False):
    final_answer: str


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
    capabilities: list[dict[str, Any]],
) -> InvestigationState:
    """새 조사 실행에 필요한 네 책임 섹션을 초기화한다."""

    return {
        "request": {
            "investigation": investigation.model_dump(mode="json"),
            "capability_catalog": list(capabilities),
            "clarification": {},
        },
        "evidence": {
            "db_report": {},
            "documents": [],
            "valid_ids": [],
        },
        "execution": {
            "collection_results": [],
            "run_status": "",
            "cannot_proceed_reason": "",
        },
        "answer": {"final_answer": ""},
    }


class InvestigationModels:
    """그래프 단계별 모델을 지연 생성하고 테스트에서 교체할 수 있게 한다."""

    def __init__(
        self,
        *,
        analysis_model: Any = None,
        taxonomy_model: Any = None,
        evidence_model: Any = None,
        action_model: Any = None,
        validation_model: Any = None,
        answer_model: Any = None,
    ):
        self.analysis_model = analysis_model
        self.taxonomy_model = taxonomy_model
        self.evidence_model = evidence_model
        self.action_model = action_model
        self.validation_model = validation_model
        self.answer_model = answer_model

    @staticmethod
    def _max_output_tokens() -> int | None:
        from agent.config import get_settings

        value = get_settings().models.commander_max_output_tokens
        return value if value > 0 else None

    def _structured(self, injected_model: Any, schema: type) -> Any:
        if injected_model is not None:
            return injected_model
        return get_structured_google_model(
            commander_model_name(),
            schema,
            temperature=0.0,
            max_output_tokens=self._max_output_tokens(),
            execution_role="commander",
        )

    def analysis(self) -> Any:
        return self._structured(self.analysis_model, RequestAnalysis)

    def evidence(self) -> Any:
        return self._structured(self.evidence_model, EvidencePlan)

    def taxonomy(self) -> Any:
        return self._structured(self.taxonomy_model, TaxonomyResolution)

    def action(self) -> Any:
        return self._structured(self.action_model, InvestigationActionPlan)

    def validation(self) -> Any:
        return self._structured(self.validation_model, EvidenceValidation)

    def answer(self) -> Any:
        if self.answer_model is not None:
            return self.answer_model
        return get_google_chat_model(
            commander_model_name(),
            temperature=0.0,
            max_output_tokens=self._max_output_tokens(),
            execution_role="commander",
        )


def message_text(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def normalize_site_slugs(
    constraints: InvestigationConstraints,
) -> InvestigationConstraints:
    """모델이 표시명을 반환해도 사이트 레지스트리 slug로 정규화한다."""

    if not constraints.sites:
        return constraints
    from agent.sites import list_supported_sites

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
    return constraints.model_copy(
        update={"sites": list(dict.fromkeys(normalized))}
    )


def capabilities_for_investigation(
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
        return catalog
    return [
        item
        for item in catalog
        if not str(item.get("tool_name") or "").startswith("realtime_scraping:")
        or str(item.get("tool_name") or "").split(":", 1)[1] in sites
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
        "evidence_requirements": [
            item.model_dump(mode="json")
            for item in investigation.evidence_requirements
        ],
        "missing_evidence": list(investigation.missing_evidence),
        "evidence_document_ids": list(investigation.evidence_document_ids),
        "collection_document_ids": list(investigation.collection_document_ids),
    }

__all__ = [
    "InvestigationState",
    "InvestigationModels",
    "build_request_prompt_context",
    "capabilities_for_investigation",
    "message_text",
    "create_investigation_state",
    "merge_investigation_section",
    "normalize_site_slugs",
]

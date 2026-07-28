"""조사 그래프가 공유하는 상태, 모델과 요청 문맥 계약."""

from __future__ import annotations

from typing import Any, TypedDict

from shared.schema.investigation_schema import (
    EvidencePlan,
    EvidenceValidation,
    InvestigationActionPlan,
    InvestigationRequest,
    RequestAnalysis,
    TaxonomyResolution,
)


class InvestigationGraphState(TypedDict, total=False):
    investigation: dict[str, Any]
    capability_catalog: list[dict[str, Any]]
    db_report: dict[str, Any]
    collection_results: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    valid_ids: list[int]
    clarification: dict[str, Any]
    final_answer: str
    run_status: str
    cannot_proceed_reason: str

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

    def analysis(self) -> Any:
        if self.analysis_model is None:
            from agent.application.model_clients import get_structured_google_model
            from agent.application.model_policy import commander_model_name

            self.analysis_model = get_structured_google_model(
                commander_model_name(),
                RequestAnalysis,
                temperature=0.0,
                max_output_tokens=self._max_output_tokens(),
            )
        return self.analysis_model

    def evidence(self) -> Any:
        if self.evidence_model is None:
            from agent.application.model_clients import get_structured_google_model
            from agent.application.model_policy import commander_model_name

            self.evidence_model = get_structured_google_model(
                commander_model_name(),
                EvidencePlan,
                temperature=0.0,
                max_output_tokens=self._max_output_tokens(),
            )
        return self.evidence_model

    def taxonomy(self) -> Any:
        if self.taxonomy_model is None:
            from agent.application.model_clients import get_structured_google_model
            from agent.application.model_policy import commander_model_name

            self.taxonomy_model = get_structured_google_model(
                commander_model_name(),
                TaxonomyResolution,
                temperature=0.0,
                max_output_tokens=self._max_output_tokens(),
            )
        return self.taxonomy_model

    def action(self) -> Any:
        if self.action_model is None:
            from agent.application.model_clients import get_structured_google_model
            from agent.application.model_policy import commander_model_name

            self.action_model = get_structured_google_model(
                commander_model_name(),
                InvestigationActionPlan,
                temperature=0.0,
                max_output_tokens=self._max_output_tokens(),
            )
        return self.action_model

    def validation(self) -> Any:
        if self.validation_model is None:
            from agent.application.model_clients import get_structured_google_model
            from agent.application.model_policy import commander_model_name

            self.validation_model = get_structured_google_model(
                commander_model_name(),
                EvidenceValidation,
                temperature=0.0,
                max_output_tokens=self._max_output_tokens(),
            )
        return self.validation_model

    def answer(self) -> Any:
        if self.answer_model is None:
            from agent.application.model_clients import get_google_chat_model
            from agent.application.model_policy import commander_model_name

            self.answer_model = get_google_chat_model(
                commander_model_name(),
                temperature=0.0,
                max_output_tokens=self._max_output_tokens(),
            )
        return self.answer_model

def parse_model_payload(value: Any, model_type: type) -> Any:
    if isinstance(value, model_type):
        return value
    if isinstance(value, dict):
        return model_type.model_validate(value)
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return model_type.model_validate_json(content)
    return model_type.model_validate(content)

def message_text(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")

def normalize_site_slugs(constraints):
    """모델이 표시명을 반환해도 사이트 레지스트리 slug로 정규화한다."""

    if not constraints.sites:
        return constraints
    try:
        from agent.sites import list_supported_sites

        aliases: dict[str, str] = {}
        for profile in list_supported_sites(enabled_only=False):
            slug = profile.slug
            values = [
                slug,
                profile.display_name,
                *(str(item) for item in profile.domains),
            ]
            for value in values:
                if value.strip():
                    aliases[value.strip().casefold()] = slug
        normalized = [
            aliases.get(str(value).strip().casefold(), str(value).strip())
            for value in constraints.sites
            if str(value).strip()
        ]
        return constraints.model_copy(update={"sites": list(dict.fromkeys(normalized))})
    except Exception:
        return constraints

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

def build_request_prompt_context(investigation: InvestigationRequest) -> dict[str, Any]:
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
    "InvestigationGraphState",
    "InvestigationModels",
    "build_request_prompt_context",
    "capabilities_for_investigation",
    "message_text",
    "normalize_site_slugs",
    "parse_model_payload",
]

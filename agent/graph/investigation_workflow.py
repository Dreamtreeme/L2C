"""확인, 근거 계획, 도구 실행을 분리한 최상위 조사 LangGraph."""

from __future__ import annotations

import json
import hashlib
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.application.clarification_service import apply_clarification_answer
from agent.application.evidence_service import inspect_job_evidence, load_job_evidence_documents
from agent.application.search_taxonomy_review_service import SearchTaxonomyReviewService
from agent.application.search_taxonomy_service import SearchTaxonomyService
from agent.application.run_context import emit_run_event, invoke_with_metrics, raise_if_cancelled
from agent.application.run_contracts import RunPhase, RunStatus
from agent.application.tool_capabilities import build_tool_capability_catalog
from agent.prompts.investigation import (
    action_plan_prompt,
    answer_prompt,
    evidence_plan_prompt,
    evidence_validation_prompt,
    request_analysis_prompt,
    taxonomy_resolution_prompt,
)
from agent.tools.realtime_scraping import realtime_scraping
from agent.runtime.investigation_checkpoint import InvestigationCheckpointRuntime
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    ClarificationOption,
    ClarificationQuestion,
    EvidencePolicy,
    EvidencePlan,
    EvidenceValidation,
    InvestigationActionPlan,
    InvestigationConstraints,
    InvestigationRequest,
    InvestigationStatus,
    RequestAnalysis,
    InvestigationPlanStep,
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


def _model_payload(value: Any, model_type: type) -> Any:
    if isinstance(value, model_type):
        return value
    if isinstance(value, dict):
        return model_type.model_validate(value)
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return model_type.model_validate_json(content)
    return model_type.model_validate(content)


def _message_text(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def _normalize_site_slugs(constraints):
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


def _capabilities_for_investigation(
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


def _request_prompt_context(investigation: InvestigationRequest) -> dict[str, Any]:
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


def _compact_db_report(report: dict[str, Any]) -> dict[str, Any]:
    """후보 판정이 끝난 뒤 계획과 답변에 필요한 근거 요약만 반환한다."""

    requirements = []
    for item in report.get("requirements", []) or []:
        if not isinstance(item, dict):
            continue
        requirements.append(
            {
                key: value
                for key, value in item.items()
                if key != "candidates"
            }
        )
    return {
        "total_db_rows": int(report.get("total_db_rows") or 0),
        "requirements": requirements,
        "sufficient": bool(report.get("sufficient", False)),
        "document_ids": list(report.get("document_ids") or []),
        "missing_evidence": list(report.get("missing_evidence") or []),
    }


def _evidence_validation_payload(
    report: dict[str, Any],
    investigation: InvestigationRequest,
) -> list[dict[str, Any]]:
    """직무 판정에 필요한 후보 메타데이터만 모델에 전달한다."""

    requirements = {
        item.requirement_id: item for item in investigation.evidence_requirements
    }
    candidate_fields = (
        "document_id",
        "position",
        "job_category",
        "experience",
        "employment_type",
        "location",
        "posted_at",
        "source_platform",
        "tech_stack",
        "requirements",
        "preferred",
        "field_presence",
    )
    groups = []
    for item in report.get("requirements", []) or []:
        if not isinstance(item, dict):
            continue
        requirement = requirements.get(str(item.get("requirement_id") or ""))
        if requirement is None or not item.get("semantic_review_required"):
            continue
        groups.append(
            {
                "requirement_id": requirement.requirement_id,
                "description": requirement.description,
                "occupation_query": requirement.occupation_query,
                "skill_queries": list(requirement.skill_queries),
                "exact_text_groups": list(requirement.exact_text_groups),
                "minimum_count": requirement.minimum_count,
                "required_fields": list(requirement.required_fields),
                "required_sites": list(requirement.required_sites),
                "posted_from": requirement.posted_from,
                "posted_to": requirement.posted_to,
                "candidates": [
                    {
                        key: candidate.get(key)
                        for key in candidate_fields
                        if key in candidate
                    }
                    for candidate in item.get("candidates", [])
                    if isinstance(candidate, dict)
                ],
            }
        )
    return groups


def _compact_collection_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """답변에 영향을 주는 수집 결과와 저장 문서만 남긴다."""

    compact = []
    keys = (
        "message",
        "site",
        "site_name",
        "keyword",
        "target_count",
        "item_count",
        "persisted_count",
        "completion_status",
        "search_scope_exhausted",
        "missing_count",
        "observed_job_ids",
    )
    for result in results or []:
        if not isinstance(result, dict):
            continue
        item = {key: result.get(key) for key in keys if key in result}
        validation = result.get("persistence_validation")
        if isinstance(validation, dict):
            item["persistence_validation"] = {
                "created_count": int(validation.get("created_count") or 0),
                "updated_count": int(validation.get("updated_count") or 0),
                "rejected_count": int(validation.get("rejected_count") or 0),
                "persisted_items": [
                    {
                        key: persisted.get(key)
                        for key in ("job_id", "company_name", "position", "operation")
                        if key in persisted
                    }
                    for persisted in (validation.get("persisted_items") or [])
                    if isinstance(persisted, dict)
                ],
            }
        compact.append(item)
    return compact


def _answer_evidence_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """구조화 상세 필드가 완전한 문서는 중복 OCR 원문을 답변 입력에서 제외한다."""

    detail_fields = ("tech_stack", "main_tasks", "requirements", "preferred", "benefits")
    projected = []
    for document in documents or []:
        if not isinstance(document, dict):
            continue
        item = dict(document)
        if all(str(item.get(field) or "").strip() for field in detail_fields):
            item.pop("raw_ocr_text", None)
        projected.append(item)
    return projected


def _normalized_collection_steps(
    plan: InvestigationActionPlan,
    investigation: InvestigationRequest,
    capability_catalog: list[dict[str, Any]],
) -> list[InvestigationPlanStep]:
    """LLM이 선택한 단계에 이미 확정된 요청 조건을 빠짐없이 전달한다."""

    allowed_sites = {
        str(item.get("tool_name") or "").split(":", 1)[1]
        for item in capability_catalog
        if str(item.get("tool_name") or "").startswith("realtime_scraping:")
    }
    requirements = {
        item.requirement_id: item for item in investigation.evidence_requirements
    }
    normalized: list[InvestigationPlanStep] = []
    signatures: set[str] = set()
    maximum_steps = len(allowed_sites) * max(1, len(requirements))
    for step in plan.steps:
        tool_name = str(step.tool_name or "")
        if tool_name != "realtime_scraping" and not tool_name.startswith(
            "realtime_scraping:"
        ):
            continue
        arguments = step.arguments.model_dump(mode="json")
        site_from_tool = tool_name.split(":", 1)[1] if ":" in tool_name else ""
        site = str(arguments.get("site") or site_from_tool).strip()
        if not site and len(investigation.constraints.sites) == 1:
            site = investigation.constraints.sites[0]
        if not site or site not in allowed_sites:
            continue

        requirement = next(
            (
                requirements[requirement_id]
                for requirement_id in step.expected_evidence
                if requirement_id in requirements
            ),
            None,
        )
        query = str(
            arguments.get("query")
            or (requirement.collection_search_term if requirement else "")
            or (requirement.occupation_query if requirement else "")
            or investigation.constraints.collection_search_term
            or investigation.constraints.occupation_query
        ).strip()
        if not query:
            continue
        posted_from = str(
            arguments.get("posted_from")
            or (requirement.posted_from if requirement else "")
            or investigation.constraints.posted_from
        )
        posted_to = str(
            arguments.get("posted_to")
            or (requirement.posted_to if requirement else "")
            or investigation.constraints.posted_to
        )
        arguments.update(
            {
                "query": query,
                "site": site,
                "original_query": investigation.original_query,
                "count_mode": investigation.constraints.count_mode,
                "target_count": investigation.constraints.target_count,
                "posted_from": posted_from,
                "posted_to": posted_to,
                "experience": investigation.constraints.experience,
                "location": investigation.constraints.location,
                "employment_type": investigation.constraints.employment_type,
                "freshness_required": bool(
                    arguments.get("freshness_required")
                    or posted_from
                    or posted_to
                ),
                "purpose": investigation.purpose.value,
                "analysis_goal": investigation.objective,
                "task_category": "검색",
            }
        )
        signature = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        if signature in signatures:
            continue
        signatures.add(signature)
        from shared.schema.agent_contract import CollectionToolArguments

        normalized.append(
            step.model_copy(
                update={
                    "tool_name": "realtime_scraping",
                    "arguments": CollectionToolArguments.model_validate(arguments),
                }
            )
        )
        if len(normalized) >= maximum_steps:
            break
    return normalized


def _normalized_evidence_requirements(
    plan: EvidencePlan,
    investigation: InvestigationRequest,
    taxonomy_service: SearchTaxonomyService,
) -> list:
    """근거 필드를 DB 계약에 맞추고 화면 전체 수집의 표본 수를 정규화한다."""

    normalized = []
    single_requirement = len(plan.requirements) == 1
    for requirement in plan.requirements:
        updates = {
            "required_fields": list(dict.fromkeys(requirement.required_fields))
        }
        if investigation.constraints.count_mode == "visible_all":
            updates["minimum_count"] = 1
        elif (
            single_requirement
            and investigation.constraints.count_mode == "explicit"
        ):
            updates["minimum_count"] = investigation.constraints.target_count
        normalized.append(
            taxonomy_service.enrich_requirement(
                requirement.model_copy(update=updates),
                investigation.constraints,
            )
        )
    return normalized


def _needs_semantic_evidence_validation(report: dict[str, Any]) -> bool:
    return any(
        item.get("semantic_review_required") and item.get("candidates")
        for item in report.get("requirements", [])
        if isinstance(item, dict)
    )


def _apply_evidence_validation(
    report: dict[str, Any],
    investigation: InvestigationRequest,
    validation: EvidenceValidation,
) -> dict[str, Any]:
    """모델 판단을 후보 집합 안에서만 허용하고 충분성을 다시 계산한다."""

    requirements = {
        item.requirement_id: item for item in investigation.evidence_requirements
    }
    decisions = {
        item.requirement_id: item
        for item in validation.decisions
    }
    all_document_ids: list[int] = []
    seen_document_ids: set[int] = set()
    reports: list[dict[str, Any]] = []
    for item in report.get("requirements", []):
        requirement = requirements.get(str(item.get("requirement_id") or ""))
        if requirement is None:
            continue
        candidates = {
            int(candidate["document_id"]): candidate
            for candidate in item.get("candidates", [])
        }
        decision = decisions.get(requirement.requirement_id)

        def valid_ids(values: list[int]) -> list[int]:
            return list(
                dict.fromkeys(
                    int(document_id)
                    for document_id in values
                    if int(document_id) in candidates
                )
            )

        if item.get("semantic_review_required"):
            matching_ids = valid_ids(
                decision.matching_document_ids if decision is not None else []
            )
        else:
            matching_ids = valid_ids(
                [candidate["document_id"] for candidate in item.get("candidates", [])]
            )
        selected_ids = matching_ids
        if (
            len(investigation.evidence_requirements) == 1
            and investigation.constraints.count_mode == "explicit"
        ):
            selected_ids = matching_ids[: investigation.constraints.target_count]
        selected = [candidates[document_id] for document_id in selected_ids]
        for document_id in selected_ids:
            if document_id not in seen_document_ids:
                seen_document_ids.add(document_id)
                all_document_ids.append(document_id)
        missing: list[str] = []
        if len(selected) < requirement.minimum_count:
            missing.append(f"의미 조건을 만족하는 표본 {requirement.minimum_count - len(selected)}건 부족")
        field_coverage = {
            field: sum(
                1
                for candidate in selected
                if candidate.get("field_presence", {}).get(field)
            )
            for field in requirement.required_fields
        }
        for field in requirement.required_fields:
            if field_coverage.get(field, 0) < requirement.minimum_count:
                missing.append(f"{field} 근거 부족")
        posted_dates = sorted(
            str(candidate.get("posted_at") or "")[:10]
            for candidate in selected
            if str(candidate.get("posted_at") or "").strip()
        )
        verified_dates = len(posted_dates)
        if (requirement.posted_from or requirement.posted_to) and verified_dates < requirement.minimum_count:
            missing.append("검증된 게시일 근거 부족")
        reports.append(
            {
                **item,
                "matching_count": len(selected),
                "verified_posted_at_count": verified_dates,
                "oldest_posted_at": posted_dates[0] if posted_dates else "",
                "newest_posted_at": posted_dates[-1] if posted_dates else "",
                "field_coverage": field_coverage,
                "document_ids": selected_ids,
                "site_counts": dict(
                    Counter(
                        str(candidate.get("source_platform") or "unknown")
                        for candidate in selected
                    )
                ),
                "candidates": selected,
                "sufficient": not missing,
                "missing": list(dict.fromkeys(missing)),
            }
        )
    missing_evidence = [
        f"{item['description']}: {reason}"
        for item in reports
        for reason in item["missing"]
    ]
    return {
        **report,
        "requirements": reports,
        "sufficient": bool(reports) and all(item["sufficient"] for item in reports),
        "document_ids": all_document_ids,
        "missing_evidence": missing_evidence,
    }


class InvestigationWorkflow:
    """도구 실행 전에 조사 계획을 확정하는 지휘자 실행기."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        checkpoint_runtime: InvestigationCheckpointRuntime | None = None,
        models: InvestigationModels | None = None,
        capabilities: list[Any] | None = None,
        collection_tool: Any = None,
        taxonomy_service: SearchTaxonomyService | None = None,
        taxonomy_review_service: SearchTaxonomyReviewService | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        self.db_path = Path(db_path)
        from shared.db.database import Database

        Database(self.db_path)
        self._owns_checkpoint_runtime = checkpoint_runtime is None
        self.checkpoint_runtime = checkpoint_runtime or InvestigationCheckpointRuntime(
            self.db_path
        )
        self.models = models or InvestigationModels()
        self.capabilities = capabilities or build_tool_capability_catalog()
        self.collection_tool = collection_tool or realtime_scraping
        self.taxonomy_service = taxonomy_service or SearchTaxonomyService(self.db_path)
        self.taxonomy_review_service = (
            taxonomy_review_service or SearchTaxonomyReviewService(self.db_path)
        )
        self.now = now or (lambda: datetime.now().astimezone())
        self.graph = self._build_graph()

    def close(self) -> None:
        """이 실행기가 만든 체크포인트 연결을 닫는다."""

        if self._owns_checkpoint_runtime:
            self.checkpoint_runtime.close()

    @staticmethod
    def _non_taxonomy_questions(
        questions: list[ClarificationQuestion],
    ) -> list[ClarificationQuestion]:
        taxonomy_facets = {
            "occupation_domain",
            "occupation_family",
            "occupation",
            "semantic_occupation",
        }
        return [
            question
            for question in questions
            if question.facet_type not in taxonomy_facets
        ]

    def _semantic_occupation_question(
        self,
        constraints: InvestigationConstraints,
    ) -> tuple[InvestigationConstraints, ClarificationQuestion | None]:
        if not (
            constraints.occupation_domain_concept_keys
            and constraints.occupation_query
            and not constraints.occupation_concept_keys
            and constraints.occupation_resolution == "unresolved"
        ):
            return constraints, None
        candidates = self.taxonomy_service.occupation_resolution_candidates(
            constraints.occupation_domain_concept_keys
        )
        if not candidates:
            updated = constraints.model_copy(
                update={"occupation_resolution": "semantic_no_match"}
            )
            return updated, None
        resolution = _model_payload(
            invoke_with_metrics(
                self.models.taxonomy(),
                [
                    SystemMessage(content=taxonomy_resolution_prompt()),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "occupation_query": constraints.occupation_query,
                                "occupation_domain_concept_keys": list(
                                    constraints.occupation_domain_concept_keys
                                ),
                                "candidates": candidates,
                            },
                            ensure_ascii=False,
                        )
                    ),
                ],
                "investigation_taxonomy_resolution",
            ),
            TaxonomyResolution,
        )
        by_key = {str(item["concept_key"]): item for item in candidates}
        selected_key = str(resolution.selected_concept_key or "")
        ordered_keys = [
            selected_key,
            *(str(key) for key in resolution.alternative_concept_keys),
        ]
        valid_keys = list(
            dict.fromkeys(key for key in ordered_keys if key in by_key)
        )[:4]
        if resolution.decision == "no_match" or selected_key not in by_key:
            self.taxonomy_service.record_occupation_candidate(
                constraints.occupation_query,
                metadata={
                    "occupation_domain_concept_keys": list(
                        constraints.occupation_domain_concept_keys
                    ),
                    "candidate_count": len(candidates),
                    "resolution_reason": resolution.reason,
                },
            )
            updated = constraints.model_copy(
                update={"occupation_resolution": "semantic_no_match"}
            )
            return updated, None

        options: list[ClarificationOption] = []
        for concept_key in valid_keys:
            item = by_key[concept_key]
            matching_count = len(
                self.taxonomy_service.matching_occupation_job_ids(
                    [concept_key],
                    constraints,
                )
            )
            options.append(
                ClarificationOption(
                    option_id=(
                        "concept-"
                        + hashlib.sha1(concept_key.encode("utf-8")).hexdigest()[:10]
                    ),
                    label=str(item["label"]),
                    value=concept_key,
                    collection_search_term=str(item["label"]),
                    matching_count=matching_count,
                    concept_count=self.taxonomy_service.occupation_descendant_count(
                        concept_key
                    ),
                    description=str(item["definition"] or ""),
                )
            )
        fingerprint = hashlib.sha1(
            (
                constraints.occupation_query
                + "|"
                + "|".join(sorted(constraints.occupation_domain_concept_keys))
            ).encode("utf-8")
        ).hexdigest()[:12]
        return constraints, ClarificationQuestion(
            question_id=f"semantic_occupation:{fingerprint}",
            field="occupation_concept_keys",
            question=f"'{constraints.occupation_query}'의 의미를 어떤 직무로 확정할까요?",
            options=options,
            allow_custom=True,
            reason=(
                resolution.reason
                or "선택한 업무 영역의 사전 후보 중 의미가 가까운 직무를 확인합니다."
            ),
            candidate_count=len(
                self.taxonomy_service.matching_occupation_job_ids(
                    constraints.occupation_domain_concept_keys,
                    constraints,
                )
            ),
            concept_count=len(candidates),
            facet_type="semantic_occupation",
        )

    def _prepare_taxonomy_questions(
        self,
        constraints: InvestigationConstraints,
        questions: list[ClarificationQuestion],
        *,
        answered_question_ids: list[str] | tuple[str, ...] = (),
    ) -> tuple[InvestigationConstraints, list[ClarificationQuestion]]:
        pending = self._non_taxonomy_questions(questions)
        if pending:
            return constraints, pending
        constraints, semantic_question = self._semantic_occupation_question(
            constraints
        )
        if semantic_question is not None:
            return constraints, [semantic_question]
        next_question = self.taxonomy_service.build_next_scope_question(
            constraints,
            answered_question_ids=answered_question_ids,
        )
        return constraints, [next_question] if next_question is not None else []

    def _accept_confirmed_semantic_alias(
        self,
        investigation: InvestigationRequest,
        answer: ClarificationAnswer,
    ) -> None:
        if answer.custom_value.strip():
            return
        question = next(
            (
                item
                for item in investigation.clarification_questions
                if item.question_id == answer.question_id
            ),
            None,
        )
        if question is None or question.facet_type != "semantic_occupation":
            return
        selected = next(
            (
                option
                for option in question.options
                if option.option_id == answer.selected_option_id
                or (answer.value and option.value == answer.value)
            ),
            None,
        )
        if selected is None or not investigation.constraints.occupation_query:
            return
        self.taxonomy_review_service.add_reviewed_alias(
            selected.value,
            investigation.constraints.occupation_query,
            note="직무 의미 확인 질문에서 사용자가 선택",
        )

    def _understand(self, state: InvestigationGraphState) -> dict[str, Any]:
        raise_if_cancelled()
        existing = InvestigationRequest.model_validate(state["investigation"])
        if existing.objective:
            return {}
        emit_run_event("request_understanding", RunPhase.PLANNING, "요청의 목적과 부족한 조건을 확인하고 있습니다.")
        analysis = _model_payload(
            invoke_with_metrics(
                self.models.analysis(),
                [
                    SystemMessage(content=request_analysis_prompt(self.now())),
                    HumanMessage(content=existing.original_query),
                ],
                "investigation_request_analysis",
            ),
            RequestAnalysis,
        )
        constraints = self.taxonomy_service.enrich_constraints(
            _normalize_site_slugs(analysis.constraints)
        )
        questions = [
            question
            for question in analysis.clarification_questions
            if not (
                question.field == "occupation_query"
                and constraints.occupation_concept_keys
            )
        ]
        constraints, questions = self._prepare_taxonomy_questions(
            constraints,
            questions,
        )
        updated = existing.model_copy(
            update={
                "objective": analysis.objective,
                "deliverable": analysis.deliverable,
                "purpose": analysis.purpose,
                "evidence_policy": analysis.evidence_policy,
                "constraints": constraints,
                "assumptions": analysis.assumptions,
                "clarification_questions": questions,
                "status": (
                    InvestigationStatus.AWAITING_CLARIFICATION
                    if questions
                    else InvestigationStatus.CHECKING_EVIDENCE
                ),
            }
        )
        return {"investigation": updated.model_dump(mode="json")}

    @staticmethod
    def _route_after_understand(state: InvestigationGraphState) -> str:
        investigation = InvestigationRequest.model_validate(state["investigation"])
        if investigation.clarification_questions:
            return "clarify"
        if investigation.evidence_policy == EvidencePolicy.MODEL_KNOWLEDGE:
            return "answer"
        return "define_evidence"

    def _clarify(self, state: InvestigationGraphState) -> dict[str, Any]:
        investigation = InvestigationRequest.model_validate(state["investigation"])
        question = next(iter(investigation.clarification_questions), None)
        if question is None:
            raise ValueError("사용자에게 확인할 질문이 없습니다.")
        payload = {
            "needs_clarification": True,
            **question.model_dump(mode="json"),
            "missing_fields": [
                item.field for item in investigation.clarification_questions
            ],
            "investigation_id": investigation.investigation_id,
        }
        answer = ClarificationAnswer.model_validate(interrupt(payload))
        self._accept_confirmed_semantic_alias(investigation, answer)
        updated = apply_clarification_answer(
            investigation,
            answer,
            today=self.now().date(),
        )
        constraints = self.taxonomy_service.enrich_constraints(updated.constraints)
        answered_question_ids = [
            item.question_id for item in updated.clarification_answers
        ]
        constraints, remaining_questions = self._prepare_taxonomy_questions(
            constraints,
            updated.clarification_questions,
            answered_question_ids=answered_question_ids,
        )
        updated = updated.model_copy(
            update={
                "constraints": constraints,
                "clarification_questions": remaining_questions,
                "status": (
                    InvestigationStatus.AWAITING_CLARIFICATION
                    if remaining_questions
                    else InvestigationStatus.CHECKING_EVIDENCE
                ),
            }
        )
        return {
            "investigation": updated.model_dump(mode="json"),
        }

    def _define_evidence(self, state: InvestigationGraphState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = InvestigationRequest.model_validate(state["investigation"])
        emit_run_event("evidence_planning", RunPhase.PLANNING, "답변에 필요한 근거를 정리하고 있습니다.")
        plan = _model_payload(
            invoke_with_metrics(
                self.models.evidence(),
                [
                    SystemMessage(content=evidence_plan_prompt(self.now())),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "request": _request_prompt_context(investigation),
                                "tool_capabilities": _capabilities_for_investigation(
                                    state["capability_catalog"], investigation
                                ),
                            },
                            ensure_ascii=False,
                        )
                    ),
                ],
                "investigation_evidence_plan",
            ),
            EvidencePlan,
        )
        updated = investigation.model_copy(
            update={
                "evidence_requirements": _normalized_evidence_requirements(
                    plan,
                    investigation,
                    self.taxonomy_service,
                ),
                "status": InvestigationStatus.CHECKING_EVIDENCE,
            }
        )
        return {"investigation": updated.model_dump(mode="json")}

    def _inspect_evidence(self, state: InvestigationGraphState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = InvestigationRequest.model_validate(state["investigation"])
        emit_run_event("database_check", RunPhase.DATABASE, "DB에 필요한 근거가 있는지 확인하고 있습니다.")
        collected_web_evidence = (
            investigation.evidence_policy == EvidencePolicy.WEB_REQUIRED
            and bool(investigation.executed_step_ids)
        )
        report = inspect_job_evidence(
            self.db_path,
            investigation.evidence_requirements,
            investigation.constraints,
            document_scope_ids=(
                investigation.collection_document_ids
                if collected_web_evidence
                else None
            ),
            force_semantic_review=collected_web_evidence,
        )
        if _needs_semantic_evidence_validation(report):
            validation = _model_payload(
                invoke_with_metrics(
                    self.models.validation(),
                    [
                        SystemMessage(content=evidence_validation_prompt()),
                        HumanMessage(
                            content=json.dumps(
                                {
                                    "request": _request_prompt_context(investigation),
                                    "candidate_groups": _evidence_validation_payload(
                                        report,
                                        investigation,
                                    ),
                                },
                                ensure_ascii=False,
                            )
                        ),
                    ],
                    "investigation_evidence_validation",
                ),
                EvidenceValidation,
            )
            report = _apply_evidence_validation(report, investigation, validation)
        evidence_document_ids = list(report.get("document_ids", []))
        report["document_ids"] = evidence_document_ids
        updated = investigation.model_copy(
            update={
                "evidence_snapshot": report,
                "missing_evidence": report.get("missing_evidence", []),
                "evidence_document_ids": evidence_document_ids,
                "status": (
                    InvestigationStatus.ANSWERING
                    if report.get("sufficient")
                    else InvestigationStatus.PLANNING
                ),
            }
        )
        return {
            "investigation": updated.model_dump(mode="json"),
            "db_report": report,
            "valid_ids": evidence_document_ids,
        }

    @staticmethod
    def _route_after_evidence(state: InvestigationGraphState) -> str:
        investigation = InvestigationRequest.model_validate(state["investigation"])
        if investigation.evidence_policy == EvidencePolicy.DATABASE_ONLY:
            return "load_documents"
        if (
            investigation.evidence_policy == EvidencePolicy.WEB_REQUIRED
            and not investigation.executed_step_ids
        ):
            return "plan_actions"
        if state.get("db_report", {}).get("sufficient"):
            return "load_documents"
        pending = [
            step
            for step in investigation.plan
            if step.step_id not in investigation.executed_step_ids
        ]
        if pending:
            return "execute"
        if investigation.plan:
            return "load_documents"
        return "plan_actions"

    def _plan_actions(self, state: InvestigationGraphState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = InvestigationRequest.model_validate(state["investigation"])
        emit_run_event("action_planning", RunPhase.PLANNING, "부족한 자료를 확보할 행동계획을 세우고 있습니다.")
        plan = _model_payload(
            invoke_with_metrics(
                self.models.action(),
                [
                    SystemMessage(content=action_plan_prompt()),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "request": _request_prompt_context(investigation),
                                "db_report": _compact_db_report(
                                    state.get("db_report", {})
                                ),
                                "tool_capabilities": _capabilities_for_investigation(
                                    state["capability_catalog"], investigation
                                ),
                            },
                            ensure_ascii=False,
                        )
                    ),
                ],
                "investigation_action_plan",
            ),
            InvestigationActionPlan,
        )
        allowed_steps = _normalized_collection_steps(
            plan,
            investigation,
            state["capability_catalog"],
        )
        updated = investigation.model_copy(
            update={
                "plan": allowed_steps,
                "status": (
                    InvestigationStatus.EXECUTING
                    if allowed_steps
                    else InvestigationStatus.ANSWERING
                ),
            }
        )
        return {
            "investigation": updated.model_dump(mode="json"),
            "cannot_proceed_reason": plan.cannot_proceed_reason,
        }

    @staticmethod
    def _route_after_plan(state: InvestigationGraphState) -> str:
        investigation = InvestigationRequest.model_validate(state["investigation"])
        return "execute" if investigation.plan else "load_documents"

    def _execute(self, state: InvestigationGraphState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = InvestigationRequest.model_validate(state["investigation"])
        step = next(
            item
            for item in investigation.plan
            if item.step_id not in investigation.executed_step_ids
        )
        emit_run_event("collection_started", RunPhase.COLLECTION, step.purpose or "계획한 채용공고 수집을 실행하고 있습니다.")
        raw_result = self.collection_tool.invoke(
            step.arguments.model_dump(mode="json")
        )
        try:
            parsed_result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        except json.JSONDecodeError:
            parsed_result = {"raw_result": str(raw_result)}
        executed = [*investigation.executed_step_ids, step.step_id]
        persistence_validation = (
            parsed_result.get("persistence_validation", {})
            if isinstance(parsed_result, dict)
            else {}
        )
        observed_ids = {
            int(item["job_id"])
            for item in persistence_validation.get("persisted_items", [])
            if isinstance(item, dict) and item.get("job_id") is not None
        }
        observed_ids.update(
            int(job_id)
            for job_id in (
                parsed_result.get("observed_job_ids", [])
                if isinstance(parsed_result, dict)
                else []
            )
            if str(job_id).isdigit() and int(job_id) > 0
        )
        steps = [
            item.model_copy(update={"status": "completed"}) if item.step_id == step.step_id else item
            for item in investigation.plan
        ]
        updated = investigation.model_copy(
            update={
                "executed_step_ids": executed,
                "collection_document_ids": sorted(
                    set(investigation.collection_document_ids) | observed_ids
                ),
                "plan": steps,
                "status": InvestigationStatus.VALIDATING,
            }
        )
        return {
            "investigation": updated.model_dump(mode="json"),
            "collection_results": [*state.get("collection_results", []), parsed_result],
        }

    def _load_documents(self, state: InvestigationGraphState) -> dict[str, Any]:
        investigation = InvestigationRequest.model_validate(state["investigation"])
        ids = sorted(set(investigation.evidence_document_ids))
        if not ids:
            return {"documents": [], "valid_ids": []}
        documents = load_job_evidence_documents(self.db_path, ids)
        return {
            "documents": [document.model_dump(mode="json") for document in documents],
            "valid_ids": [document.id for document in documents],
        }

    def _answer(self, state: InvestigationGraphState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = InvestigationRequest.model_validate(state["investigation"])
        emit_run_event("answering_started", RunPhase.ANSWERING, "검증된 근거로 답변을 정리하고 있습니다.")
        response = invoke_with_metrics(
            self.models.answer(),
            [
                SystemMessage(content=answer_prompt()),
                HumanMessage(
                    content=json.dumps(
                        {
                            "request": _request_prompt_context(investigation),
                            "db_report": _compact_db_report(
                                state.get("db_report", {})
                            ),
                            "collection_results": _compact_collection_results(
                                state.get("collection_results", [])
                            ),
                            "cannot_proceed_reason": state.get("cannot_proceed_reason", ""),
                            "documents": _answer_evidence_documents(
                                state.get("documents", [])
                            ),
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            "investigation_answer",
        )
        answer = _message_text(response)
        updated = investigation.model_copy(
            update={
                "final_answer": answer,
                "status": InvestigationStatus.COMPLETED,
            }
        )
        emit_run_event("run_completed", RunPhase.COMPLETED, "답변을 완료했습니다.", status=RunStatus.COMPLETED)
        return {
            "investigation": updated.model_dump(mode="json"),
            "final_answer": answer,
            "run_status": RunStatus.COMPLETED.value,
        }

    def _build_graph(self):
        workflow = StateGraph(InvestigationGraphState)
        workflow.add_node("understand", self._understand)
        workflow.add_node("clarify", self._clarify)
        workflow.add_node("define_evidence", self._define_evidence)
        workflow.add_node("inspect_evidence", self._inspect_evidence)
        workflow.add_node("plan_actions", self._plan_actions)
        workflow.add_node("execute", self._execute)
        workflow.add_node("load_documents", self._load_documents)
        workflow.add_node("answer", self._answer)
        workflow.add_edge(START, "understand")
        workflow.add_conditional_edges(
            "understand",
            self._route_after_understand,
            {
                "clarify": "clarify",
                "answer": "answer",
                "define_evidence": "define_evidence",
            },
        )
        workflow.add_conditional_edges(
            "clarify",
            self._route_after_understand,
            {
                "clarify": "clarify",
                "answer": "answer",
                "define_evidence": "define_evidence",
            },
        )
        workflow.add_edge("define_evidence", "inspect_evidence")
        workflow.add_conditional_edges(
            "inspect_evidence",
            self._route_after_evidence,
            {
                "load_documents": "load_documents",
                "execute": "execute",
                "plan_actions": "plan_actions",
            },
        )
        workflow.add_conditional_edges(
            "plan_actions",
            self._route_after_plan,
            {"execute": "execute", "load_documents": "load_documents"},
        )
        workflow.add_edge("execute", "inspect_evidence")
        workflow.add_edge("load_documents", "answer")
        workflow.add_edge("answer", END)
        return workflow.compile(checkpointer=self.checkpoint_runtime.saver)

    @staticmethod
    def _thread_config(investigation_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": investigation_id}}

    @staticmethod
    def _pending_clarification(result: dict[str, Any]) -> dict[str, Any] | None:
        interruptions = result.get("__interrupt__") or ()
        if not interruptions:
            return None
        payload = getattr(interruptions[0], "value", None)
        return dict(payload) if isinstance(payload, dict) else None

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        clarification = self._pending_clarification(result)
        normalized = {
            key: value for key, value in result.items() if key != "__interrupt__"
        }
        if clarification is None:
            return normalized
        emit_run_event(
            "clarification_required",
            RunPhase.CLARIFICATION,
            str(clarification.get("question") or "추가 정보가 필요합니다."),
            status=RunStatus.WAITING_INPUT,
            data=clarification,
        )
        normalized.update(
            {
                "clarification": clarification,
                "final_answer": str(clarification.get("question") or ""),
                "run_status": RunStatus.WAITING_INPUT.value,
            }
        )
        return normalized

    def run(
        self,
        query: str,
        *,
        conversation_id: str = "",
        investigation_id: str = "",
        clarification_answer: ClarificationAnswer | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if investigation_id:
            config = self._thread_config(investigation_id)
            snapshot = self.graph.get_state(config)
            if not snapshot.values:
                raise ValueError("재개할 조사 상태를 찾을 수 없습니다.")
            if clarification_answer is None:
                raise ValueError("조사를 재개하려면 확인 질문의 답변이 필요합니다.")
            answer = ClarificationAnswer.model_validate(clarification_answer)
            pending = [
                interrupt_value.value
                for task in snapshot.tasks
                for interrupt_value in task.interrupts
                if isinstance(interrupt_value.value, dict)
            ]
            if not pending:
                raise ValueError("현재 재개할 확인 질문이 없습니다.")
            if str(pending[0].get("question_id") or "") != answer.question_id:
                raise ValueError("현재 확인 질문과 답변의 식별자가 다릅니다.")
            result = self.graph.invoke(
                Command(resume=answer.model_dump(mode="json")),
                config=config,
            )
            return self._normalize_result(dict(result))

        investigation = InvestigationRequest(
            investigation_id=f"investigation-{uuid.uuid4().hex}",
            conversation_id=conversation_id,
            original_query=str(query or "").strip(),
        )
        state: InvestigationGraphState = {
            "investigation": investigation.model_dump(mode="json"),
            "capability_catalog": [item.model_dump(mode="json") for item in self.capabilities],
            "collection_results": [],
            "valid_ids": [],
        }
        result = self.graph.invoke(
            state,
            config=self._thread_config(investigation.investigation_id),
        )
        return self._normalize_result(dict(result))


__all__ = ["InvestigationModels", "InvestigationWorkflow"]

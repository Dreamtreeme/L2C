"""사용자 요청 해석과 확인 질문을 담당하는 조사 그래프 노드."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from agent.application.clarification_service import apply_clarification_answer
from agent.application.run_context import (
    emit_run_event,
    invoke_with_metrics,
    raise_if_cancelled,
)
from agent.application.run_contracts import RunPhase
from agent.application.search_taxonomy_review_service import (
    SearchTaxonomyReviewService,
)
from agent.application.search_taxonomy_service import SearchTaxonomyService
from agent.graph.investigation_context import (
    InvestigationGraphState,
    InvestigationModels,
    normalize_site_slugs,
    parse_model_payload,
)
from agent.prompts.investigation import (
    request_analysis_prompt,
    taxonomy_resolution_prompt,
)
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    ClarificationOption,
    ClarificationQuestion,
    EvidencePolicy,
    InvestigationConstraints,
    InvestigationRequest,
    InvestigationStatus,
    RequestAnalysis,
    TaxonomyResolution,
)


class InvestigationRequestNodes:
    """요청 의미와 사용자 확인이 필요한 조건을 확정한다."""

    def __init__(
        self,
        *,
        models: InvestigationModels,
        taxonomy_service: SearchTaxonomyService,
        taxonomy_review_service: SearchTaxonomyReviewService,
        now: Callable[[], datetime],
    ) -> None:
        self.models = models
        self.taxonomy_service = taxonomy_service
        self.taxonomy_review_service = taxonomy_review_service
        self.now = now

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
        resolution = parse_model_payload(
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

    def understand(self, state: InvestigationGraphState) -> dict[str, Any]:
        raise_if_cancelled()
        existing = InvestigationRequest.model_validate(state["investigation"])
        if existing.objective:
            return {}
        emit_run_event("request_understanding", RunPhase.PLANNING, "요청의 목적과 부족한 조건을 확인하고 있습니다.")
        analysis = parse_model_payload(
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
            normalize_site_slugs(analysis.constraints)
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
    def route_after_understand(state: InvestigationGraphState) -> str:
        investigation = InvestigationRequest.model_validate(state["investigation"])
        if investigation.clarification_questions:
            return "clarify"
        if investigation.evidence_policy == EvidencePolicy.MODEL_KNOWLEDGE:
            return "answer"
        return "define_evidence"

    def clarify(self, state: InvestigationGraphState) -> dict[str, Any]:
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


__all__ = ["InvestigationRequestNodes"]

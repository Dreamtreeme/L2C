"""직무 사전 후보를 사용자 확인 질문으로 변환한다."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from agent.application.run_context import invoke_with_metrics
from agent.application.search_taxonomy_review_service import (
    SearchTaxonomyReviewService,
)
from agent.application.search_taxonomy_service import SearchTaxonomyService
from agent.prompts.investigation import taxonomy_resolution_prompt
from agent.utils.model_payload import parse_model_payload
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    ClarificationOption,
    ClarificationQuestion,
    InvestigationConstraints,
    InvestigationRequest,
    TaxonomyResolution,
)


class OccupationClarificationService:
    """직무 범위를 좁히는 질문을 만들고 확인된 별칭을 기록한다."""

    def __init__(
        self,
        *,
        taxonomy_model: Callable[[], Any],
        taxonomy_service: SearchTaxonomyService,
        taxonomy_review_service: SearchTaxonomyReviewService,
    ) -> None:
        self.taxonomy_model = taxonomy_model
        self.taxonomy_service = taxonomy_service
        self.taxonomy_review_service = taxonomy_review_service

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

    def _semantic_question(
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
            return constraints.model_copy(
                update={"occupation_resolution": "semantic_no_match"}
            ), None

        resolution = parse_model_payload(
            invoke_with_metrics(
                self.taxonomy_model(),
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
            return constraints.model_copy(
                update={"occupation_resolution": "semantic_no_match"}
            ), None

        options: list[ClarificationOption] = []
        for concept_key in valid_keys:
            item = by_key[concept_key]
            options.append(
                ClarificationOption(
                    option_id=(
                        "concept-"
                        + hashlib.sha1(
                            concept_key.encode("utf-8")
                        ).hexdigest()[:10]
                    ),
                    label=str(item["label"]),
                    value=concept_key,
                    collection_search_term=str(item["label"]),
                    matching_count=len(
                        self.taxonomy_service.matching_occupation_job_ids(
                            [concept_key],
                            constraints,
                        )
                    ),
                    concept_count=(
                        self.taxonomy_service.occupation_descendant_count(
                            concept_key
                        )
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
            question=(
                f"'{constraints.occupation_query}'의 의미를 어떤 직무로 "
                "확정할까요?"
            ),
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

    def prepare_questions(
        self,
        constraints: InvestigationConstraints,
        questions: list[ClarificationQuestion],
        *,
        answered_question_ids: list[str] | tuple[str, ...] = (),
    ) -> tuple[InvestigationConstraints, list[ClarificationQuestion]]:
        pending = self._non_taxonomy_questions(questions)
        if pending:
            return constraints, pending
        constraints, semantic_question = self._semantic_question(constraints)
        if semantic_question is not None:
            return constraints, [semantic_question]
        next_question = self.taxonomy_service.build_next_scope_question(
            constraints,
            answered_question_ids=answered_question_ids,
        )
        return constraints, [next_question] if next_question is not None else []

    def enrich_constraints(
        self,
        constraints: InvestigationConstraints,
    ) -> InvestigationConstraints:
        return self.taxonomy_service.enrich_constraints(constraints)

    def accept_answer(
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


__all__ = ["OccupationClarificationService"]

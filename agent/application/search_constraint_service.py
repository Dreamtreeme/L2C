"""직무·기술 별칭을 조사 조건에 연결하고 부족한 직무를 질문한다."""

from __future__ import annotations

from collections.abc import Iterable

from agent.application.search_taxonomy_service import SearchTaxonomyService
from shared.schema.investigation_schema import (
    ClarificationQuestion,
    InvestigationConstraints,
)


class SearchConstraintService:
    """검색 사전 적용과 직무 확인 질문을 한 경계에서 처리한다."""

    def __init__(self, taxonomy: SearchTaxonomyService) -> None:
        self.taxonomy = taxonomy

    def enrich(
        self,
        constraints: InvestigationConstraints,
    ) -> InvestigationConstraints:
        return self.taxonomy.enrich_constraints(constraints)

    @staticmethod
    def prepare_questions(
        constraints: InvestigationConstraints,
        questions: list[ClarificationQuestion],
        *,
        answered_question_ids: Iterable[str] = (),
    ) -> list[ClarificationQuestion]:
        if questions:
            return questions
        if (
            not constraints.occupation_scope_required
            or constraints.occupation_query
            or constraints.occupation_concept_keys
            or "occupation_query" in set(answered_question_ids)
        ):
            return []
        return [
            ClarificationQuestion(
                question_id="occupation_query",
                field="occupation_query",
                question="어떤 직무의 채용공고를 찾을까요?",
                allow_custom=True,
                reason="채용공고를 검색할 직무가 필요합니다.",
            )
        ]


__all__ = ["SearchConstraintService"]

"""조사 그래프가 외부 기능을 호출할 때 사용하는 경계 계약."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol, Sequence

from shared.schema.collection_intent import CollectionIntent
from shared.schema.collection_run import CollectionBatch, PersistedCollection
from shared.schema.jd_schema import StoredJob
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    ClarificationQuestion,
    EvidenceRequirement,
    InvestigationConstraints,
    InvestigationRequest,
    ConversationTurn,
)


class OccupationClarificationPort(Protocol):
    """직무 사전으로 조사 조건과 확인 질문을 보완한다."""

    def enrich_constraints(
        self,
        constraints: InvestigationConstraints,
    ) -> InvestigationConstraints: ...

    def prepare_questions(
        self,
        constraints: InvestigationConstraints,
        questions: list[ClarificationQuestion],
        *,
        answered_question_ids: list[str] | tuple[str, ...] = (),
    ) -> tuple[InvestigationConstraints, list[ClarificationQuestion]]: ...

    def accept_answer(
        self,
        investigation: InvestigationRequest,
        answer: ClarificationAnswer,
    ) -> None: ...


class ClarificationAnswerPort(Protocol):
    """사용자 답변을 조사 요청에 반영한다."""

    def __call__(
        self,
        investigation: InvestigationRequest,
        answer: ClarificationAnswer,
        *,
        today: date,
    ) -> InvestigationRequest: ...


class TaxonomyRequirementPort(Protocol):
    """근거 요구사항을 검색 사전 기준으로 확장한다."""

    def enrich_requirement(
        self,
        requirement: EvidenceRequirement,
        constraints: InvestigationConstraints,
    ) -> EvidenceRequirement: ...


class EvidenceInspectorPort(Protocol):
    """현재 DB 근거가 조사 조건을 충족하는지 검사한다."""

    def __call__(
        self,
        requirements: list[EvidenceRequirement],
        constraints: InvestigationConstraints,
        *,
        document_scope_ids: list[int] | set[int] | None = None,
        force_semantic_review: bool = False,
    ) -> dict[str, Any]: ...


class StoredJobLoaderPort(Protocol):
    """검증된 식별자의 정규화된 공고를 조회한다."""

    def __call__(self, document_ids: list[int]) -> Sequence[StoredJob]: ...


class CollectionRunnerPort(Protocol):
    """확정된 수집 의도로 비전 작업자 하위 그래프를 실행한다."""

    def __call__(self, intent: CollectionIntent) -> CollectionBatch: ...


class CollectionPersistencePort(Protocol):
    """작업자 결과를 DB와 실행 기록 저장소에 확정한다."""

    def __call__(self, batch: CollectionBatch) -> PersistedCollection: ...


class ConversationContextPort(Protocol):
    """대화와 재개 실행에서 요청 해석에 필요한 이전 문맥을 읽는다."""

    def __call__(
        self,
        conversation_id: str,
        resume_run_id: str = "",
    ) -> list[ConversationTurn]: ...


class RunLookupPort(Protocol):
    """API 실행 ID로 체크포인트 재개 메타데이터를 찾는다."""

    def __call__(self, run_id: str) -> dict[str, Any] | None: ...


__all__ = [
    "ClarificationAnswerPort",
    "CollectionPersistencePort",
    "CollectionRunnerPort",
    "ConversationContextPort",
    "EvidenceInspectorPort",
    "OccupationClarificationPort",
    "RunLookupPort",
    "TaxonomyRequirementPort",
    "StoredJobLoaderPort",
]

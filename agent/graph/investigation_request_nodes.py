"""사용자 요청 해석과 확인 질문을 담당하는 조사 그래프 노드."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from agent.observability.run_context import (
    emit_run_event,
    invoke_with_metrics,
    raise_if_cancelled,
)
from agent.observability.run_contracts import RunPhase
from agent.graph.investigation_context import (
    InvestigationState,
    InvestigationModels,
    normalize_site_slugs,
)
from agent.graph.investigation_ports import (
    ClarificationAnswerPort,
    ConversationContextPort,
    OccupationClarificationPort,
)
from agent.prompts.investigation import request_analysis_prompt
from agent.utils.model_conversion import parse_model_payload
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    EvidencePolicy,
    InvestigationStatus,
    RequestAnalysis,
)


class InvestigationRequestNodes:
    """요청 의미와 사용자 확인이 필요한 조건을 확정한다."""

    def __init__(
        self,
        *,
        models: InvestigationModels,
        occupation_clarification: OccupationClarificationPort,
        apply_clarification: ClarificationAnswerPort,
        load_conversation_context: ConversationContextPort,
        now: Callable[[], datetime],
    ) -> None:
        self.models = models
        self.occupation_clarification = occupation_clarification
        self.apply_clarification = apply_clarification
        self.load_conversation_context = load_conversation_context
        self.now = now

    def load_context(self, state: InvestigationState) -> dict[str, Any]:
        """현재 대화와 재개 요청의 문맥을 그래프 상태에 적재한다."""

        investigation = state["request"]["investigation"]
        history = self.load_conversation_context(
            investigation.conversation_id,
            state["request"].get("resume_run_id", ""),
        )
        return {"request": {"conversation_history": history}}

    def understand(self, state: InvestigationState) -> dict[str, Any]:
        raise_if_cancelled()
        existing = state["request"]["investigation"]
        if existing.objective:
            return {}
        emit_run_event(
            "request_understanding",
            RunPhase.PLANNING,
            "요청의 목적과 부족한 조건을 확인하고 있습니다.",
        )
        analysis = parse_model_payload(
            invoke_with_metrics(
                self.models.analysis(),
                [
                    SystemMessage(content=request_analysis_prompt(self.now())),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "current_request": existing.original_query,
                                "recent_conversation": [
                                    turn.model_dump(mode="json")
                                    for turn in state["request"].get(
                                        "conversation_history", []
                                    )
                                ],
                            },
                            ensure_ascii=False,
                        )
                    ),
                ],
                "investigation_request_analysis",
            ),
            RequestAnalysis,
        )
        constraints = self.occupation_clarification.enrich_constraints(
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
        constraints, questions = self.occupation_clarification.prepare_questions(
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
        return {"request": {"investigation": updated}}

    @staticmethod
    def route_after_understand(state: InvestigationState) -> str:
        investigation = state["request"]["investigation"]
        if investigation.clarification_questions:
            return "clarify"
        if investigation.evidence_policy == EvidencePolicy.MODEL_KNOWLEDGE:
            return "answer"
        return "define_evidence"

    def clarify(self, state: InvestigationState) -> dict[str, Any]:
        investigation = state["request"]["investigation"]
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
        self.occupation_clarification.accept_answer(investigation, answer)
        updated = self.apply_clarification(
            investigation,
            answer,
            today=self.now().date(),
        )
        constraints = self.occupation_clarification.enrich_constraints(
            updated.constraints
        )
        answered_question_ids = [
            item.question_id for item in updated.clarification_answers
        ]
        constraints, remaining_questions = (
            self.occupation_clarification.prepare_questions(
                constraints,
                updated.clarification_questions,
                answered_question_ids=answered_question_ids,
            )
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
            "request": {"investigation": updated},
        }


__all__ = ["InvestigationRequestNodes"]

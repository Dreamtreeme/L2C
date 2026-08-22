"""조사 노드를 조립하고 체크포인트 중단·재개를 관리한다."""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.observability.run_context import emit_run_event
from agent.observability.run_contracts import RunPhase
from agent.graph.investigation_answer_nodes import InvestigationAnswerNodes
from agent.graph.investigation_collection_nodes import InvestigationCollectionNodes
from agent.graph.investigation_context import (
    InvestigationState,
    create_investigation_state,
)
from agent.graph.investigation_evidence_nodes import InvestigationEvidenceNodes
from agent.graph.investigation_request_nodes import InvestigationRequestNodes
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    ClarificationQuestion,
    InvestigationOutcome,
    InvestigationRequest,
    InvestigationResumeMode,
)
from shared.schema.run_schema import RunStatus


class InvestigationWorkflow:
    """주입된 조사 노드의 순서, 분기와 중단·재개만 관리한다."""

    def __init__(
        self,
        *,
        checkpointer: Any,
        request_nodes: InvestigationRequestNodes,
        evidence_nodes: InvestigationEvidenceNodes,
        collection_nodes: InvestigationCollectionNodes,
        answer_nodes: InvestigationAnswerNodes,
    ) -> None:
        self.request_nodes = request_nodes
        self.evidence_nodes = evidence_nodes
        self.collection_nodes = collection_nodes
        self.answer_nodes = answer_nodes
        self.graph = self._build_graph(checkpointer)

    def _build_graph(self, checkpointer: Any):
        workflow = StateGraph(InvestigationState)
        workflow.add_node("understand", self.request_nodes.understand)
        workflow.add_node("clarify", self.request_nodes.clarify)
        workflow.add_node("define_evidence", self.evidence_nodes.define_evidence)
        workflow.add_node("inspect_evidence", self.evidence_nodes.inspect_evidence)
        workflow.add_node("plan_actions", self.evidence_nodes.plan_actions)
        workflow.add_node("replan_actions", self.evidence_nodes.replan_actions)
        workflow.add_node("collect", self.collection_nodes.collect)
        workflow.add_node("persist", self.collection_nodes.persist)
        workflow.add_node("answer", self.answer_nodes.answer)
        workflow.add_edge(START, "understand")
        workflow.add_conditional_edges(
            "understand",
            self.request_nodes.route_after_understand,
            {
                "clarify": "clarify",
                "answer": "answer",
                "define_evidence": "define_evidence",
            },
        )
        workflow.add_conditional_edges(
            "clarify",
            self.request_nodes.route_after_understand,
            {
                "clarify": "clarify",
                "answer": "answer",
                "define_evidence": "define_evidence",
            },
        )
        workflow.add_edge("define_evidence", "inspect_evidence")
        workflow.add_conditional_edges(
            "inspect_evidence",
            self.evidence_nodes.route_after_evidence,
            {
                "answer": "answer",
                "collect": "collect",
                "plan_actions": "plan_actions",
                "replan_actions": "replan_actions",
            },
        )
        workflow.add_conditional_edges(
            "plan_actions",
            self.evidence_nodes.route_after_plan,
            {"collect": "collect", "answer": "answer"},
        )
        workflow.add_conditional_edges(
            "replan_actions",
            self.evidence_nodes.route_after_plan,
            {"collect": "collect", "answer": "answer"},
        )
        workflow.add_conditional_edges(
            "collect",
            self._route_after_collect,
            {
                "persist": "persist",
                "collect": "collect",
                "replan_actions": "replan_actions",
                "answer": "answer",
            },
        )
        workflow.add_conditional_edges(
            "persist",
            self._route_after_persist,
            {
                "inspect_evidence": "inspect_evidence",
                "collect": "collect",
                "replan_actions": "replan_actions",
                "answer": "answer",
            },
        )
        workflow.add_edge("answer", END)
        return workflow.compile(checkpointer=checkpointer)

    def _route_after_collect(self, state: InvestigationState) -> str:
        route = self.collection_nodes.route_after_collect(state)
        if route == "answer" and self.evidence_nodes.can_replan(state):
            return "replan_actions"
        return route

    def _route_after_persist(self, state: InvestigationState) -> str:
        route = self.collection_nodes.route_after_persist(state)
        if route == "answer" and self.evidence_nodes.can_replan(state):
            return "replan_actions"
        return route

    @staticmethod
    def _thread_config(investigation_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": investigation_id}}

    @staticmethod
    def _pending_clarification(
        result: dict[str, Any],
    ) -> ClarificationQuestion | None:
        interruptions = result.get("__interrupt__") or ()
        if not interruptions:
            return None
        payload = getattr(interruptions[0], "value", None)
        if not isinstance(payload, dict):
            return None
        return ClarificationQuestion.model_validate(payload)

    def _normalize_result(
        self,
        result: dict[str, Any],
        *,
        resume_mode: InvestigationResumeMode,
    ) -> InvestigationOutcome:
        clarification = self._pending_clarification(result)
        request = dict(result.get("request") or {})
        answer = dict(result.get("answer") or {})
        investigation = request.get("investigation")
        if not isinstance(investigation, InvestigationRequest):
            raise TypeError("조사 그래프 결과에 InvestigationRequest가 없습니다.")
        status_value = str(answer.get("run_status") or RunStatus.COMPLETED.value)
        run_status = RunStatus(status_value)
        final_answer = str(answer.get("final_answer") or "")
        if clarification is not None:
            run_status = RunStatus.WAITING_INPUT
            final_answer = clarification.question
            emit_run_event(
                "clarification_required",
                RunPhase.CLARIFICATION,
                final_answer or "추가 정보가 필요합니다.",
                status=RunStatus.WAITING_INPUT,
                data=clarification.model_dump(mode="json"),
            )
        return InvestigationOutcome(
            investigation=investigation,
            run_status=run_status,
            final_answer=final_answer,
            grounded_answer=answer.get("grounded_answer"),
            clarification=clarification,
            resume_mode=resume_mode,
        )

    def run(
        self,
        query: str,
        *,
        conversation_id: str = "",
        investigation_id: str = "",
        clarification_answer: ClarificationAnswer | None = None,
    ) -> InvestigationOutcome:
        if investigation_id:
            config = self._thread_config(investigation_id)
            snapshot = self.graph.get_state(config)
            if not snapshot.values:
                raise ValueError("재개할 조사 상태를 찾을 수 없습니다.")
            if clarification_answer is None:
                raise ValueError("조사를 재개하려면 확인 질문의 답변이 필요합니다.")
            answer = clarification_answer
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
            return self._normalize_result(
                dict(result),
                resume_mode="checkpoint_resume",
            )

        investigation = InvestigationRequest(
            investigation_id=f"investigation-{uuid.uuid4().hex}",
            conversation_id=conversation_id,
            original_query=str(query or "").strip(),
        )
        state = create_investigation_state(investigation)
        result = self.graph.invoke(
            state,
            config=self._thread_config(investigation.investigation_id),
        )
        return self._normalize_result(
            dict(result),
            resume_mode="",
        )


__all__ = ["InvestigationWorkflow"]

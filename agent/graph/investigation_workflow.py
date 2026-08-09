"""조사 노드를 조립하고 체크포인트 중단·재개를 관리한다."""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.observability.run_context import emit_run_event
from agent.observability.run_contracts import RunPhase, RunStatus
from agent.graph.investigation_answer_nodes import InvestigationAnswerNodes
from agent.graph.investigation_collection_nodes import InvestigationCollectionNodes
from agent.graph.investigation_context import (
    InvestigationState,
    create_investigation_state,
)
from agent.graph.investigation_evidence_nodes import InvestigationEvidenceNodes
from agent.graph.investigation_request_nodes import InvestigationRequestNodes
from agent.graph.investigation_ports import RunLookupPort
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    InvestigationRequest,
)


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
        capabilities: list[Any],
        lookup_run: RunLookupPort,
    ) -> None:
        self.capabilities = list(capabilities)
        self.request_nodes = request_nodes
        self.evidence_nodes = evidence_nodes
        self.collection_nodes = collection_nodes
        self.answer_nodes = answer_nodes
        self.lookup_run = lookup_run
        self.graph = self._build_graph(checkpointer)

    def _build_graph(self, checkpointer: Any):
        workflow = StateGraph(InvestigationState)
        workflow.add_node("load_context", self.request_nodes.load_context)
        workflow.add_node("understand", self.request_nodes.understand)
        workflow.add_node("clarify", self.request_nodes.clarify)
        workflow.add_node("define_evidence", self.evidence_nodes.define_evidence)
        workflow.add_node("inspect_evidence", self.evidence_nodes.inspect_evidence)
        workflow.add_node("plan_actions", self.evidence_nodes.plan_actions)
        workflow.add_node("collect", self.collection_nodes.collect)
        workflow.add_node("persist", self.collection_nodes.persist)
        workflow.add_node("load_documents", self.answer_nodes.load_documents)
        workflow.add_node("answer", self.answer_nodes.answer)
        workflow.add_edge(START, "load_context")
        workflow.add_edge("load_context", "understand")
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
                "load_documents": "load_documents",
                "collect": "collect",
                "plan_actions": "plan_actions",
            },
        )
        workflow.add_conditional_edges(
            "plan_actions",
            self.evidence_nodes.route_after_plan,
            {"collect": "collect", "load_documents": "load_documents"},
        )
        workflow.add_edge("collect", "persist")
        workflow.add_edge("persist", "inspect_evidence")
        workflow.add_edge("load_documents", "answer")
        workflow.add_edge("answer", END)
        return workflow.compile(checkpointer=checkpointer)

    @staticmethod
    def _thread_config(investigation_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": investigation_id}}

    def _resolve_resume(
        self,
        query: str,
        *,
        resume_run_id: str,
        investigation_id: str,
        clarification_answer: ClarificationAnswer | dict[str, Any] | None,
    ) -> tuple[str, str, ClarificationAnswer | dict[str, Any] | None, str]:
        if investigation_id:
            return query, investigation_id, clarification_answer, "checkpoint_resume"
        if not resume_run_id:
            return query, "", clarification_answer, ""

        previous = self.lookup_run(resume_run_id) or {}
        result = dict(previous.get("result") or {})
        clarification = dict(result.get("clarification") or {})
        checkpoint_id = str(result.get("investigation_id") or "")
        if (
            previous.get("status") == RunStatus.WAITING_INPUT.value
            and checkpoint_id
            and clarification.get("question_id")
        ):
            answer = clarification_answer or ClarificationAnswer(
                question_id=str(clarification["question_id"]),
                custom_value=query,
            )
            return query, checkpoint_id, answer, "checkpoint_resume"
        return query, "", clarification_answer, "restart_from_request"

    @staticmethod
    def _pending_clarification(result: dict[str, Any]) -> dict[str, Any] | None:
        interruptions = result.get("__interrupt__") or ()
        if not interruptions:
            return None
        payload = getattr(interruptions[0], "value", None)
        return dict(payload) if isinstance(payload, dict) else None

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        clarification = self._pending_clarification(result)
        request = dict(result.get("request") or {})
        evidence = dict(result.get("evidence") or {})
        execution = dict(result.get("execution") or {})
        answer = dict(result.get("answer") or {})
        investigation = request.get("investigation")
        normalized = {
            "investigation": (
                investigation.model_dump(mode="json")
                if isinstance(investigation, InvestigationRequest)
                else {}
            ),
            "db_report": evidence.get("db_report", {}),
            "documents": evidence.get("documents", []),
            "valid_ids": evidence.get("valid_ids", []),
            "collection_results": [
                item.model_dump(mode="json")
                for item in execution.get("collection_results", [])
            ],
            "run_status": execution.get("run_status", ""),
            "cannot_proceed_reason": execution.get(
                "cannot_proceed_reason",
                "",
            ),
            "final_answer": answer.get("final_answer", ""),
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
        resume_run_id: str = "",
        investigation_id: str = "",
        clarification_answer: ClarificationAnswer | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query, investigation_id, clarification_answer, resume_mode = (
            self._resolve_resume(
                query,
                resume_run_id=resume_run_id,
                investigation_id=investigation_id,
                clarification_answer=clarification_answer,
            )
        )
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
            normalized = self._normalize_result(dict(result))
            normalized["resume_mode"] = resume_mode
            return normalized

        investigation = InvestigationRequest(
            investigation_id=f"investigation-{uuid.uuid4().hex}",
            conversation_id=conversation_id,
            original_query=str(query or "").strip(),
        )
        state = create_investigation_state(
            investigation,
            [item.model_dump(mode="json") for item in self.capabilities],
            resume_run_id=resume_run_id,
        )
        result = self.graph.invoke(
            state,
            config=self._thread_config(investigation.investigation_id),
        )
        normalized = self._normalize_result(dict(result))
        normalized["resume_mode"] = resume_mode
        return normalized


__all__ = ["InvestigationWorkflow"]

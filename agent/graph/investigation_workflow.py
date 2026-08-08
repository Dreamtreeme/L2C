"""조사 노드를 조립하고 체크포인트 중단·재개를 관리한다."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.observability.run_context import emit_run_event
from agent.observability.run_contracts import RunPhase, RunStatus
from agent.application.occupation_clarification_service import (
    OccupationClarificationService,
)
from agent.application.search_taxonomy_review_service import (
    SearchTaxonomyReviewService,
)
from agent.application.search_taxonomy_service import SearchTaxonomyService
from agent.application.tool_capabilities import build_tool_capability_catalog
from agent.graph.investigation_answer_nodes import InvestigationAnswerNodes
from agent.graph.investigation_collection_nodes import InvestigationCollectionNodes
from agent.graph.investigation_context import (
    InvestigationWorkerState,
    InvestigationModels,
)
from agent.graph.investigation_evidence_nodes import InvestigationEvidenceNodes
from agent.graph.investigation_request_nodes import InvestigationRequestNodes
from agent.runtime.investigation_checkpoint import InvestigationCheckpointRuntime
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    InvestigationRequest,
)


class InvestigationWorkflow:
    """조사 노드와 런타임 자원을 연결하는 최상위 그래프 실행기."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        collect_jobs: Callable[[Any], Any],
        checkpoint_runtime: InvestigationCheckpointRuntime | None = None,
        models: InvestigationModels | None = None,
        capabilities: list[Any] | None = None,
        taxonomy_service: SearchTaxonomyService | None = None,
        taxonomy_review_service: SearchTaxonomyReviewService | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        resolved_db_path = Path(db_path)
        from shared.db.database import Database

        Database(resolved_db_path)
        self._owns_checkpoint_runtime = checkpoint_runtime is None
        self.checkpoint_runtime = (
            checkpoint_runtime
            or InvestigationCheckpointRuntime(resolved_db_path)
        )
        resolved_models = models or InvestigationModels()
        self.capabilities = capabilities or build_tool_capability_catalog()
        resolved_taxonomy_service = (
            taxonomy_service or SearchTaxonomyService(resolved_db_path)
        )
        resolved_review_service = (
            taxonomy_review_service
            or SearchTaxonomyReviewService(resolved_db_path)
        )
        now_provider = now or (lambda: datetime.now().astimezone())

        occupation_clarification = OccupationClarificationService(
            taxonomy_model=resolved_models.taxonomy,
            taxonomy_service=resolved_taxonomy_service,
            taxonomy_review_service=resolved_review_service,
        )
        self.request_nodes = InvestigationRequestNodes(
            models=resolved_models,
            occupation_clarification=occupation_clarification,
            now=now_provider,
        )
        self.evidence_nodes = InvestigationEvidenceNodes(
            db_path=resolved_db_path,
            models=resolved_models,
            taxonomy_service=resolved_taxonomy_service,
            now=now_provider,
        )
        self.collection_nodes = InvestigationCollectionNodes(collect_jobs)
        self.answer_nodes = InvestigationAnswerNodes(
            db_path=resolved_db_path,
            models=resolved_models,
        )
        self.graph = self._build_graph()

    def close(self) -> None:
        """이 실행기가 만든 체크포인트 연결을 닫는다."""

        if self._owns_checkpoint_runtime:
            self.checkpoint_runtime.close()

    def _build_graph(self):
        workflow = StateGraph(InvestigationWorkerState)
        workflow.add_node("understand", self.request_nodes.understand)
        workflow.add_node("clarify", self.request_nodes.clarify)
        workflow.add_node("define_evidence", self.evidence_nodes.define_evidence)
        workflow.add_node("inspect_evidence", self.evidence_nodes.inspect_evidence)
        workflow.add_node("plan_actions", self.evidence_nodes.plan_actions)
        workflow.add_node("execute", self.collection_nodes.execute)
        workflow.add_node("load_documents", self.answer_nodes.load_documents)
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
                "load_documents": "load_documents",
                "execute": "execute",
                "plan_actions": "plan_actions",
            },
        )
        workflow.add_conditional_edges(
            "plan_actions",
            self.evidence_nodes.route_after_plan,
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
        state: InvestigationWorkerState = {
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


__all__ = ["InvestigationWorkflow"]

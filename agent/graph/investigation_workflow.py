"""확인, 근거 계획, 도구 실행을 분리한 최상위 조사 LangGraph."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.application.clarification_service import apply_clarification_answer
from agent.application.evidence_service import inspect_job_evidence
from agent.application.investigation_store import InvestigationStore
from agent.application.run_context import emit_run_event, invoke_with_metrics, raise_if_cancelled
from agent.application.run_contracts import RunPhase, RunStatus
from agent.application.tool_capabilities import build_tool_capability_catalog
from agent.prompts.investigation import (
    action_plan_prompt,
    answer_prompt,
    evidence_plan_prompt,
    request_analysis_prompt,
)
from agent.tools.realtime_scraping import realtime_scraping
from agent.tools.sqlite_query import sqlite_query
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    EvidencePlan,
    InvestigationActionPlan,
    InvestigationRequest,
    InvestigationStatus,
    RequestAnalysis,
)


class InvestigationGraphState(TypedDict, total=False):
    investigation: dict[str, Any]
    capability_catalog: list[dict[str, Any]]
    db_report: dict[str, Any]
    collection_results: list[dict[str, Any]]
    documents: str
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
        evidence_model: Any = None,
        action_model: Any = None,
        answer_model: Any = None,
    ):
        self.analysis_model = analysis_model
        self.evidence_model = evidence_model
        self.action_model = action_model
        self.answer_model = answer_model

    def analysis(self) -> Any:
        if self.analysis_model is None:
            from agent.application.model_clients import get_structured_google_model

            self.analysis_model = get_structured_google_model(
                "gemini-3.5-flash", RequestAnalysis, temperature=0.0
            )
        return self.analysis_model

    def evidence(self) -> Any:
        if self.evidence_model is None:
            from agent.application.model_clients import get_structured_google_model

            self.evidence_model = get_structured_google_model(
                "gemini-3.5-flash", EvidencePlan, temperature=0.0
            )
        return self.evidence_model

    def action(self) -> Any:
        if self.action_model is None:
            from agent.application.model_clients import get_structured_google_model

            self.action_model = get_structured_google_model(
                "gemini-3.5-flash", InvestigationActionPlan, temperature=0.0
            )
        return self.action_model

    def answer(self) -> Any:
        if self.answer_model is None:
            from agent.application.model_clients import get_google_chat_model

            self.answer_model = get_google_chat_model("gemini-3.5-flash", temperature=0.0)
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


class InvestigationWorkflow:
    """도구 실행 전에 조사 계획을 확정하는 지휘자 실행기."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        store: InvestigationStore | None = None,
        models: InvestigationModels | None = None,
        capabilities: list[Any] | None = None,
        collection_tool: Any = None,
        query_tool: Any = None,
        now: Callable[[], datetime] | None = None,
    ):
        self.db_path = Path(db_path)
        from shared.db.database import Database

        Database(self.db_path)
        self.store = store or InvestigationStore(self.db_path)
        self.models = models or InvestigationModels()
        self.capabilities = capabilities or build_tool_capability_catalog()
        self.collection_tool = collection_tool or realtime_scraping
        self.query_tool = query_tool or sqlite_query
        self.now = now or (lambda: datetime.now().astimezone())
        self.graph = self._build_graph()

    def _save(self, investigation: InvestigationRequest) -> dict[str, Any]:
        self.store.save(investigation)
        return investigation.model_dump(mode="json")

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
        questions = analysis.clarification_questions
        unresolved = list(dict.fromkeys(analysis.unresolved_fields))
        if questions:
            unresolved = list(dict.fromkeys([*unresolved, *(item.field for item in questions)]))
        updated = existing.model_copy(
            update={
                "objective": analysis.objective,
                "deliverable": analysis.deliverable,
                "purpose": analysis.purpose,
                "constraints": analysis.constraints,
                "unresolved_fields": unresolved,
                "assumptions": analysis.assumptions,
                "clarification_questions": questions,
                "status": (
                    InvestigationStatus.AWAITING_CLARIFICATION
                    if unresolved
                    else InvestigationStatus.CHECKING_EVIDENCE
                ),
            }
        )
        return {"investigation": self._save(updated)}

    @staticmethod
    def _route_after_understand(state: InvestigationGraphState) -> str:
        investigation = InvestigationRequest.model_validate(state["investigation"])
        return "clarify" if investigation.unresolved_fields else "define_evidence"

    def _clarify(self, state: InvestigationGraphState) -> dict[str, Any]:
        investigation = InvestigationRequest.model_validate(state["investigation"])
        question = next(
            (
                item
                for item in investigation.clarification_questions
                if item.field in investigation.unresolved_fields
            ),
            None,
        )
        if question is None:
            raise ValueError("미확정 조건에 대응하는 확인 질문이 없습니다.")
        payload = {
            "needs_clarification": True,
            **question.model_dump(mode="json"),
            "missing_fields": list(investigation.unresolved_fields),
            "investigation_id": investigation.investigation_id,
        }
        emit_run_event(
            "clarification_required",
            RunPhase.CLARIFICATION,
            question.question,
            status=RunStatus.WAITING_INPUT,
            data=payload,
        )
        return {
            "clarification": payload,
            "final_answer": question.question,
            "run_status": RunStatus.WAITING_INPUT.value,
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
                                "request": investigation.model_dump(mode="json"),
                                "tool_capabilities": state["capability_catalog"],
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
                "evidence_requirements": plan.requirements,
                "status": InvestigationStatus.CHECKING_EVIDENCE,
            }
        )
        return {"investigation": self._save(updated)}

    def _inspect_evidence(self, state: InvestigationGraphState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = InvestigationRequest.model_validate(state["investigation"])
        emit_run_event("database_check", RunPhase.DATABASE, "DB에 필요한 근거가 있는지 확인하고 있습니다.")
        report = inspect_job_evidence(
            self.db_path,
            investigation.evidence_requirements,
            investigation.constraints,
        )
        updated = investigation.model_copy(
            update={
                "evidence_snapshot": report,
                "missing_evidence": report.get("missing_evidence", []),
                "evidence_document_ids": report.get("document_ids", []),
                "status": (
                    InvestigationStatus.ANSWERING
                    if report.get("sufficient")
                    else InvestigationStatus.PLANNING
                ),
            }
        )
        return {
            "investigation": self._save(updated),
            "db_report": report,
            "valid_ids": report.get("document_ids", []),
        }

    @staticmethod
    def _route_after_evidence(state: InvestigationGraphState) -> str:
        investigation = InvestigationRequest.model_validate(state["investigation"])
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
                                "request": investigation.model_dump(mode="json"),
                                "db_report": state.get("db_report", {}),
                                "tool_capabilities": state["capability_catalog"],
                            },
                            ensure_ascii=False,
                        )
                    ),
                ],
                "investigation_action_plan",
            ),
            InvestigationActionPlan,
        )
        allowed_steps = [step for step in plan.steps if step.tool_name == "realtime_scraping"][:4]
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
            "investigation": self._save(updated),
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
        raw_result = self.collection_tool.invoke(step.arguments)
        try:
            parsed_result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        except json.JSONDecodeError:
            parsed_result = {"raw_result": str(raw_result)}
        executed = [*investigation.executed_step_ids, step.step_id]
        steps = [
            item.model_copy(update={"status": "completed"}) if item.step_id == step.step_id else item
            for item in investigation.plan
        ]
        updated = investigation.model_copy(
            update={
                "executed_step_ids": executed,
                "plan": steps,
                "status": InvestigationStatus.VALIDATING,
            }
        )
        return {
            "investigation": self._save(updated),
            "collection_results": [*state.get("collection_results", []), parsed_result],
        }

    def _load_documents(self, state: InvestigationGraphState) -> dict[str, Any]:
        investigation = InvestigationRequest.model_validate(state["investigation"])
        ids = sorted(set(investigation.evidence_document_ids))
        if not ids:
            return {"documents": "", "valid_ids": []}
        selected = ",".join(str(item) for item in ids)
        query = (
            "SELECT id, url, company_name, position, job_category, experience_text, "
            "employment_type, location, posted_at, posted_at_text, tech_stack, main_tasks, "
            f"requirements, preferred, benefits, raw_ocr_text FROM jobs WHERE id IN ({selected})"
        )
        return {
            "documents": self.query_tool.invoke({"query": query}),
            "valid_ids": ids,
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
                            "request": investigation.model_dump(mode="json"),
                            "db_report": state.get("db_report", {}),
                            "collection_results": state.get("collection_results", []),
                            "cannot_proceed_reason": state.get("cannot_proceed_reason", ""),
                            "documents": state.get("documents", ""),
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
        self._save(updated)
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
            {"clarify": "clarify", "define_evidence": "define_evidence"},
        )
        workflow.add_edge("clarify", END)
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
        return workflow.compile()

    def run(
        self,
        query: str,
        *,
        conversation_id: str = "",
        investigation_id: str = "",
        clarification_answer: ClarificationAnswer | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if investigation_id:
            investigation = self.store.get(investigation_id)
            if investigation is None:
                raise ValueError("재개할 조사 상태를 찾을 수 없습니다.")
            if clarification_answer is not None:
                investigation = apply_clarification_answer(
                    investigation,
                    ClarificationAnswer.model_validate(clarification_answer),
                    today=self.now().date(),
                )
                self.store.save(investigation)
        else:
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
        return self.graph.invoke(state)


__all__ = ["InvestigationModels", "InvestigationWorkflow"]

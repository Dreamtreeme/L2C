"""DB 근거 충분성 검사와 부족 자료 수집 계획을 담당하는 노드."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from agent.application.evidence_service import inspect_job_evidence
from agent.observability.run_context import (
    emit_run_event,
    invoke_with_metrics,
    raise_if_cancelled,
)
from agent.observability.run_contracts import RunPhase
from agent.application.search_taxonomy_service import SearchTaxonomyService
from agent.graph.investigation_context import (
    InvestigationWorkerState,
    InvestigationModels,
    build_request_prompt_context,
    capabilities_for_investigation,
)
from agent.graph.investigation_evidence_policy import (
    apply_evidence_validation,
    build_evidence_validation_payload,
    compact_db_report,
    needs_semantic_evidence_validation,
    normalize_collection_steps,
    normalize_evidence_requirements,
)
from agent.prompts.investigation import (
    action_plan_prompt,
    evidence_plan_prompt,
    evidence_validation_prompt,
)
from agent.utils.model_payload import parse_model_payload
from shared.schema.investigation_schema import (
    EvidencePlan,
    EvidencePolicy,
    EvidenceValidation,
    InvestigationActionPlan,
    InvestigationRequest,
    InvestigationStatus,
)


class InvestigationEvidenceNodes:
    """필요 근거를 정의하고 DB 충분성 및 수집 계획을 결정한다."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        models: InvestigationModels,
        taxonomy_service: SearchTaxonomyService,
        now: Callable[[], datetime],
    ) -> None:
        self.db_path = Path(db_path)
        self.models = models
        self.taxonomy_service = taxonomy_service
        self.now = now

    def define_evidence(self, state: InvestigationWorkerState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = InvestigationRequest.model_validate(state["investigation"])
        emit_run_event("evidence_planning", RunPhase.PLANNING, "답변에 필요한 근거를 정리하고 있습니다.")
        plan = parse_model_payload(
            invoke_with_metrics(
                self.models.evidence(),
                [
                    SystemMessage(content=evidence_plan_prompt(self.now())),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "request": build_request_prompt_context(investigation),
                                "tool_capabilities": capabilities_for_investigation(
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
                "evidence_requirements": normalize_evidence_requirements(
                    plan,
                    investigation,
                    self.taxonomy_service,
                ),
                "status": InvestigationStatus.CHECKING_EVIDENCE,
            }
        )
        return {"investigation": updated.model_dump(mode="json")}

    def inspect_evidence(self, state: InvestigationWorkerState) -> dict[str, Any]:
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
            taxonomy_service=self.taxonomy_service,
        )
        if needs_semantic_evidence_validation(report):
            validation = parse_model_payload(
                invoke_with_metrics(
                    self.models.validation(),
                    [
                        SystemMessage(content=evidence_validation_prompt()),
                        HumanMessage(
                            content=json.dumps(
                                {
                                    "request": build_request_prompt_context(investigation),
                                    "candidate_groups": build_evidence_validation_payload(
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
            report = apply_evidence_validation(report, investigation, validation)
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
    def route_after_evidence(state: InvestigationWorkerState) -> str:
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

    def plan_actions(self, state: InvestigationWorkerState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = InvestigationRequest.model_validate(state["investigation"])
        emit_run_event("action_planning", RunPhase.PLANNING, "부족한 자료를 확보할 행동계획을 세우고 있습니다.")
        plan = parse_model_payload(
            invoke_with_metrics(
                self.models.action(),
                [
                    SystemMessage(content=action_plan_prompt()),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "request": build_request_prompt_context(investigation),
                                "db_report": compact_db_report(
                                    state.get("db_report", {})
                                ),
                                "tool_capabilities": capabilities_for_investigation(
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
        allowed_steps = normalize_collection_steps(
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
    def route_after_plan(state: InvestigationWorkerState) -> str:
        investigation = InvestigationRequest.model_validate(state["investigation"])
        return "execute" if investigation.plan else "load_documents"

__all__ = ["InvestigationEvidenceNodes"]

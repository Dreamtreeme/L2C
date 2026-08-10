"""DB 근거 충분성 검사와 부족 자료 수집 계획을 담당하는 노드."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from agent.observability.run_context import (
    emit_run_event,
    invoke_with_metrics,
    raise_if_cancelled,
)
from agent.observability.run_contracts import RunPhase
from agent.graph.investigation_context import (
    InvestigationState,
    InvestigationModels,
    build_request_prompt_context,
    collection_capabilities_for,
)
from agent.graph.investigation_evidence_policy import (
    apply_evidence_validation,
    build_database_lookup_evidence_plan,
    build_evidence_validation_payload,
    compact_db_report,
    needs_semantic_evidence_validation,
    normalize_evidence_requirements,
    select_collection_steps,
)
from agent.prompts.investigation import (
    action_plan_prompt,
    evidence_plan_prompt,
    evidence_validation_prompt,
)
from agent.utils.model_conversion import parse_model_payload
from shared.schema.investigation_schema import (
    EvidencePlan,
    EvidencePolicy,
    EvidenceValidation,
    InvestigationActionPlan,
    InvestigationPurpose,
    ToolCapability,
)


class InvestigationEvidenceNodes:
    """필요 근거를 정의하고 DB 충분성 및 수집 계획을 결정한다."""

    def __init__(
        self,
        *,
        models: InvestigationModels,
        taxonomy_service: Any,
        inspect_evidence: Callable[..., dict[str, Any]],
        capabilities: list[ToolCapability],
        now: Callable[[], datetime],
    ) -> None:
        self.models = models
        self.taxonomy_service = taxonomy_service
        self.inspect_evidence_data = inspect_evidence
        self.collection_capabilities = [
            item.model_dump(mode="json") for item in capabilities
        ]
        self.now = now

    def define_evidence(self, state: InvestigationState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = state["request"]["investigation"]
        emit_run_event(
            "evidence_planning",
            RunPhase.PLANNING,
            "답변에 필요한 근거를 정리하고 있습니다.",
        )
        if (
            investigation.evidence_policy == EvidencePolicy.DATABASE_ONLY
            and investigation.purpose == InvestigationPurpose.LOOKUP
        ):
            plan = build_database_lookup_evidence_plan(investigation)
        else:
            plan = parse_model_payload(
                invoke_with_metrics(
                    self.models.evidence(),
                    [
                        SystemMessage(content=evidence_plan_prompt(self.now())),
                        HumanMessage(
                            content=json.dumps(
                                {
                                    "request": build_request_prompt_context(
                                        investigation
                                    )
                                },
                                ensure_ascii=False,
                            )
                        ),
                    ],
                    "investigation_evidence_plan",
                ),
                EvidencePlan,
            )
        requirements = normalize_evidence_requirements(
            plan,
            self.taxonomy_service,
        )
        return {
            "evidence": {"requirements": requirements},
        }

    def inspect_evidence(self, state: InvestigationState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = state["request"]["investigation"]
        evidence = state["evidence"]
        execution = state["execution"]
        emit_run_event(
            "database_check",
            RunPhase.DATABASE,
            "DB에 필요한 근거가 있는지 확인하고 있습니다.",
        )
        collected_web_evidence = (
            investigation.evidence_policy == EvidencePolicy.WEB_REQUIRED
            and bool(execution.get("executed_step_ids"))
        )
        report = self.inspect_evidence_data(
            evidence.get("requirements", []),
            document_scope_ids=(
                execution.get("collection_document_ids", [])
                if collected_web_evidence
                else None
            ),
            force_semantic_review=collected_web_evidence,
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
                                    "request": build_request_prompt_context(
                                        investigation
                                    ),
                                    "candidate_groups": build_evidence_validation_payload(
                                        report,
                                        evidence.get("requirements", []),
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
            report = apply_evidence_validation(
                report,
                evidence.get("requirements", []),
                validation,
            )
        report["document_ids"] = list(report.get("document_ids", []))
        return {
            "evidence": {"db_report": report},
        }

    @staticmethod
    def route_after_evidence(state: InvestigationState) -> str:
        investigation = state["request"]["investigation"]
        execution = state["execution"]
        if investigation.evidence_policy == EvidencePolicy.DATABASE_ONLY:
            return "answer"
        if (
            investigation.evidence_policy == EvidencePolicy.WEB_REQUIRED
            and not execution.get("executed_step_ids")
        ):
            return "plan_actions"
        if state["evidence"].get("db_report", {}).get("sufficient"):
            return "answer"
        pending = [
            step
            for step in execution.get("plan", [])
            if step.step_id not in execution.get("executed_step_ids", [])
        ]
        if pending:
            return "collect"
        if execution.get("plan"):
            return "answer"
        return "plan_actions"

    def plan_actions(self, state: InvestigationState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = state["request"]["investigation"]
        emit_run_event(
            "action_planning",
            RunPhase.PLANNING,
            "부족한 자료를 확보할 행동계획을 세우고 있습니다.",
        )
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
                                    state["evidence"].get("db_report", {})
                                ),
                                "tool_capabilities": collection_capabilities_for(
                                    self.collection_capabilities, investigation
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
        allowed_steps = select_collection_steps(
            plan,
            state["evidence"].get("requirements", []),
            self.collection_capabilities,
        )
        return {
            "execution": {
                "plan": allowed_steps,
                "cannot_proceed_reason": plan.cannot_proceed_reason,
            },
        }

    @staticmethod
    def route_after_plan(state: InvestigationState) -> str:
        return "collect" if state["execution"].get("plan") else "answer"


__all__ = ["InvestigationEvidenceNodes"]

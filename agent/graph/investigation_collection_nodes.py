"""확정된 수집 계획의 단일 도구 단계를 실행하는 노드."""

from __future__ import annotations

import json
from typing import Any

from agent.application.run_context import emit_run_event, raise_if_cancelled
from agent.application.run_contracts import RunPhase
from agent.graph.investigation_context import InvestigationGraphState
from shared.schema.investigation_schema import (
    InvestigationRequest,
    InvestigationStatus,
)


class InvestigationCollectionNodes:
    """승인된 계획에서 아직 실행하지 않은 수집 단계 하나를 실행한다."""

    def __init__(self, collection_tool: Any) -> None:
        self.collection_tool = collection_tool

    def execute(self, state: InvestigationGraphState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = InvestigationRequest.model_validate(state["investigation"])
        step = next(
            item
            for item in investigation.plan
            if item.step_id not in investigation.executed_step_ids
        )
        emit_run_event(
            "collection_started",
            RunPhase.COLLECTION,
            step.purpose or "계획한 채용공고 수집을 실행하고 있습니다.",
        )
        raw_result = self.collection_tool.invoke(
            step.arguments.model_dump(mode="json")
        )
        try:
            parsed_result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        except json.JSONDecodeError:
            parsed_result = {"raw_result": str(raw_result)}
        executed = [*investigation.executed_step_ids, step.step_id]
        persistence_validation = (
            parsed_result.get("persistence_validation", {})
            if isinstance(parsed_result, dict)
            else {}
        )
        observed_ids = {
            int(item["job_id"])
            for item in persistence_validation.get("persisted_items", [])
            if isinstance(item, dict) and item.get("job_id") is not None
        }
        observed_ids.update(
            int(job_id)
            for job_id in (
                parsed_result.get("observed_job_ids", [])
                if isinstance(parsed_result, dict)
                else []
            )
            if str(job_id).isdigit() and int(job_id) > 0
        )
        steps = [
            item.model_copy(update={"status": "completed"})
            if item.step_id == step.step_id
            else item
            for item in investigation.plan
        ]
        updated = investigation.model_copy(
            update={
                "executed_step_ids": executed,
                "collection_document_ids": sorted(
                    set(investigation.collection_document_ids) | observed_ids
                ),
                "plan": steps,
                "status": InvestigationStatus.VALIDATING,
            }
        )
        return {
            "investigation": updated.model_dump(mode="json"),
            "collection_results": [*state.get("collection_results", []), parsed_result],
        }


__all__ = ["InvestigationCollectionNodes"]

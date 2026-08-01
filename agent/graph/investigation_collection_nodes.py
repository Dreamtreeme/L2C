"""조사 계획의 다음 수집 단계를 실행한다."""

from __future__ import annotations

from typing import Any, Callable

from agent.application.run_context import emit_run_event, raise_if_cancelled
from agent.application.run_contracts import RunPhase
from agent.graph.investigation_context import InvestigationGraphState
from shared.schema.collection_intent import CollectionResult
from shared.schema.investigation_schema import (
    InvestigationRequest,
    InvestigationStatus,
)


class InvestigationCollectionNodes:
    def __init__(
        self,
        collect_jobs: Callable[[Any], CollectionResult | dict[str, Any]],
    ) -> None:
        self.collect_jobs = collect_jobs

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
        result = CollectionResult.model_validate(self.collect_jobs(step.arguments))
        executed = [*investigation.executed_step_ids, step.step_id]
        updated = investigation.model_copy(
            update={
                "executed_step_ids": executed,
                "collection_document_ids": sorted(
                    set(investigation.collection_document_ids)
                    | set(result.document_ids)
                ),
                "status": InvestigationStatus.VALIDATING,
            }
        )
        return {
            "investigation": updated.model_dump(mode="json"),
            "collection_results": [
                *state.get("collection_results", []),
                result.model_dump(mode="json"),
            ],
        }


__all__ = ["InvestigationCollectionNodes"]

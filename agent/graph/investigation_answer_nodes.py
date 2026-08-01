"""검증된 DB 문서를 읽고 사용자 답변을 생성하는 노드."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agent.application.evidence_service import load_job_evidence_documents
from agent.application.run_context import (
    emit_run_event,
    invoke_with_metrics,
    raise_if_cancelled,
)
from agent.application.run_contracts import RunPhase, RunStatus
from agent.graph.investigation_context import (
    InvestigationGraphState,
    InvestigationModels,
    build_request_prompt_context,
    message_text,
)
from agent.graph.investigation_evidence_policy import compact_db_report
from agent.prompts.investigation import answer_prompt
from agent.utils.job_fields import DETAIL_JOB_FIELDS
from shared.schema.investigation_schema import (
    InvestigationRequest,
    InvestigationStatus,
)


def compact_collection_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """답변에 영향을 주는 수집 결과와 저장 문서만 남긴다."""

    compact = []
    keys = (
        "message",
        "site",
        "site_name",
        "keyword",
        "target_count",
        "item_count",
        "persisted_count",
        "completion_status",
        "run_status",
        "search_scope_exhausted",
        "missing_count",
        "observed_job_ids",
    )
    for result in results or []:
        if not isinstance(result, dict):
            continue
        item = {key: result.get(key) for key in keys if key in result}
        validation = result.get("persistence_validation")
        if isinstance(validation, dict):
            item["persistence_validation"] = {
                "created_count": int(validation.get("created_count") or 0),
                "updated_count": int(validation.get("updated_count") or 0),
                "rejected_count": int(validation.get("rejected_count") or 0),
                "persisted_items": [
                    {
                        key: persisted.get(key)
                        for key in ("job_id", "company_name", "position", "operation")
                        if key in persisted
                    }
                    for persisted in (validation.get("persisted_items") or [])
                    if isinstance(persisted, dict)
                ],
            }
        compact.append(item)
    return compact

def build_answer_evidence_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """구조화 상세 필드가 완전한 문서는 중복 OCR 원문을 답변 입력에서 제외한다."""

    projected = []
    for document in documents or []:
        if not isinstance(document, dict):
            continue
        item = dict(document)
        if all(str(item.get(field) or "").strip() for field in DETAIL_JOB_FIELDS):
            item.pop("raw_ocr_text", None)
        projected.append(item)
    return projected

class InvestigationAnswerNodes:
    """검증된 문서만 로드하고 근거 식별자가 포함된 답변을 만든다."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        models: InvestigationModels,
    ) -> None:
        self.db_path = Path(db_path)
        self.models = models

    def load_documents(self, state: InvestigationGraphState) -> dict[str, Any]:
        investigation = InvestigationRequest.model_validate(state["investigation"])
        ids = sorted(set(investigation.evidence_document_ids))
        if not ids:
            return {"documents": [], "valid_ids": []}
        documents = load_job_evidence_documents(self.db_path, ids)
        return {
            "documents": [document.model_dump(mode="json") for document in documents],
            "valid_ids": [document.id for document in documents],
        }

    def answer(self, state: InvestigationGraphState) -> dict[str, Any]:
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
                            "request": build_request_prompt_context(investigation),
                            "db_report": compact_db_report(
                                state.get("db_report", {})
                            ),
                            "collection_results": compact_collection_results(
                                state.get("collection_results", [])
                            ),
                            "cannot_proceed_reason": state.get("cannot_proceed_reason", ""),
                            "documents": build_answer_evidence_documents(
                                state.get("documents", [])
                            ),
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            "investigation_answer",
        )
        answer = message_text(response)
        updated = investigation.model_copy(
            update={
                "final_answer": answer,
                "status": InvestigationStatus.COMPLETED,
            }
        )
        emit_run_event(
            "run_completed",
            RunPhase.COMPLETED,
            "답변을 완료했습니다.",
            status=RunStatus.COMPLETED,
        )
        return {
            "investigation": updated.model_dump(mode="json"),
            "final_answer": answer,
            "run_status": RunStatus.COMPLETED.value,
        }


__all__ = ["InvestigationAnswerNodes"]

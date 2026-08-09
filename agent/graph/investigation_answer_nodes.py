"""검증된 DB 문서를 읽고 사용자 답변을 생성하는 노드."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.observability.run_context import (
    emit_run_event,
    invoke_with_metrics,
    raise_if_cancelled,
)
from agent.graph.investigation_ports import StoredJobLoaderPort
from agent.observability.run_contracts import RunPhase, RunStatus
from agent.graph.investigation_context import (
    InvestigationState,
    InvestigationModels,
    build_request_prompt_context,
    message_text,
)
from agent.graph.investigation_evidence_policy import compact_db_report
from agent.prompts.investigation import answer_prompt
from agent.utils.job_fields import DETAIL_JOB_FIELDS
from shared.schema.collection_intent import CollectionResult
from shared.schema.investigation_schema import (
    InvestigationStatus,
)


def validate_citations(answer: str, valid_ids: list[int]) -> str:
    """답변의 job_id 인용이 실제 근거 문서에 포함됐는지 검증한다."""

    valid = {str(job_id) for job_id in valid_ids}

    def expand_group(match: re.Match[str]) -> str:
        citation_ids = re.findall(r"\d+", match.group(1))
        return " ".join(f"[job_id:{job_id}]" for job_id in citation_ids)

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1) in valid else "[출처 확인 불가]"

    normalized = re.sub(
        r"\[job_id:(\d+(?:\s*,\s*\d+)+)\]",
        expand_group,
        answer,
    )
    return re.sub(r"\[job_id:(\d+)\]", replace, normalized)


def compact_collection_results(
    results: list[CollectionResult],
) -> list[dict[str, Any]]:
    """답변에 영향을 주는 수집 결과와 저장 문서만 남긴다."""

    compact = []
    keys = (
        "status",
        "message",
        "site",
        "site_name",
        "search_keyword",
        "target_count",
        "collected_count",
        "resolved_count",
        "persisted_count",
        "created_count",
        "updated_count",
        "rejected_count",
        "scope_exhausted",
        "observed_job_ids",
    )
    for collection_result in results or []:
        result = collection_result.model_dump(mode="json")
        item = {key: result.get(key) for key in keys if key in result}
        item["persisted_items"] = [
            {
                key: persisted.get(key)
                for key in ("job_id", "company_name", "position", "operation")
                if key in persisted
            }
            for persisted in result.get("persisted_items", [])
            if isinstance(persisted, dict)
        ]
        compact.append(item)
    return compact


def build_answer_evidence_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
        models: InvestigationModels,
        load_documents: StoredJobLoaderPort,
    ) -> None:
        self.models = models
        self.load_evidence_documents = load_documents

    def load_documents(self, state: InvestigationState) -> dict[str, Any]:
        investigation = state["request"]["investigation"]
        ids = sorted(set(investigation.evidence_document_ids))
        if not ids:
            return {"evidence": {"documents": [], "valid_ids": []}}
        documents = self.load_evidence_documents(ids)
        return {
            "evidence": {
                "documents": [
                    document.model_dump(mode="json") for document in documents
                ],
                "valid_ids": [document.id for document in documents],
            }
        }

    def answer(self, state: InvestigationState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = state["request"]["investigation"]
        evidence = state["evidence"]
        execution = state["execution"]
        emit_run_event(
            "answering_started",
            RunPhase.ANSWERING,
            "검증된 근거로 답변을 정리하고 있습니다.",
        )
        response = invoke_with_metrics(
            self.models.answer(),
            [
                SystemMessage(content=answer_prompt()),
                HumanMessage(
                    content=json.dumps(
                        {
                            "request": build_request_prompt_context(investigation),
                            "db_report": compact_db_report(
                                evidence.get("db_report", {})
                            ),
                            "collection_results": compact_collection_results(
                                execution.get("collection_results", [])
                            ),
                            "cannot_proceed_reason": execution.get(
                                "cannot_proceed_reason",
                                "",
                            ),
                            "documents": build_answer_evidence_documents(
                                evidence.get("documents", [])
                            ),
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            "investigation_answer",
        )
        answer = validate_citations(
            message_text(response),
            [int(item) for item in evidence.get("valid_ids", [])],
        )
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
            "request": {"investigation": updated},
            "answer": {"final_answer": answer},
            "execution": {"run_status": RunStatus.COMPLETED.value},
        }


__all__ = ["InvestigationAnswerNodes", "validate_citations"]

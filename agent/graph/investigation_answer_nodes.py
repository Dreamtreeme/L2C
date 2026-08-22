"""검증된 DB 문서를 읽고 사용자 답변을 생성하는 노드."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

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
)
from agent.graph.investigation_evidence_policy import compact_db_report
from agent.prompts.investigation import answer_prompt
from shared.schema.collection_intent import CollectionResult
from shared.schema.jd_schema import JOB_DETAIL_FIELDS, JobField, StoredJob
from shared.schema.investigation_schema import (
    AnswerEvidenceRef,
    GroundedAnswer,
    GroundedAnswerDraft,
    GroundedAnswerDraftLine,
    GroundedAnswerLine,
    EvidencePolicy,
    InvestigationPurpose,
)
from shared.schema.run_schema import RunStatus


_LIST_EVIDENCE_FIELDS = {
    JobField.TECH_STACK,
    JobField.MAIN_TASKS,
    JobField.REQUIREMENTS,
    JobField.PREFERRED,
    JobField.BENEFITS,
}


def _evidence_value(
    reference: AnswerEvidenceRef,
    documents_by_id: dict[int, dict[str, Any]],
) -> Any | None:
    document = documents_by_id.get(reference.document_id)
    if document is None:
        return None
    value = document.get(reference.field.value)
    if reference.field in _LIST_EVIDENCE_FIELDS:
        if not isinstance(value, list) or reference.item_index is None:
            return None
        if reference.item_index >= len(value):
            return None
        item = value[reference.item_index]
        return item if str(item or "").strip() else None
    if reference.item_index is not None or value in (None, "", [], {}):
        return None
    return value


def _answer_line_references(
    line: GroundedAnswerDraftLine,
    documents_by_id: dict[int, dict[str, Any]],
    citations: list[AnswerEvidenceRef],
    citation_ids_by_pointer: dict[tuple[int, JobField, int | None], int],
) -> list[int]:
    """초안 포인터를 실제 DB 값이 포함된 인용으로 변환한다."""

    citation_ids: list[int] = []
    for pointer in line.evidence:
        if line.kind == "detail" and line.document_id != pointer.document_id:
            continue
        key = (pointer.document_id, pointer.field, pointer.item_index)
        citation_id = citation_ids_by_pointer.get(key)
        if citation_id is not None:
            citation_ids.append(citation_id)
            continue
        reference = AnswerEvidenceRef(
            citation_id=len(citations) + 1,
            document_id=pointer.document_id,
            field=pointer.field,
            item_index=pointer.item_index,
        )
        evidence_value = _evidence_value(reference, documents_by_id)
        if evidence_value is None:
            continue
        reference = reference.model_copy(
            update={"evidence_text": str(evidence_value).strip()}
        )
        citation_ids_by_pointer[key] = reference.citation_id
        citations.append(reference)
        citation_ids.append(reference.citation_id)
    return list(dict.fromkeys(citation_ids))


def validate_grounded_answer(
    answer: GroundedAnswerDraft,
    documents: list[dict[str, Any]],
    *,
    require_evidence: bool = True,
    maximum_document_sections: int | None = None,
) -> GroundedAnswer:
    """존재하는 DB 필드와 항목을 가리키는 문장만 답변에 남긴다."""

    documents_by_id = {
        int(document["id"]): document
        for document in documents
        if int(document.get("id") or 0) > 0
    }
    allowed_document_sections: list[int] = []
    citation_ids_by_pointer: dict[tuple[int, JobField, int | None], int] = {}
    citations: list[AnswerEvidenceRef] = []
    lines: list[GroundedAnswerLine] = []
    for line in answer.lines:
        document_id = line.document_id
        if document_id is not None and document_id not in documents_by_id:
            continue
        if line.kind == "detail" and document_id is None and not line.title.strip():
            continue
        if (
            line.kind == "detail"
            and document_id is not None
            and document_id not in allowed_document_sections
            and maximum_document_sections is not None
            and len(allowed_document_sections) >= maximum_document_sections
        ):
            continue
        citation_ids = _answer_line_references(
            line,
            documents_by_id,
            citations,
            citation_ids_by_pointer,
        )
        if require_evidence and line.kind != "caveat" and not citation_ids:
            continue
        if line.kind == "detail" and document_id is not None:
            if document_id not in allowed_document_sections:
                allowed_document_sections.append(document_id)
        lines.append(
            GroundedAnswerLine(
                kind=line.kind,
                document_id=document_id,
                title=line.title.strip(),
                text=line.text.strip(),
                citation_ids=citation_ids,
            )
        )

    return GroundedAnswer(lines=lines, citations=citations)


def _citation_suffix(
    line: GroundedAnswerLine,
    citations_by_id: dict[int, AnswerEvidenceRef],
) -> str:
    document_ids = list(
        dict.fromkeys(
            citations_by_id[citation_id].document_id
            for citation_id in line.citation_ids
            if citation_id in citations_by_id
        )
    )
    if not document_ids:
        return ""
    return " " + " ".join(
        f"[job_id:{document_id}]" for document_id in document_ids
    )


def render_grounded_answer(
    answer: GroundedAnswer,
    documents: list[dict[str, Any]],
) -> str:
    """검증된 구조를 기존 채팅 UI가 표시할 수 있는 Markdown으로 만든다."""

    documents_by_id = {
        int(document["id"]): document
        for document in documents
        if int(document.get("id") or 0) > 0
    }
    citations_by_id = {
        citation.citation_id: citation for citation in answer.citations
    }
    blocks = [
        f"{line.text}{_citation_suffix(line, citations_by_id)}"
        for line in answer.lines
        if line.kind == "overview"
    ]
    detail_groups: list[tuple[int | None, str, list[GroundedAnswerLine]]] = []
    detail_group_indexes: dict[tuple[int | None, str], int] = {}
    for line in answer.lines:
        if line.kind != "detail":
            continue
        key = (line.document_id, "" if line.document_id is not None else line.title)
        if key not in detail_group_indexes:
            detail_group_indexes[key] = len(detail_groups)
            detail_groups.append((line.document_id, line.title, []))
        detail_groups[detail_group_indexes[key]][2].append(line)

    document_number = 0
    for document_id, section_title, lines in detail_groups:
        if document_id is not None:
            document_number += 1
            document = documents_by_id[document_id]
            heading = " - ".join(
                value
                for value in (
                    str(document.get("company_name") or "").strip(),
                    str(document.get("position") or "").strip(),
                )
                if value
            ) or f"공고 {document_id}"
            title = f"### {document_number}. {heading} [job_id:{document_id}]"
        else:
            title = f"### {section_title}"
        claims = "\n".join(
            f"- {line.text}{_citation_suffix(line, citations_by_id)}"
            for line in lines
        )
        blocks.append(f"{title}\n{claims}")
    caveat_lines = [line for line in answer.lines if line.kind == "caveat"]
    if caveat_lines:
        caveats = "\n".join(
            f"- {line.text}{_citation_suffix(line, citations_by_id)}"
            for line in caveat_lines
        )
        blocks.append(f"### 참고\n{caveats}")
    return "\n\n".join(blocks) or "검증 가능한 근거가 부족합니다."


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
        "stored_count",
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
        if all(str(item.get(field.value) or "").strip() for field in JOB_DETAIL_FIELDS):
            item.pop("raw_ocr_text", None)
        projected.append(item)
    return projected


def investigation_run_status(state: InvestigationState) -> RunStatus:
    """최종 근거와 수집 결과로 사용자 요청의 완료 상태를 한 번만 판정한다."""

    investigation = state["request"]["investigation"]
    if investigation.evidence_policy in {
        EvidencePolicy.MODEL_KNOWLEDGE,
        EvidencePolicy.DATABASE_ONLY,
    }:
        return RunStatus.COMPLETED

    report = state["evidence"].get("db_report", {})
    if report.get("sufficient"):
        return RunStatus.COMPLETED

    results = list(state["execution"].get("collection_results", []))
    if not results:
        return RunStatus.PARTIAL

    verified_results = [
        result
        for result in results
        if result.resolved_count > 0
        or (
            result.scope_exhausted
            and result.worker_finished
            and not result.error_code
        )
    ]
    failed_results = [
        result
        for result in results
        if result.status == "failed"
        and not (
            result.scope_exhausted
            and result.worker_finished
            and not result.error_code
        )
    ]
    if failed_results and not verified_results:
        return RunStatus.FAILED
    if len(verified_results) == len(results) and not report.get("missing_evidence"):
        return RunStatus.COMPLETED
    return RunStatus.PARTIAL


def _terminal_run_event(status: RunStatus) -> tuple[str, RunPhase, str]:
    if status == RunStatus.COMPLETED:
        return "run_completed", RunPhase.COMPLETED, "답변을 완료했습니다."
    if status == RunStatus.PARTIAL:
        return "run_partial", RunPhase.PARTIAL, "확보한 근거 범위에서 답변했습니다."
    return "run_failed", RunPhase.FAILED, "검증 가능한 근거를 확보하지 못했습니다."


class InvestigationAnswerNodes:
    """검증된 문서만 로드하고 근거 식별자가 포함된 답변을 만든다."""

    def __init__(
        self,
        *,
        models: InvestigationModels,
        load_documents: Callable[[list[int]], Sequence[StoredJob]],
    ) -> None:
        self.models = models
        self.load_evidence_documents = load_documents

    def _load_documents(self, state: InvestigationState) -> list[dict[str, Any]]:
        ids = list(
            dict.fromkeys(
                int(document_id)
                for document_id in state["evidence"]
                .get("db_report", {})
                .get("document_ids", [])
                if int(document_id) > 0
            )
        )
        if not ids:
            return []
        documents = self.load_evidence_documents(ids)
        documents_by_id = {
            int(document.id): document
            for document in documents
            if int(document.id) > 0
        }
        return [
            documents_by_id[document_id].model_dump(mode="json")
            for document_id in ids
            if document_id in documents_by_id
        ]

    def answer(self, state: InvestigationState) -> dict[str, Any]:
        raise_if_cancelled()
        investigation = state["request"]["investigation"]
        evidence = state["evidence"]
        execution = state["execution"]
        documents = self._load_documents(state)
        emit_run_event(
            "answering_started",
            RunPhase.ANSWERING,
            "검증된 근거로 답변을 정리하고 있습니다.",
        )
        use_low_thinking = (
            investigation.purpose == InvestigationPurpose.LOOKUP and bool(documents)
        )
        response = invoke_with_metrics(
            self.models.answer(
                thinking_level="low" if use_low_thinking else None,
            ),
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
                            "documents": build_answer_evidence_documents(documents),
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            "investigation_answer",
        )
        grounded_answer = validate_grounded_answer(
            GroundedAnswerDraft.model_validate(response),
            documents,
            require_evidence=(
                investigation.evidence_policy != EvidencePolicy.MODEL_KNOWLEDGE
                and (
                    bool(documents)
                    or investigation.evidence_policy != EvidencePolicy.DATABASE_ONLY
                )
            ),
            maximum_document_sections=investigation.constraints.result_limit,
        )
        answer = render_grounded_answer(grounded_answer, documents)
        run_status = investigation_run_status(state)
        event_name, event_phase, event_message = _terminal_run_event(run_status)
        emit_run_event(
            event_name,
            event_phase,
            event_message,
            status=run_status,
        )
        return {
            "answer": {
                "final_answer": answer,
                "grounded_answer": grounded_answer,
                "run_status": run_status.value,
            }
        }


__all__ = [
    "InvestigationAnswerNodes",
    "investigation_run_status",
    "render_grounded_answer",
    "validate_grounded_answer",
]

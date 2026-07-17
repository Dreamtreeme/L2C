"""프론트엔드와 외부 에이전트가 조회할 수 있는 백엔드 JSON 계약."""

from __future__ import annotations

from typing import Any

from agent.application.run_contracts import ChatFinalPayload, ChatRequest, RunEvent
from shared.schema.agent_contract import CollectionToolArguments, EvidenceDocument
from shared.schema.collection_intent import CollectionIntent
from shared.schema.investigation_schema import RequestAnalysis, TaxonomyResolution


def build_backend_contract_manifest() -> dict[str, Any]:
    """실행 코드의 Pydantic 모델에서 계약을 생성한다."""

    return {
        "version": 3,
        "transport": {
            "chat_endpoint": "/api/chat",
            "media_type": "text/event-stream",
            "frames": ["PROCESSING", "EVENT", "FINAL", "ERROR", "DONE"],
        },
        "schemas": {
            "chat_request": ChatRequest.model_json_schema(),
            "request_analysis": RequestAnalysis.model_json_schema(),
            "taxonomy_resolution": TaxonomyResolution.model_json_schema(),
            "collection_intent": CollectionIntent.model_json_schema(),
            "collection_tool_arguments": CollectionToolArguments.model_json_schema(),
            "evidence_document": EvidenceDocument.model_json_schema(),
            "run_event": RunEvent.model_json_schema(),
            "chat_final": ChatFinalPayload.model_json_schema(),
        },
    }


__all__ = ["build_backend_contract_manifest"]

"""사용자 요청의 실행 상태와 관측 이벤트 계약."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from shared.schema.investigation_schema import ClarificationAnswer
from shared.schema.run_schema import RunStatus


class RunPhase(str, Enum):
    RECEIVED = "received"
    DATABASE = "database"
    PLANNING = "planning"
    CLARIFICATION = "clarification"
    COLLECTION = "collection"
    VALIDATION = "validation"
    PERSISTENCE = "persistence"
    ANSWERING = "answering"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunEvent(BaseModel):
    """UI와 로그가 함께 사용하는 실행 진행 이벤트."""

    run_id: str
    event: str
    phase: RunPhase
    status: RunStatus = RunStatus.RUNNING
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    """사용자 채팅 실행 요청."""

    query: str
    resume_run_id: str | None = None
    conversation_id: str = ""
    investigation_id: str = ""
    clarification_answer: ClarificationAnswer | None = None


class ChatResult(BaseModel):
    """실행 저장소와 UI가 그대로 공유하는 채팅 결과."""

    run_id: str
    status: RunStatus
    text: str = ""
    clarification: dict[str, Any] | None = None
    investigation_id: str = ""
    resumed_from_run_id: str | None = None
    resume_mode: str = ""
    conversation_id: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)


class ChatStartedPayload(BaseModel):
    """채팅 실행이 등록된 직후 반환하는 식별자."""

    run_id: str


class ChatErrorPayload(BaseModel):
    """SSE ERROR 프레임의 JSON 본문."""

    run_id: str
    message: str


@dataclass(frozen=True, slots=True)
class ChatStreamFrame:
    """애플리케이션 서비스가 전송 계층에 넘기는 스트림 프레임."""

    kind: Literal["processing", "event", "final", "error", "done"]
    payload: ChatStartedPayload | RunEvent | ChatResult | ChatErrorPayload | None = None


RunEventSink = Callable[[RunEvent], None]


def new_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


__all__ = [
    "RunEvent",
    "RunEventSink",
    "RunPhase",
    "ChatErrorPayload",
    "ChatRequest",
    "ChatResult",
    "ChatStartedPayload",
    "ChatStreamFrame",
    "new_run_id",
]

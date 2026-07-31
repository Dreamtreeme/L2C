"""백엔드 요청 실행 상태와 진행 이벤트 계약."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from shared.schema.investigation_schema import ClarificationAnswer


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunPhase(str, Enum):
    RECEIVED = "received"
    DATABASE = "database"
    PLANNING = "planning"
    CLARIFICATION = "clarification"
    COLLECTION = "collection"
    REVIEW = "review"
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
    """POST /api/chat 요청 본문."""

    query: str
    resume_run_id: str | None = None
    conversation_id: str = ""
    investigation_id: str = ""
    clarification_answer: ClarificationAnswer | None = None


class ChatFinalPayload(BaseModel):
    """SSE FINAL 프레임의 JSON 본문."""

    run_id: str
    text: str = ""
    status: str = RunStatus.COMPLETED.value
    clarification: dict[str, Any] | None = None
    investigation_id: str = ""
    resumed_from_run_id: str | None = None
    resume_mode: str = ""
    conversation_id: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)


class ChatErrorPayload(BaseModel):
    """SSE ERROR 프레임의 JSON 본문."""

    run_id: str
    message: str


RunEventSink = Callable[[RunEvent], None]


def new_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


__all__ = [
    "RunEvent",
    "RunEventSink",
    "RunPhase",
    "RunStatus",
    "ChatErrorPayload",
    "ChatFinalPayload",
    "ChatRequest",
    "new_run_id",
]

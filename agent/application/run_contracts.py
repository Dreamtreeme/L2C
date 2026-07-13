"""백엔드 요청 실행 상태와 진행 이벤트 계약."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
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


RunEventSink = Callable[[RunEvent], None]


def new_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


__all__ = [
    "RunEvent",
    "RunEventSink",
    "RunPhase",
    "RunStatus",
    "new_run_id",
]

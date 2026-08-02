"""작업자 상태 행동의 실행 결과와 상태 변경 계약."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StateActionUpdate:
    """상태 행동이 선택적으로 갱신하는 작업자 필드."""

    job_card_queue: list[dict[str, Any]] | None = None
    job_results_memory: dict[str, Any] | None = None
    job_results_availability: dict[str, Any] | None = None
    job_detail_buffer: dict[str, Any] | None = None
    job_detail_coverage: dict[str, Any] | None = None
    job_detail_followup: dict[str, Any] | None = None


@dataclass(frozen=True)
class StateActionOutcome:
    """상태 행동의 공개 결과, 공고 데이터와 상태 변경을 분리한다."""

    result: dict[str, Any]
    jobs: dict[str, Any]
    state_update: StateActionUpdate = field(default_factory=StateActionUpdate)


__all__ = ["StateActionOutcome", "StateActionUpdate"]

"""작업자 그래프가 애플리케이션 데이터 기능을 호출하는 포트 계약."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.runtime.worker_contracts import WorkerState


DetailJobExtractor = Callable[[WorkerState, str], dict[str, Any]]
ExistingJobCardMarker = Callable[
    [list[dict[str, Any]], str],
    tuple[list[dict[str, Any]], list[dict[str, Any]]],
]
ExistingJobUrlLookup = Callable[[str, Any], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class WorkerDataServices:
    """상세 정제와 중복 조회 구현을 그래프에 주입한다."""

    extract_job_detail: DetailJobExtractor
    mark_existing_job_cards: ExistingJobCardMarker
    find_existing_job_url: ExistingJobUrlLookup


__all__ = ["WorkerDataServices"]

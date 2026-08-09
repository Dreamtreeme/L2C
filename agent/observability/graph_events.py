"""LangGraph custom stream을 백엔드 실행 이벤트로 연결한다."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from langgraph.config import get_stream_writer

from agent.observability.run_contracts import RunPhase
from agent.observability.run_context import current_run_context
from agent.utils.logger import logger


def _stream_writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return None


def write_graph_event(payload: dict[str, Any]) -> None:
    """그래프 스트림 안에서만 이벤트를 내고 직접 실행에서는 조용히 건너뛴다."""

    writer = _stream_writer()
    if writer is not None:
        writer(dict(payload))


@contextmanager
def graph_step(stage: str) -> Iterator[dict[str, Any]]:
    """노드 시작·종료를 같은 단계 이름과 단조 시계로 기록한다."""

    component = f"graph:{stage}"
    started = time.perf_counter()
    write_graph_event(
        {
            "schema_version": 1,
            "event": "graph_step_started",
            "stage": stage,
            "component": component,
        }
    )
    details: dict[str, Any] = {}
    try:
        yield details
    except BaseException as exc:
        write_graph_event(
            {
                "schema_version": 1,
                "event": "graph_step_finished",
                "stage": stage,
                "component": component,
                "duration_sec": round(time.perf_counter() - started, 6),
                "success": False,
                "failure_code": type(exc).__name__,
                "error": str(exc)[:300],
                **details,
            }
        )
        raise
    else:
        write_graph_event(
            {
                "schema_version": 1,
                "event": "graph_step_finished",
                "stage": stage,
                "component": component,
                "duration_sec": round(time.perf_counter() - started, 6),
                "success": True,
                **details,
            }
        )


def forward_graph_event(payload: Any) -> None:
    """custom stream 이벤트 하나를 SSE, 로컬 지표, 로그에 한 번만 반영한다."""

    if not isinstance(payload, dict):
        return
    event = str(payload.get("event") or "")
    if event not in {"graph_step_started", "graph_step_finished"}:
        return
    stage = str(payload.get("stage") or "unknown")
    component = str(payload.get("component") or f"graph:{stage}")
    context = current_run_context()
    if context is None:
        logger.info("Graph progress event", **payload)
        return

    data = dict(payload)
    data.pop("event", None)
    if event == "graph_step_finished":
        context.record_step(
            component,
            float(payload.get("duration_sec") or 0.0),
            log_event=False,
            **{
                key: value
                for key, value in data.items()
                if key not in {"component", "duration_sec", "stage"}
            },
            stage=stage,
        )
    context.emit(
        event,
        RunPhase.COLLECTION,
        f"작업자 단계 {stage} {'완료' if event.endswith('finished') else '시작'}",
        data=data,
    )


__all__ = ["forward_graph_event", "graph_step", "write_graph_event"]

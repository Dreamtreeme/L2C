"""LangSmith 추적을 애플리케이션 실행 계약과 분리하는 어댑터."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from agent.config import get_settings
from agent.utils.logger import logger


def langsmith_tracing_enabled() -> bool:
    """표준 LangSmith 설정을 기준으로 추적 활성 여부를 반환한다."""

    try:
        from langsmith.utils import tracing_is_enabled

        return bool(tracing_is_enabled())
    except Exception:
        return False


def langsmith_project_name() -> str:
    return get_settings().observability.langsmith_project.strip() or "l2c-local"


@contextmanager
def langsmith_trace(
    name: str,
    *,
    run_type: str = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: Sequence[str] | None = None,
) -> Iterator[Any | None]:
    """추적 실패가 실제 작업을 중단하지 않는 LangSmith 실행 범위."""

    if not langsmith_tracing_enabled():
        yield None
        return

    try:
        from langsmith import trace

        manager = trace(
            str(name),
            run_type=run_type,
            inputs=inputs or {},
            metadata=metadata or {},
            tags=[str(tag) for tag in (tags or ()) if str(tag)],
            project_name=langsmith_project_name(),
        )
        run = manager.__enter__()
    except Exception as exc:
        logger.warning("LangSmith trace setup failed", trace_name=name, error=str(exc))
        yield None
        return

    try:
        yield run
    except BaseException as exc:
        try:
            manager.__exit__(type(exc), exc, exc.__traceback__)
        except Exception as trace_exc:
            logger.warning(
                "LangSmith trace finalization failed",
                trace_name=name,
                error=str(trace_exc),
            )
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception as exc:
            logger.warning(
                "LangSmith trace finalization failed",
                trace_name=name,
                error=str(exc),
            )


def finish_langsmith_trace(
    run: Any | None,
    *,
    outputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    error: str = "",
) -> None:
    """현재 trace에 결과와 토큰 사용량을 기록한다."""

    if run is None:
        return
    try:
        if metadata:
            run.add_metadata(metadata)
        if usage:
            run.usage_metadata = dict(usage)
        run.end(outputs=outputs or {}, error=error or None)
    except Exception as exc:
        logger.warning(
            "LangSmith trace update failed",
            trace_name=str(getattr(run, "name", "")),
            error=str(exc),
        )


def publish_langsmith_feedback(
    trace_id: str,
    feedback: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """E2E에서 계산한 결정론적 점수를 root trace에 연결한다."""

    if not trace_id:
        return {"status": "skipped", "reason": "trace_id_missing", "published": 0}
    if not langsmith_tracing_enabled():
        return {"status": "skipped", "reason": "tracing_disabled", "published": 0}
    if not get_settings().observability.langsmith_e2e_feedback:
        return {"status": "skipped", "reason": "feedback_disabled", "published": 0}

    published = 0
    try:
        from langsmith import Client

        client = Client()
        for item in feedback:
            key = str(item.get("key") or "").strip()
            if not key:
                continue
            kwargs: dict[str, Any] = {
                "trace_id": trace_id,
                "key": key,
                "source_info": {"source": "l2c_e2e"},
            }
            if item.get("score") is not None:
                kwargs["score"] = item["score"]
            if item.get("value") is not None:
                kwargs["value"] = item["value"]
            if item.get("comment"):
                kwargs["comment"] = str(item["comment"])
            client.create_feedback(**kwargs)
            published += 1
        timeout = get_settings().observability.langsmith_flush_timeout_sec
        client.flush(timeout=timeout)
        return {"status": "published", "reason": "", "published": published}
    except Exception as exc:
        logger.warning("LangSmith feedback publishing failed", error=str(exc))
        return {
            "status": "failed",
            "reason": str(exc)[:300],
            "published": published,
        }


__all__ = [
    "finish_langsmith_trace",
    "langsmith_project_name",
    "langsmith_trace",
    "langsmith_tracing_enabled",
    "publish_langsmith_feedback",
]

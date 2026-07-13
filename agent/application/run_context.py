"""한 사용자 요청의 로그, 시간, LLM 사용량을 같은 실행 식별자로 묶는다."""

from __future__ import annotations

import inspect
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.callbacks.usage import get_usage_metadata_callback
from structlog.contextvars import bound_contextvars

from agent.application.run_contracts import (
    RunEvent,
    RunEventSink,
    RunPhase,
    RunStatus,
    new_run_id,
)
from agent.application.llm_cost import estimate_llm_cost
from agent.utils.logger import logger


class RunCancelled(RuntimeError):
    """사용자가 현재 실행의 중단을 요청했습니다."""


def _usage_value(usage: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(usage.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _usage_details(usage: dict[str, Any], key: str) -> dict[str, int]:
    raw = usage.get(key)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for name, value in raw.items():
        try:
            out[str(name)] = max(0, int(value or 0))
        except (TypeError, ValueError):
            continue
    return out


def normalize_usage(usage: dict[str, Any]) -> dict[str, Any]:
    input_tokens = _usage_value(usage, "input_tokens") or _usage_value(usage, "prompt_tokens")
    output_tokens = _usage_value(usage, "output_tokens") or _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens") or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_token_details": _usage_details(usage, "input_token_details"),
        "output_token_details": _usage_details(usage, "output_token_details"),
    }


def _merge_usage(target: dict[str, Any], usage: dict[str, Any]) -> None:
    normalized = normalize_usage(usage)
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        target[key] = int(target.get(key) or 0) + int(normalized.get(key) or 0)
    for detail_key in ("input_token_details", "output_token_details"):
        details = target.setdefault(detail_key, {})
        for name, value in normalized.get(detail_key, {}).items():
            details[name] = int(details.get(name) or 0) + int(value or 0)


@dataclass
class RunContext:
    run_id: str
    query: str = ""
    event_sink: RunEventSink | None = None
    started_at: float = field(default_factory=time.perf_counter)
    usage_callback: UsageMetadataCallbackHandler | None = None
    step_metrics: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def emit(
        self,
        event: str,
        phase: RunPhase,
        message: str = "",
        *,
        status: RunStatus = RunStatus.RUNNING,
        data: dict[str, Any] | None = None,
    ) -> RunEvent:
        item = RunEvent(
            run_id=self.run_id,
            event=event,
            phase=phase,
            status=status,
            message=message,
            data=data or {},
        )
        logger.info(
            "Run event",
            run_event=event,
            phase=phase.value,
            status=status.value,
            message=message,
            **(data or {}),
        )
        if self.event_sink is not None:
            try:
                self.event_sink(item)
            except Exception as exc:
                logger.debug("Run event sink failed", error=str(exc), run_event=event)
        return item

    def record_step(self, component: str, duration_sec: float, **data: Any) -> None:
        metric = {
            "component": component,
            "duration_sec": round(max(0.0, float(duration_sec)), 6),
            **data,
        }
        with self._lock:
            self.step_metrics.append(metric)
        logger.info("Runtime step completed", **metric)

    def record_llm_call(
        self,
        component: str,
        provider: str,
        model: str,
        usage: dict[str, Any],
        duration_sec: float,
        *,
        success: bool = True,
        error: str = "",
    ) -> None:
        metric = {
            "component": component,
            "provider": provider,
            "model": model or "unknown",
            **normalize_usage(usage),
            "duration_sec": round(max(0.0, float(duration_sec)), 6),
            "success": bool(success),
            "error": error[:300],
        }
        with self._lock:
            self.llm_calls.append(metric)
        logger.info("LLM call completed", **metric)

    def snapshot(self) -> dict[str, Any]:
        langchain_by_model = {
            str(model): normalize_usage(dict(usage or {}))
            for model, usage in dict(
                getattr(self.usage_callback, "usage_metadata", {}) or {}
            ).items()
        }
        custom_by_model: dict[str, dict[str, Any]] = {}
        call_langchain_by_model: dict[str, dict[str, Any]] = {}
        with self._lock:
            steps = [dict(item) for item in self.step_metrics]
            calls = [dict(item) for item in self.llm_calls]
        for call in calls:
            if call.get("provider") == "langchain":
                _merge_usage(
                    call_langchain_by_model.setdefault(call.get("model") or "unknown", {}),
                    call,
                )
                continue
            model_usage = custom_by_model.setdefault(call.get("model") or "unknown", {})
            _merge_usage(model_usage, call)

        totals: dict[str, Any] = {}
        billable_by_model: dict[str, dict[str, Any]] = {}
        observed_langchain_models = set(langchain_by_model) | set(call_langchain_by_model)
        for model in observed_langchain_models:
            outer_usage = langchain_by_model.get(model, {})
            local_usage = call_langchain_by_model.get(model, {})
            usage = outer_usage if normalize_usage(outer_usage)["total_tokens"] else local_usage
            if not normalize_usage(usage)["total_tokens"]:
                continue
            _merge_usage(billable_by_model.setdefault(model, {}), usage)
        for model, usage in custom_by_model.items():
            if not normalize_usage(usage)["total_tokens"]:
                continue
            _merge_usage(billable_by_model.setdefault(model, {}), usage)
        for usage in billable_by_model.values():
            _merge_usage(totals, usage)
        return {
            "run_id": self.run_id,
            "duration_sec": round(max(0.0, time.perf_counter() - self.started_at), 6),
            "steps": steps,
            "llm": {
                "totals": normalize_usage(totals),
                "by_model": langchain_by_model,
                "custom_by_model": custom_by_model,
                "billable_by_model": billable_by_model,
                "cost": estimate_llm_cost(billable_by_model),
                "calls": calls,
            },
        }


_CURRENT_RUN_CONTEXT: ContextVar[RunContext | None] = ContextVar(
    "l2c_run_context",
    default=None,
)


def current_run_context() -> RunContext | None:
    return _CURRENT_RUN_CONTEXT.get()


def raise_if_cancelled() -> None:
    context = current_run_context()
    if context is None:
        return
    from agent.application.run_registry import get_run_registry

    if get_run_registry().is_cancel_requested(context.run_id):
        raise RunCancelled(f"run cancelled: {context.run_id}")


@contextmanager
def run_context(
    *,
    run_id: str | None = None,
    query: str = "",
    event_sink: RunEventSink | None = None,
    prefix: str = "run",
) -> Iterator[tuple[RunContext, bool]]:
    """중첩 호출이면 기존 문맥을 재사용하고, 최상위 호출만 새 문맥을 만든다."""

    existing = current_run_context()
    if existing is not None:
        yield existing, False
        return

    resolved_run_id = run_id or new_run_id(prefix)
    with get_usage_metadata_callback() as usage_callback:
        context = RunContext(
            run_id=resolved_run_id,
            query=query,
            event_sink=event_sink,
            usage_callback=usage_callback,
        )
        token = _CURRENT_RUN_CONTEXT.set(context)
        with bound_contextvars(run_id=resolved_run_id):
            context.emit(
                "run_started",
                RunPhase.RECEIVED,
                "요청을 접수했습니다.",
                data={"query": query},
            )
            try:
                yield context, True
            finally:
                _CURRENT_RUN_CONTEXT.reset(token)


@contextmanager
def measure_step(component: str, **data: Any) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - started
        context = current_run_context()
        if context is not None:
            context.record_step(component, duration, **data)
        else:
            logger.info(
                "Runtime step completed",
                component=component,
                duration_sec=round(duration, 6),
                **data,
            )


def emit_run_event(
    event: str,
    phase: RunPhase,
    message: str = "",
    *,
    status: RunStatus = RunStatus.RUNNING,
    data: dict[str, Any] | None = None,
) -> RunEvent | None:
    context = current_run_context()
    if context is None:
        return None
    return context.emit(event, phase, message, status=status, data=data)


def _supports_invoke_config(runnable: Any) -> bool:
    try:
        parameters = inspect.signature(runnable.invoke).parameters.values()
    except (TypeError, ValueError, AttributeError):
        return True
    return any(
        parameter.name == "config" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def invoke_with_metrics(runnable: Any, inputs: Any, component: str) -> Any:
    """LangChain 호출을 실행하고 구성 요소별 토큰과 시간을 기록한다."""

    raise_if_cancelled()
    callback = UsageMetadataCallbackHandler()
    context = current_run_context()
    config = {
        "callbacks": [callback],
        "metadata": {
            "run_id": context.run_id if context else "",
            "component": component,
        },
    }
    started = time.perf_counter()
    try:
        if _supports_invoke_config(runnable):
            result = runnable.invoke(inputs, config=config)
        else:
            result = runnable.invoke(inputs)
    except Exception as exc:
        duration = time.perf_counter() - started
        if context is not None:
            context.record_llm_call(
                component,
                "langchain",
                "unknown",
                {},
                duration,
                success=False,
                error=str(exc),
            )
        raise

    duration = time.perf_counter() - started
    usage_by_model = dict(callback.usage_metadata or {})
    if context is not None:
        if usage_by_model:
            for model, usage in usage_by_model.items():
                context.record_llm_call(
                    component,
                    "langchain",
                    str(model),
                    dict(usage or {}),
                    duration,
                )
        else:
            context.record_llm_call(component, "langchain", "unknown", {}, duration)
    raise_if_cancelled()
    return result


def record_external_llm_usage(
    *,
    component: str,
    provider: str,
    model: str,
    usage: dict[str, Any],
    duration_sec: float,
    success: bool = True,
    error: str = "",
) -> None:
    context = current_run_context()
    if context is None:
        return
    context.record_llm_call(
        component,
        provider,
        model,
        usage,
        duration_sec,
        success=success,
        error=error,
    )


def record_graph_state_metrics(state: dict[str, Any]) -> None:
    """LangGraph 상태에 누적된 노드 시간을 최상위 실행 요약에 병합한다."""

    context = current_run_context()
    if context is None:
        return
    for item in state.get("step_durations", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            duration = float(item.get("duration") or 0.0)
        except (TypeError, ValueError):
            continue
        data = {
            key: value
            for key, value in item.items()
            if key not in {"node", "duration"}
        }
        context.record_step(
            f"graph:{str(item.get('node') or 'unknown')}",
            duration,
            **data,
        )


__all__ = [
    "RunContext",
    "RunCancelled",
    "current_run_context",
    "emit_run_event",
    "invoke_with_metrics",
    "measure_step",
    "normalize_usage",
    "record_external_llm_usage",
    "record_graph_state_metrics",
    "raise_if_cancelled",
    "run_context",
]

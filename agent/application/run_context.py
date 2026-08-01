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
from langchain_core.messages import BaseMessageChunk
from structlog.contextvars import bound_contextvars

from agent.application.run_contracts import (
    RunEvent,
    RunEventSink,
    RunPhase,
    RunStatus,
    new_run_id,
)
from agent.application.llm_cost import estimate_llm_cost
from agent.observability.langsmith_adapter import (
    finish_langsmith_trace,
    langsmith_project_name,
    langsmith_trace,
    langsmith_tracing_enabled,
)
from agent.observability.stages import stage_for_component
from agent.utils.logger import logger


class RunCancelled(RuntimeError):
    """사용자가 현재 실행의 중단을 요청했습니다."""


class RunDeadlineExceeded(TimeoutError):
    """전체 사용자 요청의 실행 제한시간을 초과했습니다."""


class ModelRequestTimeout(TimeoutError):
    """단일 외부 모델 요청의 제한시간을 초과했습니다."""


def _failure_code(exc: BaseException) -> str:
    if isinstance(exc, RunCancelled):
        return "run_cancelled"
    if isinstance(exc, RunDeadlineExceeded):
        return "run_deadline_exceeded"
    if isinstance(exc, ModelRequestTimeout):
        return "model_request_timeout"
    if isinstance(exc, TimeoutError):
        return "timeout"
    return type(exc).__name__.removesuffix("Error").casefold() + "_error"


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
    deadline_monotonic: float | None = None
    usage_callback: UsageMetadataCallbackHandler | None = None
    trace_metadata: dict[str, Any] = field(default_factory=dict)
    trace_tags: list[str] = field(default_factory=list)
    step_metrics: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    langsmith_trace_id: str = ""
    outcome_status: str = RunStatus.RUNNING.value
    last_phase: str = RunPhase.RECEIVED.value
    failure_stage: str = ""
    failure_code: str = ""
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
        self.last_phase = phase.value
        if status != RunStatus.RUNNING:
            self.outcome_status = status.value
        if status == RunStatus.FAILED:
            self.failure_stage = phase.value
            self.failure_code = str((data or {}).get("failure_code") or event or "run_failed")
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

    def set_outcome(
        self,
        status: str | RunStatus,
        *,
        failure_stage: str = "",
        failure_code: str = "",
    ) -> None:
        resolved = status.value if isinstance(status, RunStatus) else str(status or "")
        if resolved:
            self.outcome_status = resolved
        if resolved in {"success", RunStatus.COMPLETED.value} and not (
            failure_stage or failure_code
        ):
            self.failure_stage = ""
            self.failure_code = ""
        if failure_stage:
            self.failure_stage = str(failure_stage)
        if failure_code:
            self.failure_code = str(failure_code)

    def record_step(
        self,
        component: str,
        duration_sec: float,
        *,
        log_event: bool = True,
        **data: Any,
    ) -> None:
        metric = {
            "component": component,
            "stage": str(data.pop("stage", "") or stage_for_component(component)),
            "duration_sec": round(max(0.0, float(duration_sec)), 6),
            **data,
        }
        with self._lock:
            self.step_metrics.append(metric)
        if data.get("success") is False:
            self.failure_stage = str(component)
            self.failure_code = str(data.get("failure_code") or "step_failed")
        if log_event:
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
            "deadline_remaining_sec": (
                round(
                    max(0.0, self.deadline_monotonic - time.perf_counter()),
                    6,
                )
                if self.deadline_monotonic is not None
                else None
            ),
            "steps": steps,
            "llm": {
                "totals": normalize_usage(totals),
                "by_model": langchain_by_model,
                "custom_by_model": custom_by_model,
                "billable_by_model": billable_by_model,
                "cost": estimate_llm_cost(billable_by_model),
                "calls": calls,
            },
            "outcome": {
                "status": self.outcome_status,
                "last_phase": self.last_phase,
                "failure_stage": self.failure_stage,
                "failure_code": self.failure_code,
            },
            "langsmith": {
                "enabled": langsmith_tracing_enabled(),
                "project": langsmith_project_name(),
                "trace_id": self.langsmith_trace_id,
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
    if (
        context.deadline_monotonic is not None
        and time.perf_counter() >= context.deadline_monotonic
    ):
        raise RunDeadlineExceeded(
            f"run deadline exceeded: {context.run_id}"
        )


@contextmanager
def run_context(
    *,
    run_id: str | None = None,
    query: str = "",
    event_sink: RunEventSink | None = None,
    prefix: str = "run",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    deadline_sec: float | None = None,
) -> Iterator[tuple[RunContext, bool]]:
    """중첩 호출이면 기존 문맥을 재사용하고, 최상위 호출만 새 문맥을 만든다."""

    existing = current_run_context()
    if existing is not None:
        existing.trace_metadata.update(metadata or {})
        for tag in tags or []:
            if tag and tag not in existing.trace_tags:
                existing.trace_tags.append(tag)
        yield existing, False
        return

    resolved_run_id = run_id or new_run_id(prefix)
    if deadline_sec is None:
        from agent.config import get_settings

        deadline_sec = get_settings().execution.run_deadline_sec
    resolved_deadline_sec = max(0.0, float(deadline_sec))
    with get_usage_metadata_callback() as usage_callback:
        context = RunContext(
            run_id=resolved_run_id,
            query=query,
            event_sink=event_sink,
            usage_callback=usage_callback,
            deadline_monotonic=(
                time.perf_counter() + resolved_deadline_sec
                if resolved_deadline_sec > 0
                else None
            ),
            trace_metadata=dict(metadata or {}),
            trace_tags=[str(tag) for tag in (tags or []) if str(tag)],
        )
        root_metadata = {
            "run_id": resolved_run_id,
            "run_kind": prefix,
            **context.trace_metadata,
        }
        root_tags = ["l2c", prefix, *context.trace_tags]
        with langsmith_trace(
            f"l2c.{prefix}",
            run_type="chain",
            inputs={"query": query},
            metadata=root_metadata,
            tags=root_tags,
        ) as root_trace:
            if root_trace is not None:
                context.langsmith_trace_id = str(
                    getattr(root_trace, "trace_id", None)
                    or getattr(root_trace, "id", "")
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
                except BaseException as exc:
                    context.set_outcome(
                        RunStatus.CANCELLED if isinstance(exc, RunCancelled) else RunStatus.FAILED,
                        failure_stage=context.failure_stage or context.last_phase,
                        failure_code=context.failure_code or _failure_code(exc),
                    )
                    raise
                finally:
                    snapshot = context.snapshot()
                    totals = dict((snapshot.get("llm") or {}).get("totals") or {})
                    finish_langsmith_trace(
                        root_trace,
                        outputs={
                            "run_id": resolved_run_id,
                            "status": context.outcome_status,
                            "duration_sec": snapshot.get("duration_sec", 0.0),
                            "step_count": len(snapshot.get("steps") or []),
                            "llm_call_count": len((snapshot.get("llm") or {}).get("calls") or []),
                            "total_tokens": int(totals.get("total_tokens") or 0),
                        },
                        metadata={
                            **context.trace_metadata,
                            "outcome": context.outcome_status,
                            "last_phase": context.last_phase,
                            "failure_stage": context.failure_stage,
                            "failure_code": context.failure_code,
                        },
                    )
                    _CURRENT_RUN_CONTEXT.reset(token)


@dataclass
class StepObservation:
    """실행 중 확인된 단계 결과를 종료 시점 계측에 합친다."""

    data: dict[str, Any] = field(default_factory=dict)

    def update(self, **data: Any) -> None:
        self.data.update(data)


@contextmanager
def observe_step(component: str, **data: Any) -> Iterator[StepObservation]:
    """로컬 메트릭과 LangSmith child trace를 같은 실행 구간에서 기록한다."""

    observation = StepObservation()
    started = time.perf_counter()
    stage = str(data.get("stage") or stage_for_component(component))
    trace_metadata = {"component": component, "stage": stage, **data}
    with langsmith_trace(
        component,
        run_type="tool",
        inputs=data,
        metadata=trace_metadata,
        tags=["runtime-step", component],
    ) as step_trace:
        try:
            yield observation
        except BaseException as exc:
            observation.data["success"] = False
            observation.data.setdefault("error", str(exc)[:300])
            observation.data.setdefault("failure_code", _failure_code(exc))
            raise
        finally:
            duration = time.perf_counter() - started
            metric_data = {"stage": stage, **data, **observation.data}
            metric_data.setdefault("success", True)
            context = current_run_context()
            if context is not None:
                context.record_step(component, duration, **metric_data)
            else:
                logger.info(
                    "Runtime step completed",
                    component=component,
                    duration_sec=round(duration, 6),
                    **metric_data,
                )
            finish_langsmith_trace(
                step_trace,
                outputs={"success": metric_data.get("success") is not False, **observation.data},
                metadata=metric_data,
                error=str(metric_data.get("error") or "")
                if metric_data.get("success") is False
                else "",
            )


@contextmanager
def measure_step(component: str, **data: Any) -> Iterator[None]:
    with observe_step(component, **data):
        yield


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


def _is_timeout_exception(exc: BaseException) -> bool:
    names = {
        type(item).__name__.casefold()
        for item in type(exc).mro()
    }
    if any("timeout" in name for name in names):
        return True
    text = str(exc).casefold()
    return "timed out" in text or "deadline exceeded" in text


def _stream_result(
    runnable: Any,
    inputs: Any,
    config: dict[str, Any],
) -> tuple[Any, float]:
    """스트림 청크를 최종 응답으로 조립하고 첫 청크 도착 시간을 반환한다."""

    started = time.perf_counter()
    chunks = (
        runnable.stream(inputs, config=config)
        if _supports_invoke_config(runnable)
        else runnable.stream(inputs)
    )
    result: Any = None
    message_chunk: BaseMessageChunk | None = None
    first_chunk_sec: float | None = None
    for chunk in chunks:
        if first_chunk_sec is None:
            first_chunk_sec = time.perf_counter() - started
        if isinstance(chunk, BaseMessageChunk):
            message_chunk = chunk if message_chunk is None else message_chunk + chunk
            result = message_chunk
        else:
            # 구조화 출력 파서는 완성도가 높아진 객체를 내보내므로 마지막 값을 사용합니다.
            result = chunk
    if result is None:
        raise RuntimeError("LLM stream returned no chunks")
    return result, float(first_chunk_sec or 0.0)


def invoke_with_metrics(
    runnable: Any,
    inputs: Any,
    component: str,
    *,
    stream: bool = False,
) -> Any:
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
    first_chunk_sec: float | None = None
    use_stream = stream and callable(getattr(runnable, "stream", None))
    try:
        if use_stream:
            result, first_chunk_sec = _stream_result(runnable, inputs, config)
        elif _supports_invoke_config(runnable):
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
        if _is_timeout_exception(exc):
            raise ModelRequestTimeout(
                f"{component} model request timed out"
            ) from exc
        raise

    duration = time.perf_counter() - started
    if use_stream:
        logger.info(
            "LLM stream completed",
            component=component,
            first_chunk_sec=round(float(first_chunk_sec or 0.0), 6),
            duration_sec=round(duration, 6),
        )
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


__all__ = [
    "RunContext",
    "RunCancelled",
    "RunDeadlineExceeded",
    "ModelRequestTimeout",
    "current_run_context",
    "emit_run_event",
    "invoke_with_metrics",
    "measure_step",
    "normalize_usage",
    "observe_step",
    "raise_if_cancelled",
    "run_context",
]

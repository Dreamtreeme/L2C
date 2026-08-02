"""단일 로컬 비전 작업자의 그래프 실행과 브라우저 생명주기를 관리한다."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from typing import Any, Callable, TypeVar

from langgraph.errors import GraphRecursionError

from agent.application.run_context import emit_run_event, measure_step
from agent.application.run_contracts import RunPhase
from agent.graph.action_request import build_action_event
from agent.observability.graph_events import forward_graph_event
from agent.utils.logger import logger


class OcrWorkerReadinessError(RuntimeError):
    """첫 화면 로직 전에 OCR 작업자를 준비하지 못한 경우."""


class WorkerStartScreenError(RuntimeError):
    """브라우저 첫 화면을 유효한 작업자 상태로 준비하지 못한 경우."""


WorkerResult = TypeVar("WorkerResult")


class WorkerExecutionService:
    """한 작업자의 잠금, 실행과 브라우저 정리를 한 경계에서 관리한다."""

    def __init__(
        self,
        worker_runtime: Any,
        worker_runner: Callable[..., WorkerResult],
    ) -> None:
        self.worker_runtime = worker_runtime
        self.worker_runner = worker_runner

    def run(self, *args: Any, **kwargs: Any) -> WorkerResult:
        """로컬 화면을 잠근 동안 작업자를 실행하고 브라우저를 정리한다."""

        with self.worker_runtime.execution_session():
            try:
                return self.worker_runner(
                    *args,
                    **kwargs,
                    worker_runtime=self.worker_runtime,
                )
            finally:
                try:
                    closed = self.worker_runtime.close_browser_after_run()
                    if not closed:
                        logger.warning("Browser cleanup did not close a browser")
                except Exception as exc:
                    logger.warning("Browser cleanup failed", error=str(exc))


def _ensure_ocr_ready(action_tools: Any) -> float:
    perception = getattr(action_tools, "perception", None)
    som_engine = getattr(perception, "som_engine", None)
    ensure_ready = getattr(som_engine, "ensure_ocr_worker_ready", None)
    if not callable(ensure_ready):
        raise OcrWorkerReadinessError("OCR worker readiness API is unavailable")

    started = time.perf_counter()
    ensure_ready()
    return time.perf_counter() - started


def _ensure_reasoning_models_ready(prepare_models: Callable[[], None]) -> float:
    started = time.perf_counter()
    prepare_models()
    return time.perf_counter() - started


def _open_browser_while_ocr_starts(
    action_tools: Any,
    *,
    site_slug: str,
    current_url: str,
    prepare_reasoning_models: Callable[[], None] | None = None,
) -> dict:
    """OCR·판단 모델 준비와 브라우저 열기를 겹치고 모두 끝난 뒤 반환한다."""

    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="worker-startup",
    ) as executor:
        ocr_context = copy_context()
        ocr_future = executor.submit(ocr_context.run, _ensure_ocr_ready, action_tools)
        reasoning_future = None
        if callable(prepare_reasoning_models):
            reasoning_context = copy_context()
            reasoning_future = executor.submit(
                reasoning_context.run,
                _ensure_reasoning_models_ready,
                prepare_reasoning_models,
            )
        browser_started = time.perf_counter()
        result = action_tools.open_browser(site=site_slug, current_url=current_url)
        browser_duration = time.perf_counter() - browser_started
        try:
            ocr_duration = ocr_future.result()
        except OcrWorkerReadinessError:
            raise
        except Exception as exc:
            raise OcrWorkerReadinessError("OCR worker failed to become ready") from exc
        reasoning_duration = 0.0
        if reasoning_future is not None:
            try:
                reasoning_duration = reasoning_future.result()
            except Exception as exc:
                logger.warning(
                    "Reasoning model warmup failed; lazy initialization will be used",
                    error=str(exc),
                )

    logger.info(
        "Worker startup barrier completed",
        site=site_slug,
        browser_duration_sec=round(browser_duration, 6),
        ocr_duration_sec=round(ocr_duration, 6),
        reasoning_model_duration_sec=round(reasoning_duration, 6),
        total_duration_sec=round(time.perf_counter() - started, 6),
    )
    return result


def run_graph_with_last_state(
    app: Any,
    initial_state: dict,
    recursion_limit: int,
) -> tuple[dict, bool]:
    """재귀 제한 예외가 발생해도 마지막 부분 상태를 보존한다."""

    last_state = initial_state
    try:
        for item in app.stream(
            initial_state,
            config={"recursion_limit": recursion_limit},
            stream_mode=["values", "custom"],
        ):
            from agent.application.run_context import raise_if_cancelled

            raise_if_cancelled()
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and item[0] in {"values", "custom"}
            ):
                mode, payload = item
                if mode == "custom":
                    forward_graph_event(payload)
                elif isinstance(payload, dict):
                    last_state = payload
        return last_state, False
    except GraphRecursionError as exc:
        logger.warning(
            "Graph recursion limit reached; preserving partial state",
            error=str(exc),
        )
        return last_state, True


def worker_start_url(profile) -> str:
    site = str(getattr(profile, "slug", "") or "").strip()
    if not site:
        return ""

    from agent.sites import get_official_site_url

    return get_official_site_url(site)


def prepare_worker_start_screen(
    initial_state: dict,
    site_profile,
    *,
    worker_runtime: Any = None,
) -> dict:
    """사이트 홈, OCR 작업자와 판단 모델을 병렬로 준비한다."""

    start_url = worker_start_url(site_profile)
    if not start_url:
        return initial_state

    try:
        from agent.graph.worker_resources import get_action_tools, prepare_reasoning_models

        runtime_context = (
            worker_runtime.activate()
            if worker_runtime is not None
            else nullcontext()
        )
        with runtime_context:
            action_tools = get_action_tools()
            site_slug = str(site_profile.slug or "").strip()
            result = _open_browser_while_ocr_starts(
                action_tools,
                site_slug=site_slug,
                current_url=str(initial_state.get("current_url") or ""),
                prepare_reasoning_models=prepare_reasoning_models,
            )
        prepared = dict(initial_state)
        prepared["current_url"] = start_url
        prepared["current_url_stale"] = True
        prepared["action_events"] = [
            build_action_event(
                0,
                {
                    "action": "open_browser",
                    "status": result.get("status", "unknown"),
                    "result": result.get("result"),
                    "args": {"url": start_url, "site": site_slug},
                    "screen_change_expected": True,
                },
            )
        ]
        logger.info("Worker resources prepared", start_url=start_url)
        return prepared
    except OcrWorkerReadinessError as exc:
        logger.error(
            "OCR worker was not ready; worker graph will not start",
            error=str(exc),
        )
        raise
    except Exception as exc:
        logger.error(
            "Worker start screen preparation failed",
            error=str(exc),
        )
        raise WorkerStartScreenError(
            f"작업자 시작 화면을 준비하지 못했습니다: {exc}"
        ) from exc


def execute_worker_graph(
    initial_state: dict,
    site_profile,
    recursion_limit: int,
    *,
    worker_runtime: Any = None,
) -> tuple[dict, bool]:
    """그래프 구성, 시작 화면 준비, 실행을 하나의 작업자 경계로 묶는다."""

    if worker_runtime is None:
        from agent.graph.workflow import build_graph

        app = build_graph()
    else:
        app = worker_runtime.get_graph()
    emit_run_event(
        "worker_preparing_screen",
        RunPhase.COLLECTION,
        "브라우저 시작 화면을 준비하고 있습니다.",
    )
    with measure_step("worker_prepare_screen"):
        prepared_state = prepare_worker_start_screen(
            initial_state,
            site_profile,
            worker_runtime=worker_runtime,
        )
    emit_run_event(
        "worker_graph_started",
        RunPhase.COLLECTION,
        "화면을 탐색하고 공고를 수집하고 있습니다.",
    )
    with measure_step("worker_graph", recursion_limit=recursion_limit):
        return run_graph_with_last_state(
            app,
            prepared_state,
            recursion_limit,
        )


__all__ = [
    "execute_worker_graph",
    "OcrWorkerReadinessError",
    "prepare_worker_start_screen",
    "run_graph_with_last_state",
    "WorkerExecutionService",
    "WorkerStartScreenError",
    "worker_start_url",
]

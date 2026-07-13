"""단일 로컬 비전 작업자의 그래프 실행과 브라우저 생명주기를 관리한다."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from typing import Any, Callable, Iterator

from langgraph.errors import GraphRecursionError

from agent.application.run_context import emit_run_event, measure_step
from agent.application.run_contracts import RunPhase
from agent.utils.logger import logger

_WORKER_EXECUTION_LOCK = threading.RLock()


class OcrWorkerReadinessError(RuntimeError):
    """첫 화면 로직 전에 OCR 작업자를 준비하지 못한 경우."""


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
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="worker-startup") as executor:
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


@contextmanager
def worker_execution_session() -> Iterator[None]:
    """브라우저·Perception·OCR 작업자가 한 번에 한 요청만 처리하게 한다."""

    with _WORKER_EXECUTION_LOCK:
        yield


def run_graph_with_last_state(
    app: Any,
    initial_state: dict,
    recursion_limit: int,
) -> tuple[dict, bool]:
    """재귀 제한 예외가 발생해도 마지막 부분 상태를 보존한다."""

    last_state = initial_state
    try:
        for state in app.stream(
            initial_state,
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            from agent.application.run_context import raise_if_cancelled

            raise_if_cancelled()
            if isinstance(state, dict):
                last_state = state
        return last_state, False
    except GraphRecursionError as exc:
        logger.warning(
            "Graph recursion limit reached; preserving partial state",
            error=str(exc),
        )
        return last_state, True


def worker_preopen_enabled() -> bool:
    raw = os.getenv("VISION_WORKER_PREOPEN_BROWSER", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def worker_start_url(profile: dict) -> str:
    entry = profile.get("entry", {}) if isinstance(profile, dict) else {}
    manual = profile.get("manual", {}) if isinstance(profile, dict) else {}
    site = str(entry.get("slug") or manual.get("site") or "").strip()
    if not site:
        return ""

    from agent.sites import get_official_site_url

    return get_official_site_url(site)


def prepare_worker_start_screen(initial_state: dict, site_profile: dict) -> dict:
    """사이트 홈을 물리 입력으로 열고 첫 OCR 관찰 상태를 준비한다."""

    start_url = worker_start_url(site_profile)
    if not start_url or not worker_preopen_enabled():
        return initial_state

    try:
        from agent.graph.nodes import _get_action_tools, perception_node, prepare_reasoning_models

        action_tools = _get_action_tools()
        site_slug = str((site_profile.get("entry") or {}).get("slug") or "").strip()
        try:
            open_attempts = max(
                1,
                int(os.getenv("VISION_WORKER_START_OPEN_ATTEMPTS", "2")),
            )
        except ValueError:
            open_attempts = 2

        action_history = []
        observation_state = {
            **initial_state,
            "current_url": start_url,
            "current_url_stale": True,
            "pending_transition": {},
        }
        observation = {}
        for open_attempt in range(1, open_attempts + 1):
            if open_attempt == 1:
                result = _open_browser_while_ocr_starts(
                    action_tools,
                    site_slug=site_slug,
                    current_url=str(initial_state.get("current_url") or ""),
                    prepare_reasoning_models=prepare_reasoning_models,
                )
            else:
                result = action_tools.open_browser(
                    site=site_slug,
                    current_url=observation_state.get("current_url", ""),
                )
            action_history.append(
                {
                    "action": "open_browser",
                    "status": result.get("status", "unknown"),
                    "result": result.get("result"),
                    "args": {"url": start_url, "site": site_slug},
                    "screen_change_expected": True,
                }
            )
            observation = perception_node(
                {
                    **observation_state,
                    "action_history": action_history,
                },
                max_capture_attempts=1 if open_attempt == 1 else None,
            )
            if not observation.get("low_information_screen"):
                break
            if open_attempt >= open_attempts:
                break
            logger.info(
                "Worker start screen is still loading; reopening official site",
                site=site_slug,
                start_url=start_url,
                open_attempt=open_attempt,
                open_attempts=open_attempts,
            )
            observation_state.update(observation)
            observation_state["current_url"] = start_url
            observation_state["current_url_stale"] = True

        prepared = dict(initial_state)
        prepared.update(observation)
        prepared["current_url"] = observation.get("current_url") or start_url
        prepared["current_url_stale"] = bool(observation.get("current_url_stale", False))
        prepared["action_history"] = action_history
        prepared["last_action_screen_changed"] = False
        logger.info("Worker start screen prepared", start_url=start_url)
        return prepared
    except OcrWorkerReadinessError as exc:
        logger.error(
            "OCR worker was not ready; worker graph will not start",
            error=str(exc),
        )
        raise
    except Exception as exc:
        logger.warning(
            "Worker start screen preparation failed; falling back to reasoning",
            error=str(exc),
        )
        return initial_state


def execute_worker_graph(
    initial_state: dict,
    site_profile: dict,
    recursion_limit: int,
    *,
    prepare_screen: Callable[[dict, dict], dict] | None = None,
    run_graph: Callable[[Any, dict, int], tuple[dict, bool]] | None = None,
) -> tuple[dict, bool]:
    """그래프 구성, 시작 화면 준비, 실행을 하나의 작업자 경계로 묶는다."""

    from agent.graph.workflow import build_graph

    app = build_graph()
    emit_run_event(
        "worker_preparing_screen",
        RunPhase.COLLECTION,
        "브라우저 시작 화면을 준비하고 있습니다.",
    )
    with measure_step("worker_prepare_screen"):
        prepared_state = (prepare_screen or prepare_worker_start_screen)(
            initial_state,
            site_profile,
        )
    emit_run_event(
        "worker_graph_started",
        RunPhase.COLLECTION,
        "화면을 탐색하고 공고를 수집하고 있습니다.",
    )
    with measure_step("worker_graph", recursion_limit=recursion_limit):
        return (run_graph or run_graph_with_last_state)(
            app,
            prepared_state,
            recursion_limit,
        )


def close_browser_after_run() -> None:
    """설정에 따라 작업 종료 후 브라우저 창만 닫고 OCR 작업자는 유지한다."""

    if os.getenv("VISION_CLOSE_BROWSER_AFTER_RUN", "0").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        logger.info("Browser cleanup disabled")
        return
    try:
        from agent.graph import nodes

        action_tools = getattr(nodes, "_action_tools", None)
        if action_tools is None:
            logger.info("Browser cleanup skipped", reason="action_tools_not_initialized")
            return
        result = action_tools.close_browser()
        logger.info("Browser cleanup completed", result=result)
    except Exception as exc:
        logger.debug("Browser cleanup skipped", error=str(exc))


__all__ = [
    "close_browser_after_run",
    "execute_worker_graph",
    "OcrWorkerReadinessError",
    "prepare_worker_start_screen",
    "run_graph_with_last_state",
    "worker_execution_session",
    "worker_preopen_enabled",
    "worker_start_url",
]

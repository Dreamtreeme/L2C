"""수집 요청을 단일 로컬 비전 작업자 그래프에 연결한다."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, cast

from langgraph.errors import GraphRecursionError

from agent.application.duplicate_job_service import (
    existing_job_url_trace,
    mark_existing_job_cards,
)
from agent.application.job_review_service import review_job_draft
from agent.application.collection_request_builder import build_site_goal
from agent.config import get_settings
from agent.observability.graph_events import forward_graph_event
from agent.observability.run_context import (
    emit_run_event,
    measure_step,
    raise_if_cancelled,
)
from agent.observability.run_contracts import RunPhase
from agent.runtime.action_permissions import (
    build_public_collection_permission_contract,
)
from agent.runtime.tool_schema import ACTION_TOOL_SCHEMAS
from agent.runtime.vision_worker_runtime import (
    VisionWorkerRuntime,
    WorkerDependencies,
)
from agent.runtime.worker_contracts import (
    WorkerState,
    WorkerStateUpdate,
    apply_worker_state_update,
    build_action_event,
    create_worker_state,
)
from agent.runtime.worker_data_services import WorkerDataServices
from agent.recipe.store import ExperienceRuleStore
from agent.recipe.task_category import normalize_task_category
from agent.sites import get_official_site_url, load_site_profile
from agent.sites.profile import SiteProfile
from agent.utils.job_fields import required_job_fields
from agent.utils.logger import logger
from shared.schema.collection_intent import CollectionIntent
from shared.schema.collection_run import CollectionBatch
from shared.schema.feedback_schema import ExecutionEvent, WorkerSubmission


class OcrWorkerReadinessError(RuntimeError):
    """첫 화면 로직 전에 OCR 작업자를 준비하지 못한 경우."""


class WorkerStartScreenError(RuntimeError):
    """브라우저 첫 화면을 유효한 작업자 상태로 준비하지 못한 경우."""


def build_worker_data_services(db_path: str | Path) -> WorkerDataServices:
    """작업자 그래프의 데이터 포트를 애플리케이션 구현에 연결한다."""

    rule_store = ExperienceRuleStore(db_path)
    return WorkerDataServices(
        mark_existing_job_cards=partial(mark_existing_job_cards, db_path=db_path),
        find_existing_job_url=partial(existing_job_url_trace, db_path=db_path),
        load_experience_rules=rule_store.get_site_rules,
        record_recipe_replay=rule_store.record_replay_result,
        review_job_draft=review_job_draft,
    )


def _new_worker_run_id() -> str:
    return f"worker-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _resolve_collection_intent(
    collection_intent: CollectionIntent,
    site_profile: SiteProfile,
) -> CollectionIntent:
    resolved = collection_intent.model_copy(
        update={
            "site": site_profile.slug,
            "task_category": normalize_task_category(collection_intent.task_category),
        }
    )
    return resolved.model_copy(
        update={
            "required_fields": required_job_fields(resolved)
        }
    )


def _build_initial_state(
    collection_intent: CollectionIntent,
    site_profile: SiteProfile,
    run_id: str,
) -> WorkerState:
    return create_worker_state(
        build_site_goal(collection_intent, site_profile),
        request={
            "worker_run_id": run_id,
            "collection_intent": collection_intent,
            "action_permission_contract": (
                build_public_collection_permission_contract(
                    site_profile,
                    collection_intent,
                )
            ),
        },
    )


def _observed_job_ids(state: WorkerState) -> list[int]:
    return sorted(
        {
            int(item["job_id"])
            for item in (state["collection"].get("job_card_queue", []) or [])
            if isinstance(item, dict)
            and item.get("status") == "skipped"
            and str(item.get("job_id") or "").isdigit()
            and int(item["job_id"]) > 0
        }
    )


def _build_collection_batch(
    state: WorkerState,
    collection_intent: CollectionIntent,
    site_profile: SiteProfile,
    run_id: str,
    *,
    hit_recursion_limit: bool,
) -> CollectionBatch:
    captures = list(state["collection"].get("job_captures", []))
    collected_jobs = list(state["collection"].get("collected_jobs", []))
    rejected_items = [
        {
            "index": index,
            "url": review.url,
            "issues": [
                review.status.value,
                *review.issues,
                *(
                    [
                        "missing_fields:"
                        + ",".join(field.value for field in review.missing_fields)
                    ]
                    if review.missing_fields
                    else []
                ),
            ],
        }
        for index, review in enumerate(
            state["collection"].get("job_reviews", [])
        )
        if review.status.value in {"source_incomplete", "invalid_target"}
    ]
    is_finished = bool(state["lifecycle"].get("is_finished", False))
    run_status = "stopped"
    if hit_recursion_limit:
        run_status = "recursion_limit"
    if is_finished:
        run_status = "finished"
    action_events = [
        ExecutionEvent.model_validate(item)
        for item in state["transition"].get("action_events", []) or []
    ]
    observed_job_ids = _observed_job_ids(state)
    submission = WorkerSubmission(
        run_id=run_id,
        goal=state["request"].get("goal", "") or "",
        run_status=run_status,
        collected_count=len(collected_jobs),
        observed_job_ids=observed_job_ids,
        persisted_count=0,
        action_events=action_events,
        collection_intent=collection_intent,
        extracted_summary={
            "has_data": bool(collected_jobs),
            "job_count": len(collected_jobs),
            "observed_job_count": len(observed_job_ids),
            "current_url": state["observation"].get("current_url", "") or "",
            "action_count": len(action_events),
            "worker_stage": state["progress"].get("stage", "navigation"),
            "completion_reason": state["lifecycle"].get(
                "completion_reason", ""
            ),
            "job_results_availability": dict(
                state["collection"].get("job_results_availability", {}) or {}
            ),
        },
    )
    return CollectionBatch(
        submission=submission,
        job_captures=captures,
        collected_jobs=collected_jobs,
        rejected_items=rejected_items,
        site_name=site_profile.display_name,
    )


class WorkerExecutionService:
    """수집 요청을 작업자 그래프에 연결하고 결과 묶음을 구성한다."""

    def __init__(
        self,
        worker_runtime: VisionWorkerRuntime,
        data_services: WorkerDataServices,
    ) -> None:
        self.worker_runtime = worker_runtime
        self.data_services = data_services

    def run(
        self,
        collection_intent: CollectionIntent,
        run_id: str | None = None,
    ) -> CollectionBatch:
        """비전 런타임의 실행 세션 안에서 수집 작업을 수행한다."""

        with self.worker_runtime.execution_session():
            return self._run_collection(
                collection_intent,
                run_id=run_id,
            )

    def _run_collection(
        self,
        collection_intent: CollectionIntent,
        *,
        run_id: str | None,
    ) -> CollectionBatch:
        site_profile = load_site_profile(collection_intent.site)
        resolved_intent = _resolve_collection_intent(
            collection_intent,
            site_profile,
        )
        resolved_run_id = run_id or _new_worker_run_id()
        initial_state = _build_initial_state(
            resolved_intent,
            site_profile,
            resolved_run_id,
        )
        logger.info("비전 작업자 그래프 시작", site=site_profile.slug)
        final_state, hit_recursion_limit = execute_worker_graph(
            initial_state,
            site_profile,
            get_settings().vision.recursion_limit,
            worker_runtime=self.worker_runtime,
            data_services=self.data_services,
        )
        return _build_collection_batch(
            final_state,
            resolved_intent,
            site_profile,
            resolved_run_id,
            hit_recursion_limit=hit_recursion_limit,
        )


def _measure_startup(operation: Callable[[], None]) -> float:
    started = time.perf_counter()
    operation()
    return time.perf_counter() - started


def _open_browser_while_ocr_starts(
    action_tools: Any,
    worker_runtime: VisionWorkerRuntime,
    *,
    site_slug: str,
    current_url: str,
) -> dict:
    """OCR·판단 모델 준비와 브라우저 열기를 겹치고 모두 끝난 뒤 반환한다."""

    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="worker-startup",
    ) as executor:
        ocr_context = copy_context()
        ocr_future = executor.submit(
            ocr_context.run,
            _measure_startup,
            worker_runtime.ensure_ocr_worker_ready,
        )
        reasoning_context = copy_context()
        reasoning_future = executor.submit(
            reasoning_context.run,
            _measure_startup,
            lambda: worker_runtime.prepare_reasoning_models(ACTION_TOOL_SCHEMAS),
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
        try:
            reasoning_duration = reasoning_future.result()
        except Exception as exc:
            reasoning_duration = 0.0
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
    initial_state: WorkerState,
    recursion_limit: int,
    *,
    worker_runtime: VisionWorkerRuntime,
    data_services: WorkerDataServices,
) -> tuple[WorkerState, bool]:
    """재귀 제한 예외가 발생해도 마지막 부분 상태를 보존한다."""

    last_state = initial_state
    try:
        for item in app.stream(
            initial_state,
            config={"recursion_limit": recursion_limit},
            context=WorkerDependencies(
                vision=worker_runtime,
                data=data_services,
            ),
            stream_mode=["values", "custom"],
        ):
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
                    last_state = cast(WorkerState, payload)
        return last_state, False
    except GraphRecursionError as exc:
        logger.warning(
            "Graph recursion limit reached; preserving partial state",
            error=str(exc),
        )
        return last_state, True


def prepare_worker_start_screen(
    initial_state: WorkerState,
    site_profile: SiteProfile,
    *,
    worker_runtime: Any,
) -> WorkerState:
    """사이트 홈, OCR 작업자와 판단 모델을 병렬로 준비한다."""

    start_url = get_official_site_url(site_profile.slug)

    try:
        action_tools = worker_runtime.get_action_tools()
        site_slug = str(site_profile.slug or "").strip()
        result = _open_browser_while_ocr_starts(
            action_tools,
            worker_runtime,
            site_slug=site_slug,
            current_url=str(initial_state["observation"].get("current_url") or ""),
        )
        update: WorkerStateUpdate = {
            "observation": {
                "current_url": start_url,
                "current_url_stale": True,
            },
            "transition": {
                "action_events": [
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
                ],
            },
        }
        prepared = apply_worker_state_update(initial_state, update)
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
    initial_state: WorkerState,
    site_profile: SiteProfile,
    recursion_limit: int,
    *,
    worker_runtime: Any,
    data_services: WorkerDataServices,
) -> tuple[WorkerState, bool]:
    """그래프 구성, 시작 화면 준비, 실행을 하나의 작업자 경계로 묶는다."""

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
            worker_runtime=worker_runtime,
            data_services=data_services,
        )


__all__ = [
    "build_worker_data_services",
    "execute_worker_graph",
    "OcrWorkerReadinessError",
    "prepare_worker_start_screen",
    "run_graph_with_last_state",
    "WorkerExecutionService",
    "WorkerStartScreenError",
]

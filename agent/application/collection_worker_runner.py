"""비전 작업자 한 번의 실행과 재귀 한도 결과를 관리한다."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from agent.application.collection_request_builder import (
    build_site_goal,
)
from agent.application.worker_execution_service import (
    execute_worker_graph,
)
from agent.config import get_settings
from agent.runtime.worker_contracts import (
    create_worker_state,
)
from agent.runtime.vision_worker_runtime import VisionWorkerRuntime
from agent.runtime.worker_data_services import WorkerDataServices
from agent.recipe.task_category import (
    normalize_task_category,
)
from agent.runtime.action_permissions import (
    build_public_collection_permission_contract,
)
from agent.sites import load_site_profile
from agent.utils.job_fields import required_job_fields
from shared.schema.collection_intent import CollectionIntent
from shared.schema.collection_run import CollectionBatch
from shared.schema.feedback_schema import RecordedActionEvent, WorkerSubmission

logger = logging.getLogger(__name__)


def run_worker_once(
    collection_intent: CollectionIntent,
    run_id: str | None = None,
    *,
    worker_runtime: VisionWorkerRuntime,
    data_services: WorkerDataServices,
) -> CollectionBatch:
    """비전 작업자 한 번을 실행하고 검토 전 제출물을 반환한다."""

    site_profile = load_site_profile(collection_intent.site)
    site_slug = site_profile.slug
    site_name = site_profile.display_name
    run_id = (
        run_id
        or f"worker-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    resolved_intent = collection_intent.model_copy(
        update={
            "site": site_slug,
            "task_category": normalize_task_category(collection_intent.task_category),
        },
    )
    resolved_intent = resolved_intent.model_copy(
        update={
            "required_fields": required_job_fields(
                resolved_intent,
                profile_fields=site_profile.collection_policy.required_fields,
            )
        }
    )
    goal = build_site_goal(resolved_intent, site_profile)
    initial_state = create_worker_state(
        goal,
        request={
            "worker_run_id": run_id,
            "collection_intent": resolved_intent,
            "action_permission_contract": (
                build_public_collection_permission_contract(
                    site_profile,
                    resolved_intent,
                )
            ),
        },
    )

    logger.info(
        "비전 작업자 그래프 시작: site=%s",
        site_slug,
    )
    recursion_limit = get_settings().vision.recursion_limit
    final_state, hit_recursion_limit = execute_worker_graph(
        initial_state,
        site_profile,
        recursion_limit,
        worker_runtime=worker_runtime,
        data_services=data_services,
    )
    job_captures = list(final_state["collection"].get("job_captures", []))
    is_finished = bool(final_state["lifecycle"].get("is_finished", False))
    run_status = "stopped"
    if is_finished:
        run_status = "finished"
    elif hit_recursion_limit:
        run_status = "recursion_limit"
    observation = final_state["observation"]
    transition = final_state["transition"]
    collection = final_state["collection"]
    action_events = [
        RecordedActionEvent.model_validate(item)
        for item in transition.get("action_events", []) or []
    ]
    observed_job_ids = sorted(
        {
            int(item["job_id"])
            for item in (collection.get("job_card_queue", []) or [])
            if isinstance(item, dict)
            and item.get("status") == "skipped"
            and str(item.get("job_id") or "").isdigit()
            and int(item["job_id"]) > 0
        }
    )
    submission = WorkerSubmission(
        run_id=run_id,
        goal=final_state["request"].get("goal", "") or "",
        run_status=run_status,
        collected_count=len(job_captures),
        observed_job_ids=observed_job_ids,
        persisted_count=0,
        action_events=action_events,
        collection_intent=resolved_intent,
        extracted_summary={
            "has_data": bool(job_captures),
            "job_count": len(job_captures),
            "observed_job_count": len(observed_job_ids),
            "current_url": observation.get("current_url", "") or "",
            "action_count": len(action_events),
            "job_results_availability": dict(
                collection.get("job_results_availability", {}) or {}
            ),
        },
    )

    return CollectionBatch(
        submission=submission,
        job_captures=job_captures,
        site_name=site_name,
    )


__all__ = [
    "run_worker_once",
]

"""비전 작업자 실행 상태를 저장 가능한 제출물로 조립한다."""

from __future__ import annotations

import uuid
from datetime import datetime
from agent.runtime.worker_contracts import (
    WorkerState,
    action_event_feedback,
    action_event_recipe_steps,
    action_event_results,
    action_event_transitions,
)
from agent.runtime.job_collection import job_count
from shared.schema.feedback_schema import WorkerSubmission


def new_worker_run_id() -> str:
    return f"worker-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"


def build_worker_submission(
    final_state: WorkerState,
    *,
    run_status: str = "",
    hit_recursion_limit: bool = False,
    persisted_count: int = 0,
    run_id: str | None = None,
) -> WorkerSubmission:
    """작업자 그래프 실행 결과를 구조화된 제출물(WorkerSubmission)로 만든다."""
    request = final_state["request"]
    observation = final_state["observation"]
    transition = final_state["transition"]
    collection = final_state["collection"]
    collected_jobs = list(collection.get("collected_jobs", []))
    current_url = observation.get("current_url", "") or ""
    raw_recipe_params = request.get("recipe_params")
    recipe_params = (
        raw_recipe_params if isinstance(raw_recipe_params, dict) else {}
    )
    run_id = run_id or new_worker_run_id()
    action_events = list(transition.get("action_events", []) or [])
    recorded_steps = action_event_recipe_steps(action_events)
    feedback_episodes = action_event_feedback(action_events)
    transition_records = action_event_transitions(action_events)
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
    extracted_summary = {
        "has_data": bool(collected_jobs),
        "job_count": job_count(collected_jobs),
        "observed_job_count": len(observed_job_ids),
        "current_url": current_url,
        "action_count": len(action_event_results(action_events)),
        "job_results_availability": dict(
            collection.get("job_results_availability", {}) or {}
        ),
    }
    submission = WorkerSubmission(
        run_id=run_id,
        goal=request.get("goal", "") or "",
        run_status=run_status,
        is_finished=bool(
            final_state["lifecycle"].get("is_finished", False)
        ),
        hit_recursion_limit=bool(hit_recursion_limit),
        collected_count=job_count(collected_jobs),
        observed_job_ids=observed_job_ids,
        persisted_count=int(persisted_count or 0),
        recorded_steps=recorded_steps,
        feedback_episodes=feedback_episodes,
        transition_records=transition_records,
        collection_intent=dict(recipe_params.get("collection_intent") or {}),
        extracted_summary=extracted_summary,
    )
    return submission


__all__ = [
    "build_worker_submission",
    "new_worker_run_id",
]

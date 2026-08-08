"""비전 작업자 제출물을 만들고 관찰 가능한 실행 사실을 검증한다."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from agent.runtime.worker_contracts import (
    action_event_feedback,
    action_event_recipe_steps,
    action_event_results,
    action_event_transitions,
)
from agent.runtime.job_collection import job_items as _job_items
from shared.schema.feedback_schema import WorkerSubmission


def new_worker_run_id() -> str:
    return f"worker-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"


def build_worker_submission(
    final_state: dict[str, Any],
    *,
    run_status: str = "",
    hit_recursion_limit: bool = False,
    persisted_count: int = 0,
    run_id: str | None = None,
) -> WorkerSubmission:
    """작업자 그래프 실행 결과를 구조화된 제출물(WorkerSubmission)로 만든다."""
    extracted_jd = final_state.get("extracted_jd", {}) or {}
    jobs = _job_items(extracted_jd)
    current_url = final_state.get("current_url", "") or ""
    recipe_params = final_state.get("recipe_params", {}) if isinstance(final_state.get("recipe_params"), dict) else {}
    run_id = run_id or new_worker_run_id()
    action_events = list(final_state.get("action_events", []) or [])
    recorded_steps = action_event_recipe_steps(action_events)
    feedback_episodes = action_event_feedback(action_events)
    transition_records = action_event_transitions(action_events)
    observed_job_ids = sorted(
        {
            int(item["job_id"])
            for item in (final_state.get("job_card_queue", []) or [])
            if isinstance(item, dict)
            and item.get("status") == "skipped"
            and str(item.get("job_id") or "").isdigit()
            and int(item["job_id"]) > 0
        }
    )
    extracted_summary = {
        "has_data": bool(jobs),
        "job_count": len(jobs),
        "observed_job_count": len(observed_job_ids),
        "current_url": current_url,
        "action_count": len(action_event_results(action_events)),
        "job_results_availability": dict(final_state.get("job_results_availability", {}) or {}),
    }
    submission = WorkerSubmission(
        run_id=run_id,
        goal=final_state.get("goal", "") or "",
        run_status=run_status,
        is_finished=bool(final_state.get("is_finished", False)),
        hit_recursion_limit=bool(hit_recursion_limit),
        collected_count=len(jobs),
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

"""Worker graph 상태의 단일 생성 지점."""

from __future__ import annotations

from typing import Any

from agent.graph.state import GraphState
from shared.schema.agent_contract import DEFAULT_JOB_COLLECTION_FIELDS


def create_worker_state(goal: str = "", **overrides: Any) -> GraphState:
    """모든 worker 진입점에서 동일한 초기 상태를 만든다."""

    state: GraphState = {
        "goal": goal,
        "worker_run_id": "",
        "worker_attempt_index": 0,
        "current_capture_id": "",
        "capture_sequence": 0,
        "current_screenshot": "",
        "capture_quality": {},
        "raw_screen_signature": {},
        "analysis_mode": "",
        "ocr_complete": False,
        "ui_context": "",
        "current_url": "",
        "current_page_role": "",
        "current_url_stale": True,
        "low_information_screen": False,
        "low_information_capture_count": 0,
        "current_markers": [],
        "action_history": [],
        "recent_images": [],
        "marked_image": "",
        "screen_signature": {},
        "error_count": 0,
        "is_finished": False,
        "collected_data": [],
        "extracted_jd": {},
        "pending_action": None,
        "last_action_result": None,
        "execution_records": [],
        "recorded_steps": [],
        "feedback_episodes": [],
        "reflex_trace": {},
        "followup_action_trace": {},
        "reflex_transition_contracts": {},
        "active_reflex_recipe": {},
        "reflex_blocked_recipe_keys": [],
        "recipe_params": {},
        "job_collection_contract": {
            "required_fields": list(DEFAULT_JOB_COLLECTION_FIELDS),
        },
        "transition_request": {},
        "transition_result": {
            "status": "idle",
            "needs_ocr": False,
        },
        "transition_records": [],
        "transition_probe_unchanged": False,
        "job_card_queue": [],
        "job_results_memory": {},
        "active_job_card": {},
        "job_card_replay_trace": {},
        "job_card_selection_trace": {},
        "job_results_availability": {},
        "job_page_policy_trace": {},
        "job_detail_buffer": {},
        "job_detail_coverage": {},
        "job_detail_followup": {},
        "return_to_job_results": {},
        "pending_human_approval": False,
        "human_approval_request": {},
    }
    state.update(overrides)
    return state


__all__ = ["create_worker_state"]

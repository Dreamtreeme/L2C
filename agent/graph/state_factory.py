"""Worker graph 상태의 단일 생성 지점."""

from __future__ import annotations

from typing import Any

from agent.graph.state import GraphState


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
        "reflex_transition_contracts": {},
        "reflex_blocked_recipe_keys": [],
        "recipe_params": {},
        "pending_transition": {},
        "transition_status": "",
        "transition_outcome": "",
        "transition_source": "",
        "transition_reason": "",
        "transition_visual_change_detected": False,
        "transition_visual_change_ratio": None,
        "ocr_required": False,
        "observed_transition": {},
        "transition_observations": [],
        "result_card_queue": [],
        "result_page_memory": {},
        "active_result_card": {},
        "queue_replay_trace": {},
        "result_card_selector_trace": {},
        "result_availability": {},
        "page_policy_trace": {},
        "detail_ocr_buffer": {},
        "detail_followup_required": {},
        "pending_human_approval": False,
        "human_approval_request": {},
    }
    state.update(overrides)
    return state


__all__ = ["create_worker_state"]

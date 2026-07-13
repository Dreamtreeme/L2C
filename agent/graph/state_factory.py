"""Worker graph 상태의 단일 생성 지점."""

from __future__ import annotations

from typing import Any

from agent.graph.state import GraphState


def create_worker_state(goal: str = "", **overrides: Any) -> GraphState:
    """모든 worker 진입점에서 동일한 초기 상태를 만든다."""

    state: GraphState = {
        "goal": goal,
        "ui_context": "",
        "current_url": "",
        "current_page_role": "",
        "current_url_stale": True,
        "low_information_screen": False,
        "low_information_retry_count": 0,
        "current_markers": [],
        "action_history": [],
        "recent_images": [],
        "marked_image": "",
        "screen_signature": {},
        "error_count": 0,
        "is_finished": False,
        "collected_data": [],
        "extracted_jd": {},
        "last_action_result": None,
        "plan": [],
        "current_plan_step": 0,
        "step_durations": [],
        "last_action_screen_changed": True,
        "recorded_steps": [],
        "feedback_episodes": [],
        "reflex_hit": False,
        "reflex_trace": {},
        "reflex_transition_contracts": {},
        "reflex_blocked_recipe_keys": [],
        "recipe_params": {},
        "pending_transition": {},
        "transition_status": "",
        "transition_outcome": "",
        "transition_source": "",
        "transition_observations": [],
        "result_card_queue": [],
        "result_page_memory": {},
        "active_result_card": {},
        "queue_replay_hit": False,
        "queue_replay_trace": {},
        "page_policy_hit": False,
        "page_policy_trace": {},
        "detail_ocr_buffer": {},
        "pending_human_approval": False,
        "human_approval_request": {},
    }
    state.update(overrides)
    return state


__all__ = ["create_worker_state"]

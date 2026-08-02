from typing import Any, TypedDict

from agent.graph.action_request import ActionEvent, ActionRequest


class TransitionRequest(TypedDict, total=False):
    """화면 변경 행동 뒤 다음 캡처에서 확인할 전환 요청."""

    action_seq: int
    action: str
    from_capture_id: str
    source: str
    recipe_key: str
    recipe_transition_index: int
    recipe_transition_count: int
    transition_actions: list[str]
    expected_after_state: dict[str, Any]
    after_state_match: dict[str, Any]
    step: dict[str, Any]
    before_url: str
    before_page_role: str
    before_screenshot: str
    started_at: float
    execution_failed: bool
    failed_action: str


class TransitionResult(TransitionRequest, total=False):
    """전환 요청과 현재 캡처를 비교한 판정 결과."""

    status: str
    outcome: str
    reason: str
    visual_change_detected: bool
    visual_change_ratio: float | None
    needs_ocr: bool


class WorkerIdentityState(TypedDict, total=False):
    """작업 실행과 캡처를 연결하는 식별 상태."""

    worker_run_id: str
    current_capture_id: str
    ocr_capture_id: str
    capture_sequence: int
    goal: str


class ObservationState(TypedDict, total=False):
    """한 캡처에서 얻은 화면, OCR과 브라우저 상태."""

    current_screenshot: str
    capture_quality: dict[str, Any]
    raw_screen_signature: dict[str, Any]
    analysis_mode: str
    ocr_complete: bool
    previous_screen_observation: dict[str, Any]
    ui_context: str
    current_url: str
    current_page_role: str
    current_url_stale: bool
    low_information_screen: bool
    low_information_capture_count: int
    current_markers: list[dict[str, Any]]
    marked_image: str
    screen_signature: dict[str, Any]


class ActionExecutionState(TypedDict, total=False):
    """선택된 행동의 실행과 화면 전환 판정 상태."""

    action_events: list[ActionEvent]
    pending_action: ActionRequest | None
    error_count: int
    is_finished: bool
    transition_request: TransitionRequest
    transition_result: TransitionResult


class RecipeReplayState(TypedDict, total=False):
    """자율탐색 기록과 경험 기반 탐색 재생 상태."""

    reflex_trace: dict[str, Any]
    active_reflex_recipe: dict[str, Any]
    reflex_blocked_recipe_keys: list[str]
    recipe_params: dict[str, Any]


class JobCollectionState(TypedDict, total=False):
    """공고 목록 선택, 상세 판독과 결과 누적 상태."""

    extracted_jd: dict[str, Any]
    job_collection_contract: dict[str, Any]
    job_card_queue: list[dict[str, Any]]
    job_results_memory: dict[str, Any]
    job_card_selection_trace: dict[str, Any]
    job_results_availability: dict[str, Any]
    job_detail_buffer: dict[str, Any]
    job_detail_coverage: dict[str, Any]
    job_detail_followup: dict[str, Any]
    return_to_job_results: dict[str, Any]


class ActionSafetyState(TypedDict, total=False):
    """작업 권한과 사용자 승인 대기 상태."""

    action_permission_contract: dict[str, Any]
    pending_human_approval: bool
    human_approval_request: dict[str, Any]


class GraphState(
    WorkerIdentityState,
    ObservationState,
    ActionExecutionState,
    RecipeReplayState,
    JobCollectionState,
    ActionSafetyState,
    total=False,
):
    """LangGraph 작업자 노드가 공유하는 상태 계약."""


__all__ = [
    "ActionExecutionState",
    "ActionSafetyState",
    "GraphState",
    "JobCollectionState",
    "ObservationState",
    "RecipeReplayState",
    "TransitionRequest",
    "TransitionResult",
    "WorkerIdentityState",
]

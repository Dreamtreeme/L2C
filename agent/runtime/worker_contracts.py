"""작업자 상태, 행동 요청과 실행 결과의 단일 도메인 계약."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

from agent.runtime.tool_schema import ACTION_TOOL_SCHEMAS
from shared.schema.agent_contract import DEFAULT_JOB_COLLECTION_FIELDS


def _message_text(content: Any) -> str:
    """모델 응답의 텍스트 블록을 짧은 행동 설명으로 정규화한다."""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("content") or ""
            else:
                text = str(item)
            if str(text).strip():
                parts.append(str(text).strip())
        return "\n".join(parts)
    return "" if content is None else str(content).strip()


class ToolCallRequest(BaseModel):
    """action executor에 전달할 단일 도구 호출."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "id")
    @classmethod
    def _require_non_empty_identifier(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("도구 이름과 호출 ID는 비어 있을 수 없습니다.")
        return normalized


class ActionRequest(BaseModel):
    """현재 화면 캡처를 근거로 실행할 행동 또는 검증된 행동 묶음."""

    source: str
    summary: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def _require_source(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("행동 요청 출처는 비어 있을 수 없습니다.")
        return normalized

    @model_validator(mode="after")
    def _validate_tool_contracts(self) -> "ActionRequest":
        for call in self.tool_calls:
            schema = ACTION_TOOL_SCHEMAS.get(call.name)
            if schema is None:
                raise ValueError(f"허용되지 않은 작업자 도구입니다: {call.name}")
            unknown = set(call.args) - set(schema.model_fields)
            if unknown:
                raise ValueError(
                    f"{call.name}에 정의되지 않은 인자가 있습니다: {sorted(unknown)}"
                )
            validated = schema.model_validate(call.args)
            normalized_args = validated.model_dump(exclude_none=True)
            for empty_collection_field in (
                "observed_fields",
                "unavailable_fields",
            ):
                if not normalized_args.get(empty_collection_field):
                    normalized_args.pop(empty_collection_field, None)
            call.args = normalized_args
        if len(self.tool_calls) > 1:
            from agent.runtime.replay_actions import (
                is_supported_recipe_action_group,
            )

            action_group = [
                {
                    "action": call.name,
                    "param": call.args,
                }
                for call in self.tool_calls
            ]
            if (
                self.source != "reflex"
                or self.metadata.get("execution_unit")
                != "recipe_transition"
                or not is_supported_recipe_action_group(action_group)
            ):
                raise ValueError(
                    "여러 행동은 검증된 경험 기반 탐색 전이에서만 허용됩니다."
                )
        return self


class ActionEvent(TypedDict, total=False):
    """행동 선택부터 화면 전환 검증까지 한 생명주기로 보관하는 기록."""

    seq: int
    result: dict[str, Any]
    recipe_step: dict[str, Any]
    feedback_episode: dict[str, Any]
    transition: dict[str, Any]


def build_action_event(
    seq: int,
    result: dict[str, Any],
    *,
    recipe_step: dict[str, Any] | None = None,
    feedback_episode: dict[str, Any] | None = None,
) -> ActionEvent:
    event: ActionEvent = {
        "seq": int(seq),
        "result": dict(result),
    }
    if recipe_step:
        event["recipe_step"] = dict(recipe_step)
    if feedback_episode:
        event["feedback_episode"] = dict(feedback_episode)
    return event


def action_event_results(events: Sequence[ActionEvent | dict]) -> list[dict[str, Any]]:
    return [
        dict(event.get("result") or {})
        for event in events or []
        if isinstance(event, Mapping) and isinstance(event.get("result"), Mapping)
    ]


def action_event_recipe_steps(events: Sequence[ActionEvent | dict]) -> list[dict[str, Any]]:
    return [
        dict(event.get("recipe_step") or {})
        for event in events or []
        if isinstance(event, Mapping) and isinstance(event.get("recipe_step"), Mapping)
    ]


def action_event_feedback(events: Sequence[ActionEvent | dict]) -> list[dict[str, Any]]:
    return [
        dict(event.get("feedback_episode") or {})
        for event in events or []
        if isinstance(event, Mapping)
        and isinstance(event.get("feedback_episode"), Mapping)
    ]


def action_event_transitions(events: Sequence[ActionEvent | dict]) -> list[dict[str, Any]]:
    return [
        dict(event.get("transition") or {})
        for event in events or []
        if isinstance(event, Mapping) and isinstance(event.get("transition"), Mapping)
    ]


def attach_action_transition(
    events: Sequence[ActionEvent | dict],
    transition: dict[str, Any],
) -> list[ActionEvent]:
    """행동 순번이 같은 이벤트에 화면 전환 검증 결과를 연결한다."""

    try:
        target_seq = int(transition.get("action_seq"))
    except (TypeError, ValueError):
        return [dict(event) for event in events or [] if isinstance(event, Mapping)]

    updated: list[ActionEvent] = []
    for raw_event in events or []:
        if not isinstance(raw_event, Mapping):
            continue
        event: ActionEvent = dict(raw_event)
        try:
            event_seq = int(event.get("seq"))
        except (TypeError, ValueError):
            event_seq = -1
        if event_seq == target_seq:
            event["transition"] = dict(transition)
        updated.append(event)
    return updated


def build_action_request(
    source: str,
    summary: str,
    tool_calls: list[dict[str, Any] | ToolCallRequest],
    *,
    metadata: dict[str, Any] | None = None,
    allowed_tool_names: Sequence[str] | None = None,
) -> ActionRequest:
    """정책 또는 모델 호출을 검증된 행동 요청으로 만든다."""

    calls: list[ToolCallRequest] = []
    for index, call in enumerate(tool_calls):
        if isinstance(call, ToolCallRequest):
            calls.append(call)
            continue
        payload = dict(call)
        if not str(payload.get("id") or "").strip():
            payload["id"] = f"{source}_{index}"
        calls.append(ToolCallRequest.model_validate(payload))
    if allowed_tool_names is not None:
        allowed = {str(name) for name in allowed_tool_names}
        rejected = sorted({call.name for call in calls if call.name not in allowed})
        if rejected:
            raise ValueError(f"현재 사이트에서 허용되지 않은 도구입니다: {rejected}")
    return ActionRequest(
        source=source,
        summary=summary,
        tool_calls=calls,
        metadata=dict(metadata or {}),
    )


def action_request_from_model_response(
    response: Any,
    *,
    allowed_tool_names: Sequence[str],
) -> ActionRequest:
    """LangChain 모델 응답을 추론 경계에서 한 번만 도메인 계약으로 바꾼다."""

    raw_calls = getattr(response, "tool_calls", None) or []
    return build_action_request(
        "llm",
        _message_text(getattr(response, "content", "")),
        list(raw_calls),
        allowed_tool_names=allowed_tool_names,
    )


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


class WorkerState(
    WorkerIdentityState,
    ObservationState,
    ActionExecutionState,
    RecipeReplayState,
    JobCollectionState,
    ActionSafetyState,
    total=False,
):
    """작업자 노드와 결정 로직이 공유하는 상태 계약."""


def create_worker_state(goal: str = "", **overrides: Any) -> WorkerState:
    """모든 작업자 진입점에서 동일한 초기 상태를 만든다."""

    state: WorkerState = {
        "goal": goal,
        "worker_run_id": "",
        "current_capture_id": "",
        "ocr_capture_id": "",
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
        "action_events": [],
        "marked_image": "",
        "screen_signature": {},
        "error_count": 0,
        "is_finished": False,
        "extracted_jd": {},
        "pending_action": None,
        "reflex_trace": {},
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
        "job_card_queue": [],
        "job_results_memory": {},
        "job_card_selection_trace": {},
        "job_results_availability": {},
        "job_detail_buffer": {},
        "job_detail_coverage": {},
        "job_detail_followup": {},
        "return_to_job_results": {},
        "action_permission_contract": {},
        "pending_human_approval": False,
        "human_approval_request": {},
    }
    state.update(overrides)
    return state


__all__ = [
    "ActionExecutionState",
    "ActionEvent",
    "ActionRequest",
    "ActionSafetyState",
    "WorkerState",
    "JobCollectionState",
    "ObservationState",
    "RecipeReplayState",
    "ToolCallRequest",
    "TransitionRequest",
    "TransitionResult",
    "WorkerIdentityState",
    "action_event_feedback",
    "action_event_recipe_steps",
    "action_event_results",
    "action_event_transitions",
    "action_request_from_model_response",
    "attach_action_transition",
    "build_action_event",
    "build_action_request",
    "create_worker_state",
]

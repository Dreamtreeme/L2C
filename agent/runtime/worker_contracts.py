"""작업자 상태, 행동 요청과 실행 결과의 단일 도메인 계약."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, TypedDict, cast

from pydantic import BaseModel, Field, field_validator, model_validator

from agent.runtime.tool_schema import ACTION_TOOL_SCHEMAS
from agent.runtime.worker_actions import is_supported_recipe_action_group
from shared.schema.agent_contract import DEFAULT_JOB_COLLECTION_FIELDS
from shared.schema.collection_intent import CollectionIntent
from shared.schema.feedback_schema import (
    FeedbackEpisode,
    RecordedRecipeStep,
    RecordedTransition,
)
from shared.schema.jd_schema import JobCapture


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
    observation_id: str = ""
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
            action_group = [
                {
                    "action": call.name,
                    "param": call.args,
                }
                for call in self.tool_calls
            ]
            if (
                self.source != "reflex"
                or self.metadata.get("execution_unit") != "recipe_transition"
                or not is_supported_recipe_action_group(action_group)
            ):
                raise ValueError(
                    "여러 행동은 검증된 경험 기반 탐색 전이에서만 허용됩니다."
                )
        return self


class ActionEvent(TypedDict, total=False):
    """행동 선택부터 화면 전환 검증까지 한 생명주기로 보관하는 기록."""

    seq: int
    observation_id: str
    result: dict[str, Any]
    recipe_step: RecordedRecipeStep
    feedback_episode: FeedbackEpisode
    transition: dict[str, Any]


def build_action_event(
    seq: int,
    result: dict[str, Any],
    *,
    observation_id: str = "",
    recipe_step: RecordedRecipeStep | None = None,
    feedback_episode: FeedbackEpisode | None = None,
) -> ActionEvent:
    event: ActionEvent = {
        "seq": int(seq),
        "result": dict(result),
    }
    if observation_id:
        event["observation_id"] = observation_id
    if recipe_step:
        event["recipe_step"] = recipe_step
    if feedback_episode:
        event["feedback_episode"] = feedback_episode
    return event


def action_event_results(events: Sequence[ActionEvent]) -> list[dict[str, Any]]:
    return [
        dict(event.get("result") or {})
        for event in events or []
        if event.get("result") is not None
    ]


def action_event_recipe_steps(
    events: Sequence[ActionEvent],
) -> list[RecordedRecipeStep]:
    return [
        event["recipe_step"]
        for event in events or []
        if event.get("recipe_step") is not None
    ]


def action_event_feedback(events: Sequence[ActionEvent]) -> list[FeedbackEpisode]:
    return [
        event["feedback_episode"]
        for event in events or []
        if event.get("feedback_episode") is not None
    ]


def action_event_transitions(
    events: Sequence[ActionEvent],
) -> list[dict[str, Any]]:
    return [
        dict(event["transition"])
        for event in events or []
        if isinstance(event.get("transition"), Mapping)
    ]


def attach_action_transition(
    events: Sequence[ActionEvent],
    transition: dict[str, Any],
) -> list[ActionEvent]:
    """행동 순번이 같은 이벤트에 화면 전환 검증 결과를 연결한다."""

    recorded_transition = RecordedTransition.model_validate(transition)
    target_seq = recorded_transition.action_seq
    if target_seq is None:
        return list(events or [])

    updated: list[ActionEvent] = []
    for raw_event in events or []:
        event: ActionEvent = dict(raw_event)
        if event["seq"] == target_seq:
            event["transition"] = recorded_transition.model_dump(mode="json")
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
    before_observation_id: str
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


class WorkerRequestState(TypedDict, total=False):
    """한 작업자 실행에서 변하지 않는 목표와 수집 계약."""

    worker_run_id: str
    goal: str
    collection_intent: CollectionIntent
    recipe_inputs: dict[str, Any]


class ObservationState(TypedDict, total=False):
    """한 캡처에서 얻은 화면, OCR과 브라우저 상태."""

    observation_id: str
    observation_sequence: int
    current_screenshot: str
    capture_quality: dict[str, Any]
    raw_screen_signature: dict[str, Any]
    analysis_mode: str
    ocr_complete: bool
    previous_observation: dict[str, Any]
    ui_context: str
    current_url: str
    current_page_role: str
    current_url_stale: bool
    low_information_screen: bool
    low_information_capture_count: int
    current_markers: list[dict[str, Any]]
    marked_image: str
    screen_signature: dict[str, Any]


class DecisionState(TypedDict, total=False):
    """현재 캡처에서 선택한 다음 행동과 선택 근거."""

    pending_action: ActionRequest | None
    job_card_selection_trace: dict[str, Any]


class TransitionState(TypedDict, total=False):
    """행동 실행 기록과 다음 화면 전환 판정 상태."""

    action_events: list[ActionEvent]
    error_count: int
    transition_request: TransitionRequest
    transition_result: TransitionResult


class RecipeReplayState(TypedDict, total=False):
    """자율탐색 기록과 경험 기반 탐색 재생 상태."""

    reflex_trace: dict[str, Any]
    active_reflex_recipe: dict[str, Any]
    reflex_blocked_recipe_keys: list[str]


class JobCollectionState(TypedDict, total=False):
    """공고 목록 선택, 상세 판독과 결과 누적 상태."""

    job_captures: list[JobCapture]
    job_card_queue: list[dict[str, Any]]
    job_results_memory: dict[str, Any]
    job_results_availability: dict[str, Any]
    job_detail_buffer: dict[str, Any]
    job_detail_coverage: dict[str, Any]
    job_detail_followup: dict[str, Any]


class ActionSafetyState(TypedDict, total=False):
    """작업 권한과 사용자 승인 대기 상태."""

    action_permission_contract: dict[str, Any]
    pending_human_approval: bool
    human_approval_request: dict[str, Any]


class WorkerLifecycleState(TypedDict, total=False):
    """작업자 반복 실행의 종료 상태."""

    is_finished: bool


def merge_worker_section(
    current: Mapping[str, Any] | None,
    update: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """노드가 소유한 섹션의 일부만 반환해도 기존 값을 보존한다."""

    return {**dict(current or {}), **dict(update or {})}


class WorkerState(TypedDict):
    """책임 경계별로 분리한 비전 작업자 중앙 상태."""

    request: Annotated[WorkerRequestState, merge_worker_section]
    observation: Annotated[ObservationState, merge_worker_section]
    decision: Annotated[DecisionState, merge_worker_section]
    transition: Annotated[TransitionState, merge_worker_section]
    replay: Annotated[RecipeReplayState, merge_worker_section]
    collection: Annotated[JobCollectionState, merge_worker_section]
    lifecycle: Annotated[WorkerLifecycleState, merge_worker_section]
    safety: Annotated[ActionSafetyState, merge_worker_section]


class WorkerStateUpdate(TypedDict, total=False):
    """노드가 변경할 책임 섹션만 담는 상태 패치."""

    request: WorkerRequestState
    observation: ObservationState
    decision: DecisionState
    transition: TransitionState
    replay: RecipeReplayState
    collection: JobCollectionState
    lifecycle: WorkerLifecycleState
    safety: ActionSafetyState


def apply_worker_state_update(
    state: WorkerState,
    update: WorkerStateUpdate,
) -> WorkerState:
    """노드 내부의 연속 계산에서 LangGraph와 같은 섹션 병합을 적용한다."""

    return WorkerState(
        request=cast(
            WorkerRequestState,
            merge_worker_section(state["request"], update.get("request")),
        ),
        observation=cast(
            ObservationState,
            merge_worker_section(
                state["observation"],
                update.get("observation"),
            ),
        ),
        decision=cast(
            DecisionState,
            merge_worker_section(state["decision"], update.get("decision")),
        ),
        transition=cast(
            TransitionState,
            merge_worker_section(
                state["transition"],
                update.get("transition"),
            ),
        ),
        replay=cast(
            RecipeReplayState,
            merge_worker_section(state["replay"], update.get("replay")),
        ),
        collection=cast(
            JobCollectionState,
            merge_worker_section(
                state["collection"],
                update.get("collection"),
            ),
        ),
        lifecycle=cast(
            WorkerLifecycleState,
            merge_worker_section(
                state["lifecycle"],
                update.get("lifecycle"),
            ),
        ),
        safety=cast(
            ActionSafetyState,
            merge_worker_section(state["safety"], update.get("safety")),
        ),
    )


def create_worker_state(
    goal: str = "",
    *,
    request: Mapping[str, Any] | None = None,
    observation: Mapping[str, Any] | None = None,
    decision: Mapping[str, Any] | None = None,
    transition: Mapping[str, Any] | None = None,
    replay: Mapping[str, Any] | None = None,
    collection: Mapping[str, Any] | None = None,
    lifecycle: Mapping[str, Any] | None = None,
    safety: Mapping[str, Any] | None = None,
) -> WorkerState:
    """모든 작업자 진입점에서 동일한 섹션 상태를 만든다."""

    state: WorkerState = {
        "request": {
            "goal": goal,
            "worker_run_id": "",
            "collection_intent": CollectionIntent(
                required_fields=list(DEFAULT_JOB_COLLECTION_FIELDS)
            ),
            "recipe_inputs": {},
        },
        "observation": {
            "observation_id": "",
            "observation_sequence": 0,
            "current_screenshot": "",
            "capture_quality": {},
            "raw_screen_signature": {},
            "analysis_mode": "",
            "ocr_complete": False,
            "previous_observation": {},
            "ui_context": "",
            "current_url": "",
            "current_page_role": "",
            "current_url_stale": True,
            "low_information_screen": False,
            "low_information_capture_count": 0,
            "current_markers": [],
            "marked_image": "",
            "screen_signature": {},
        },
        "decision": {
            "pending_action": None,
            "job_card_selection_trace": {},
        },
        "transition": {
            "action_events": [],
            "error_count": 0,
            "transition_request": {},
            "transition_result": {
                "status": "idle",
                "needs_ocr": False,
            },
        },
        "replay": {
            "reflex_trace": {},
            "active_reflex_recipe": {},
            "reflex_blocked_recipe_keys": [],
        },
        "collection": {
            "job_captures": [],
            "job_card_queue": [],
            "job_results_memory": {},
            "job_results_availability": {},
            "job_detail_buffer": {},
            "job_detail_coverage": {},
            "job_detail_followup": {},
        },
        "lifecycle": {"is_finished": False},
        "safety": {
            "action_permission_contract": {},
            "pending_human_approval": False,
            "human_approval_request": {},
        },
    }
    return apply_worker_state_update(
        state,
        {
            "request": cast(WorkerRequestState, dict(request or {})),
            "observation": cast(ObservationState, dict(observation or {})),
            "decision": cast(DecisionState, dict(decision or {})),
            "transition": cast(TransitionState, dict(transition or {})),
            "replay": cast(RecipeReplayState, dict(replay or {})),
            "collection": cast(JobCollectionState, dict(collection or {})),
            "lifecycle": cast(WorkerLifecycleState, dict(lifecycle or {})),
            "safety": cast(ActionSafetyState, dict(safety or {})),
        },
    )


__all__ = [
    "ActionEvent",
    "ActionRequest",
    "ActionSafetyState",
    "DecisionState",
    "WorkerState",
    "WorkerStateUpdate",
    "WorkerLifecycleState",
    "WorkerRequestState",
    "JobCollectionState",
    "ObservationState",
    "RecipeReplayState",
    "ToolCallRequest",
    "TransitionRequest",
    "TransitionResult",
    "TransitionState",
    "apply_worker_state_update",
    "action_event_feedback",
    "action_event_recipe_steps",
    "action_event_results",
    "action_event_transitions",
    "action_request_from_model_response",
    "attach_action_transition",
    "build_action_event",
    "build_action_request",
    "create_worker_state",
    "merge_worker_section",
]

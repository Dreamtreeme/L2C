"""작업자 상태, 행동 요청과 실행 결과의 단일 도메인 계약."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, NotRequired, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.runtime.tool_schema import (
    ACTION_TOOL_SCHEMAS,
    normalize_model_action_calls,
)
from agent.runtime.worker_actions import is_supported_recipe_tool_group
from shared.schema.agent_contract import DEFAULT_JOB_COLLECTION_FIELDS
from shared.schema.collection_intent import CollectionIntent
from shared.schema.jd_schema import CollectedJob, JobCapture, JobDraft, JobReview
from shared.schema.feedback_schema import ExecutionEvent
from shared.schema.execution_record_schema import (
    ObservedAction,
    ObservedTransition,
    ObservedTransitionEvidence,
    ScreenCheckpoint,
    TransitionStatus,
)
from shared.schema.experience_rule_schema import ExpectedEffect, ReplaySession


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
            call.args = normalized_args
        if len(self.tool_calls) > 1:
            action_names = [call.name for call in self.tool_calls]
            commit_key = str(self.tool_calls[-1].args.get("key") or "")
            if not is_supported_recipe_tool_group(
                action_names,
                commit_key=commit_key,
            ):
                raise ValueError("여러 행동은 입력 후 제출 조합만 허용됩니다.")
        return self


ActionEvent: TypeAlias = ExecutionEvent


class CompletedTransitionObservation(BaseModel):
    """화면 전환 판정 결과에서 경험 기록에 필요한 필드."""

    model_config = ConfigDict(extra="forbid")

    action_seq: int | None = None
    action_seqs: list[int] = Field(default_factory=list)
    action: str = ""
    before_observation_id: str = ""
    after_observation_id: str = ""
    expected_after: str = ""
    source: str = ""
    status: TransitionStatus = ""
    outcome: str = ""
    reason: str = ""
    recipe_key: str = ""
    recipe_step_index: int | None = None
    recipe_step_count: int | None = None
    transition_actions: list[str] = Field(default_factory=list)
    after_state_match: dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    elapsed_sec: float = 0.0
    phash_distance: int | None = None
    visual_change_ratio: float | None = None
    ocr_skipped: bool = False
    marker_count: int = 0
    marker_texts: list[str] = Field(default_factory=list)
    screenshot: str = ""
    marked_image: str = ""
    current_url: str = ""
    page_role: str = ""
    after_state: ScreenCheckpoint


def build_action_event(
    seq: int,
    result: dict[str, Any],
    *,
    observation_id: str = "",
    candidate_action: ObservedAction | None = None,
    before_checkpoint: ScreenCheckpoint | None = None,
    before_marker_texts: Sequence[str] | None = None,
    expected_after: str = "",
    intent: str = "",
) -> ActionEvent:
    return ExecutionEvent(
        seq=int(seq),
        observation_id=observation_id,
        result=dict(result),
        candidate_action=candidate_action,
        before_checkpoint=before_checkpoint,
        before_marker_texts=[str(item) for item in before_marker_texts or []],
        expected_after=expected_after,
        intent=intent,
    )


def _execution_event(event: ActionEvent | Mapping[str, Any]) -> ExecutionEvent:
    if isinstance(event, ExecutionEvent):
        return event
    return ExecutionEvent.model_validate(event)


def action_event_results(events: Sequence[ActionEvent]) -> list[dict[str, Any]]:
    return [dict(_execution_event(event).result) for event in events]


def action_event_transitions(
    events: Sequence[ActionEvent],
) -> list[ObservedTransition]:
    return [
        parsed.transition
        for event in events
        if (parsed := _execution_event(event)).transition is not None
    ]


def attach_action_transition(
    events: Sequence[ActionEvent],
    transition: Mapping[str, Any] | CompletedTransitionObservation,
) -> list[ActionEvent]:
    """행동 순번이 같은 이벤트를 완성된 경험 전이로 만든다."""

    observation = (
        transition
        if isinstance(transition, CompletedTransitionObservation)
        else CompletedTransitionObservation.model_validate(transition)
    )
    parsed_events = [_execution_event(event) for event in events]
    action_seqs = list(observation.action_seqs)
    if not action_seqs and observation.action_seq is not None:
        action_seqs = [observation.action_seq]
    if not action_seqs:
        return parsed_events

    selected_events = [event for event in parsed_events if event.seq in action_seqs]
    recorded_actions = [
        event.candidate_action.model_copy(deep=True)
        for event in selected_events
        if event.candidate_action is not None
    ]
    first_event = next(
        (
            event
            for event in selected_events
            if event.candidate_action is not None
            and event.before_checkpoint is not None
        ),
        None,
    )
    if (
        not recorded_actions
        or first_event is None
        or first_event.before_checkpoint is None
    ):
        return parsed_events

    result_statuses = [
        str(event.result.get("status") or "") for event in selected_events
    ]
    if result_statuses and all(status == "success" for status in result_statuses):
        result_status = "success"
    elif "error" in result_statuses:
        result_status = "error"
    elif "skipped" in result_statuses:
        result_status = "skipped"
    else:
        result_status = ""
    final_event = selected_events[-1]
    final_result = final_event.result
    evidence = ObservedTransitionEvidence(
        source=observation.source,
        result_status=result_status,
        result_reason=str(
            final_result.get("reason") or final_result.get("error") or ""
        ),
        status=observation.status,
        outcome=observation.outcome,
        reason=observation.reason,
        recipe_key=observation.recipe_key,
        recipe_step_index=observation.recipe_step_index,
        recipe_step_count=observation.recipe_step_count,
        transition_actions=observation.transition_actions,
        after_state_match=observation.after_state_match,
        attempt=observation.attempt,
        elapsed_sec=observation.elapsed_sec,
        phash_distance=observation.phash_distance,
        visual_change_ratio=observation.visual_change_ratio,
        ocr_skipped=observation.ocr_skipped,
        before_marker_texts=first_event.before_marker_texts,
        after_marker_texts=observation.marker_texts,
        screenshot=observation.screenshot,
        marked_image=observation.marked_image,
    )
    observed = ObservedTransition(
        seq=action_seqs[0],
        before=first_event.before_checkpoint,
        actions=recorded_actions,
        after=observation.after_state,
        expected_after=final_event.expected_after,
        intent=final_event.intent,
        evidence=evidence,
    )

    updated: list[ActionEvent] = []
    for raw_event in parsed_events:
        event = raw_event.model_copy(deep=True)
        if event.seq == observation.action_seq:
            event.transition = observed
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

    raw_calls = normalize_model_action_calls(
        list(getattr(response, "tool_calls", None) or [])
    )
    return build_action_request(
        "llm",
        _message_text(getattr(response, "content", "")),
        raw_calls,
        allowed_tool_names=allowed_tool_names,
    )


class ScreenMarker(TypedDict):
    """OCR과 아이콘 검출을 합친 화면 상호작용 요소."""

    id: int
    text: str
    bbox: list[int]
    type: str
    conf: NotRequired[float]


class CaptureContext(TypedDict):
    """화면 좌표를 다시 사용할 때 필요한 캡처 환경."""

    version: str
    size: list[int]
    content_top: int


class ScreenSignature(TypedDict, total=False):
    """캡처 단계에 따라 채워지는 화면 pHash와 OCR 요약."""

    algorithm: str
    phash: str
    size: list[int]
    marker_count: int
    marker_count_bucket: str
    anchors: list[str]
    capture_context: CaptureContext


class PreviousObservation(TypedDict, total=False):
    """뒤로가기와 OCR 재사용에 쓰는 직전 완료 관찰."""

    observation_id: str
    screenshot: str
    current_url: str
    markers: list[ScreenMarker]
    ui_context: str
    marked_image: str
    screen_signature: ScreenSignature
    page_role: str


class ReflexTrace(TypedDict, total=False):
    """경험 경로 후보 선택과 실패 원인을 남기는 실행 추적값."""

    hit: bool
    source: str
    reason: str
    site: str
    task_category: str
    candidate_count: int
    rejected_count: int
    last_reason: str
    reject_reasons: dict[str, int]
    candidate_rejections: list[dict[str, Any]]
    recipe_key: str
    source_node_id: str
    recipe_step_index: int
    recipe_step_count: int
    path_failed: bool
    actions: list[str]
    tool_calls: dict[str, dict[str, Any]]


class JobDetailLine(TypedDict):
    """상세 화면에서 중복 제거한 OCR 본문 한 줄."""

    line_id: int
    text: str
    bbox: list[int]
    bbox_ratio: list[float]
    marker_ids: list[int]
    first_screen: str


class JobDetailScreenEvidence(TypedDict):
    """상세 본문 줄이 추가된 화면 근거."""

    path: str
    added_lines: int
    duplicate_lines: int


class JobDetailStats(TypedDict):
    """상세 OCR 누적 현황."""

    screen_count: int
    added_lines_last_screen: int
    duplicate_lines_last_screen: int
    total_lines: int


class JobDetailBuffer(TypedDict, total=False):
    """공고 하나를 여러 화면에 걸쳐 읽는 동안 누적하는 OCR 본문."""

    url: str
    detail_key: str
    lines: list[JobDetailLine]
    seen_keys: list[str]
    screens: list[str]
    screen_evidence: list[JobDetailScreenEvidence]
    stats: JobDetailStats


class TransitionRequest(TypedDict):
    """화면 변경 행동 뒤 다음 캡처에서 확인할 전환 요청."""

    action_seq: int
    action_seqs: list[int]
    action: str
    before_observation_id: str
    source: str
    recipe_key: str
    transition_actions: list[str]
    expected_effect: ExpectedEffect | None
    expected_after: str
    input_text: str
    target_marker_id: int | None
    before_url: str
    before_page_role: str
    before_screenshot: str
    started_at: float
    recipe_step_index: NotRequired[int]
    recipe_step_count: NotRequired[int]
    source_reasoning_call_count: NotRequired[int]
    after_state_match: NotRequired[dict[str, Any]]
    execution_failed: NotRequired[bool]


class TransitionResult(TypedDict, total=False):
    """전환 요청과 현재 캡처를 비교한 판정 결과."""

    action_seq: int
    action_seqs: list[int]
    action: str
    before_observation_id: str
    source: str
    recipe_key: str
    transition_actions: list[str]
    expected_effect: ExpectedEffect | None
    expected_after: str
    input_text: str
    target_marker_id: int | None
    before_url: str
    before_page_role: str
    before_screenshot: str
    started_at: float
    recipe_step_index: int
    recipe_step_count: int
    source_reasoning_call_count: int
    after_state_match: dict[str, Any]
    execution_failed: bool
    status: str
    outcome: str
    reason: str
    visual_change_detected: bool
    visual_change_ratio: float | None
    needs_ocr: bool


class WorkerRequestState(TypedDict):
    """한 작업자 실행에서 변하지 않는 목표와 수집 계약."""

    worker_run_id: str
    goal: str
    collection_intent: CollectionIntent
    action_permission_contract: dict[str, Any]


class ObservationState(TypedDict):
    """한 캡처에서 얻은 화면, OCR과 브라우저 상태."""

    observation_id: str
    observation_sequence: int
    current_screenshot: str
    raw_screen_signature: ScreenSignature
    ocr_complete: bool
    previous_observation: PreviousObservation
    ui_context: str
    current_url: str
    current_page_role: str
    current_url_stale: bool
    low_information_screen: bool
    low_information_capture_count: int
    current_markers: list[ScreenMarker]
    marked_image: str
    screen_signature: ScreenSignature


WorkerStage: TypeAlias = Literal[
    "navigation",
    "results",
    "opening_detail",
    "detail",
    "finished",
]


class DecisionState(TypedDict):
    """현재 캡처에서 선택한 다음 행동과 선택 근거."""

    pending_action: ActionRequest | None
    reasoning_call_count: int
    reasoning_stage: WorkerStage
    reasoning_stage_call_count: int


class TransitionState(TypedDict):
    """행동 실행 기록과 다음 화면 전환 판정 상태."""

    action_events: list[ActionEvent]
    error_count: int
    transition_request: TransitionRequest | None
    transition_result: TransitionResult


class RecipeReplayState(TypedDict):
    """자율탐색 기록과 경험 기반 탐색 재생 상태."""

    reflex_trace: ReflexTrace
    replay_session: ReplaySession | None
    reflex_blocked_recipe_keys: list[str]


class JobCollectionState(TypedDict):
    """공고 목록 선택, 상세 판독과 결과 누적 상태."""

    job_captures: list[JobCapture]
    collected_jobs: list[CollectedJob]
    job_card_queue: list[dict[str, Any]]
    job_results_availability: dict[str, Any]
    job_detail_buffer: JobDetailBuffer
    pending_job_draft: JobDraft | None
    last_job_review: JobReview | None
    job_reviews: list[JobReview]


WorkerCompletionReason: TypeAlias = Literal[
    "",
    "target_reached",
    "visible_scope_completed",
    "scope_exhausted",
    "agent_finished",
    "screen_unavailable",
    "reasoning_limit",
]


class WorkerProgressState(TypedDict):
    """화면 역할과 분리해 관리하는 수집 업무의 진행 단계."""

    stage: WorkerStage


class WorkerLifecycleState(TypedDict):
    """작업자 반복 실행의 종료 상태."""

    is_finished: bool
    completion_reason: WorkerCompletionReason


class WorkerRequestPatch(TypedDict, total=False):
    worker_run_id: str
    goal: str
    collection_intent: CollectionIntent
    action_permission_contract: dict[str, Any]


class ObservationPatch(TypedDict, total=False):
    observation_id: str
    observation_sequence: int
    current_screenshot: str
    raw_screen_signature: ScreenSignature
    ocr_complete: bool
    previous_observation: PreviousObservation
    ui_context: str
    current_url: str
    current_page_role: str
    current_url_stale: bool
    low_information_screen: bool
    low_information_capture_count: int
    current_markers: list[ScreenMarker]
    marked_image: str
    screen_signature: ScreenSignature


class DecisionPatch(TypedDict, total=False):
    pending_action: ActionRequest | None
    reasoning_call_count: int
    reasoning_stage: WorkerStage
    reasoning_stage_call_count: int


class TransitionPatch(TypedDict, total=False):
    action_events: list[ActionEvent]
    error_count: int
    transition_request: TransitionRequest | None
    transition_result: TransitionResult


class RecipeReplayPatch(TypedDict, total=False):
    reflex_trace: ReflexTrace
    replay_session: ReplaySession | None
    reflex_blocked_recipe_keys: list[str]


class JobCollectionPatch(TypedDict, total=False):
    job_captures: list[JobCapture]
    collected_jobs: list[CollectedJob]
    job_card_queue: list[dict[str, Any]]
    job_results_availability: dict[str, Any]
    job_detail_buffer: JobDetailBuffer
    pending_job_draft: JobDraft | None
    last_job_review: JobReview | None
    job_reviews: list[JobReview]


class WorkerProgressPatch(TypedDict, total=False):
    stage: WorkerStage


class WorkerLifecyclePatch(TypedDict, total=False):
    is_finished: bool
    completion_reason: WorkerCompletionReason


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
    progress: Annotated[WorkerProgressState, merge_worker_section]
    lifecycle: Annotated[WorkerLifecycleState, merge_worker_section]


class WorkerStateUpdate(TypedDict, total=False):
    """노드가 변경할 책임 섹션만 담는 상태 패치."""

    request: WorkerRequestPatch
    observation: ObservationPatch
    decision: DecisionPatch
    transition: TransitionPatch
    replay: RecipeReplayPatch
    collection: JobCollectionPatch
    progress: WorkerProgressPatch
    lifecycle: WorkerLifecyclePatch


def apply_worker_state_update(
    state: WorkerState,
    update: WorkerStateUpdate,
) -> WorkerState:
    """노드 내부의 연속 계산에서 LangGraph와 같은 섹션 병합을 적용한다."""

    request = state["request"].copy()
    observation = state["observation"].copy()
    decision = state["decision"].copy()
    transition = state["transition"].copy()
    replay = state["replay"].copy()
    collection = state["collection"].copy()
    progress = state["progress"].copy()
    lifecycle = state["lifecycle"].copy()

    if request_update := update.get("request"):
        request.update(request_update)
    if observation_update := update.get("observation"):
        observation.update(observation_update)
    if decision_update := update.get("decision"):
        decision.update(decision_update)
    if transition_update := update.get("transition"):
        transition.update(transition_update)
    if replay_update := update.get("replay"):
        replay.update(replay_update)
    if collection_update := update.get("collection"):
        collection.update(collection_update)
    if progress_update := update.get("progress"):
        progress.update(progress_update)
    if lifecycle_update := update.get("lifecycle"):
        lifecycle.update(lifecycle_update)

    return {
        "request": request,
        "observation": observation,
        "decision": decision,
        "transition": transition,
        "replay": replay,
        "collection": collection,
        "progress": progress,
        "lifecycle": lifecycle,
    }


def create_worker_state(
    goal: str = "",
    *,
    request: WorkerRequestPatch | None = None,
    observation: ObservationPatch | None = None,
    decision: DecisionPatch | None = None,
    transition: TransitionPatch | None = None,
    replay: RecipeReplayPatch | None = None,
    collection: JobCollectionPatch | None = None,
    progress: WorkerProgressPatch | None = None,
    lifecycle: WorkerLifecyclePatch | None = None,
) -> WorkerState:
    """모든 작업자 진입점에서 동일한 섹션 상태를 만든다."""

    state: WorkerState = {
        "request": {
            "goal": goal,
            "worker_run_id": "",
            "collection_intent": CollectionIntent(
                required_fields=list(DEFAULT_JOB_COLLECTION_FIELDS)
            ),
            "action_permission_contract": {},
        },
        "observation": {
            "observation_id": "",
            "observation_sequence": 0,
            "current_screenshot": "",
            "raw_screen_signature": {},
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
            "reasoning_call_count": 0,
            "reasoning_stage": "navigation",
            "reasoning_stage_call_count": 0,
        },
        "transition": {
            "action_events": [],
            "error_count": 0,
            "transition_request": None,
            "transition_result": {
                "status": "idle",
                "needs_ocr": False,
            },
        },
        "replay": {
            "reflex_trace": {},
            "replay_session": None,
            "reflex_blocked_recipe_keys": [],
        },
        "collection": {
            "job_captures": [],
            "collected_jobs": [],
            "job_card_queue": [],
            "job_results_availability": {},
            "job_detail_buffer": {},
            "pending_job_draft": None,
            "last_job_review": None,
            "job_reviews": [],
        },
        "progress": {
            "stage": "navigation",
        },
        "lifecycle": {
            "is_finished": False,
            "completion_reason": "",
        },
    }
    update: WorkerStateUpdate = {}
    if request:
        update["request"] = request
    if observation:
        update["observation"] = observation
    if decision:
        update["decision"] = decision
    if transition:
        update["transition"] = transition
    if replay:
        update["replay"] = replay
    if collection:
        update["collection"] = collection
    if progress:
        update["progress"] = progress
    if lifecycle:
        update["lifecycle"] = lifecycle
    return apply_worker_state_update(state, update)


__all__ = [
    "ActionEvent",
    "ActionRequest",
    "CaptureContext",
    "DecisionState",
    "DecisionPatch",
    "JobDetailBuffer",
    "JobDetailLine",
    "JobDetailScreenEvidence",
    "JobDetailStats",
    "WorkerState",
    "WorkerStateUpdate",
    "WorkerStage",
    "WorkerCompletionReason",
    "WorkerProgressState",
    "WorkerProgressPatch",
    "WorkerLifecycleState",
    "WorkerLifecyclePatch",
    "WorkerRequestState",
    "WorkerRequestPatch",
    "JobCollectionState",
    "JobCollectionPatch",
    "ObservationState",
    "ObservationPatch",
    "PreviousObservation",
    "ReflexTrace",
    "RecipeReplayState",
    "RecipeReplayPatch",
    "ScreenMarker",
    "ScreenSignature",
    "ToolCallRequest",
    "TransitionRequest",
    "TransitionResult",
    "TransitionState",
    "TransitionPatch",
    "apply_worker_state_update",
    "action_event_results",
    "action_event_transitions",
    "action_request_from_model_response",
    "attach_action_transition",
    "build_action_event",
    "build_action_request",
    "create_worker_state",
    "merge_worker_section",
]

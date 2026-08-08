"""행동 요청 하나의 입력, 가변 결과와 최종 상태 패치를 관리한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from agent.graph.worker_execution_policy import state_snapshot_for_action
from agent.runtime.vision_worker_runtime import VisionWorkerRuntime
from agent.runtime.worker_data_services import WorkerDataServices
from agent.runtime.worker_contracts import (
    ActionEvent,
    ActionRequest,
    WorkerState,
    WorkerStateUpdate,
    apply_worker_state_update,
)
from agent.vision.target_snapshot import marker_by_id


WORKER_SECTIONS = (
    "request",
    "observation",
    "decision",
    "transition",
    "replay",
    "collection",
    "lifecycle",
    "safety",
)


@dataclass(frozen=True)
class ActionExecutionInput:
    """행동 실행 중 바뀌지 않는 그래프 입력과 런타임 의존성."""

    state: WorkerState
    action_request: ActionRequest
    worker_runtime: VisionWorkerRuntime
    data_services: WorkerDataServices


@dataclass
class ActionExecutionResult:
    """행동 실행이 만든 작업 상태와 관측 가능한 결과."""

    state: WorkerState
    prior_events: list[ActionEvent]
    new_actions: list[dict[str, Any]] = field(default_factory=list)
    new_events: list[ActionEvent] = field(default_factory=list)
    screen_changed: bool = False
    next_pending_action: ActionRequest | None = None


@dataclass
class WorkerExecutionContext:
    """행동 실행 입력과 아직 커밋하지 않은 결과를 보관한다."""

    input: ActionExecutionInput
    result: ActionExecutionResult

    @classmethod
    def from_state(
        cls,
        state: WorkerState,
        action_request: ActionRequest,
        worker_runtime: VisionWorkerRuntime,
        data_services: WorkerDataServices,
    ) -> "WorkerExecutionContext":
        """그래프 상태를 행동 실행 전용 문맥으로 변환한다."""

        return cls(
            input=ActionExecutionInput(
                state=state,
                action_request=action_request,
                worker_runtime=worker_runtime,
                data_services=data_services,
            ),
            result=ActionExecutionResult(
                state=apply_worker_state_update(state, {}),
                prior_events=[
                    dict(event)
                    for event in (
                        state["transition"].get("action_events", []) or []
                    )
                    if isinstance(event, dict)
                ],
            ),
        )

    def next_action_sequence(self) -> int:
        return len(self.result.prior_events) + len(self.result.new_events)

    def marker_bbox(self, marker_id: int) -> list[int]:
        markers = list(
            self.result.state["observation"].get("current_markers", []) or []
        )
        marker = marker_by_id(markers, marker_id)
        if marker:
            return marker["bbox"]
        raise ValueError(f"Marker ID {marker_id} not found in current screen.")

    def before_snapshot(self) -> dict[str, Any]:
        state = self.result.state
        current_url = str(state["observation"].get("current_url") or "")
        return state_snapshot_for_action(state, current_url)

    def build_state_update(self) -> WorkerStateUpdate:
        """실행 중 실제로 바뀐 작업자 섹션만 그래프에 반환한다."""

        state = self.result.state
        state["decision"]["pending_action"] = self.result.next_pending_action
        state["transition"]["action_events"] = [
            *self.result.prior_events,
            *self.result.new_events,
        ]
        transition_request = dict(
            state["transition"].get("transition_request", {}) or {}
        )
        state["transition"]["transition_result"] = {
            **transition_request,
            "status": "waiting_capture" if transition_request else "idle",
            "outcome": "",
            "reason": "",
            "visual_change_detected": False,
            "visual_change_ratio": None,
            "needs_ocr": False,
        }
        return cast(
            WorkerStateUpdate,
            {
                section: dict(state[section])
                for section in WORKER_SECTIONS
                if state[section] != self.input.state[section]
            },
        )


__all__ = [
    "ActionExecutionInput",
    "ActionExecutionResult",
    "WorkerExecutionContext",
]

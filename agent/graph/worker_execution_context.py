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


@dataclass
class WorkerExecutionContext:
    """행동 요청 하나의 원본 상태와 실행 중 변경을 보관한다."""

    original_state: WorkerState
    state: WorkerState
    action_request: ActionRequest
    worker_runtime: VisionWorkerRuntime
    data_services: WorkerDataServices
    prior_events: list[ActionEvent]
    new_actions: list[dict[str, Any]] = field(default_factory=list)
    new_events: list[ActionEvent] = field(default_factory=list)
    screen_changed: bool = False
    next_pending_action: ActionRequest | None = None

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
            original_state=state,
            state=apply_worker_state_update(state, {}),
            action_request=action_request,
            worker_runtime=worker_runtime,
            data_services=data_services,
            prior_events=[
                dict(event)
                for event in (state["transition"].get("action_events", []) or [])
                if isinstance(event, dict)
            ],
        )

    def next_action_sequence(self) -> int:
        return len(self.prior_events) + len(self.new_events)

    def marker_bbox(self, marker_id: int) -> list[int]:
        markers = list(self.state["observation"].get("current_markers", []) or [])
        marker = marker_by_id(markers, marker_id)
        if marker:
            return marker["bbox"]
        raise ValueError(f"Marker ID {marker_id} not found in current screen.")

    def before_snapshot(self) -> dict[str, Any]:
        state = self.state
        current_url = str(state["observation"].get("current_url") or "")
        return state_snapshot_for_action(state, current_url)

    def build_state_update(self) -> WorkerStateUpdate:
        """실행 중 실제로 바뀐 작업자 섹션만 그래프에 반환한다."""

        state = self.state
        state["decision"]["pending_action"] = self.next_pending_action
        state["transition"]["action_events"] = [
            *self.prior_events,
            *self.new_events,
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
                section: dict(value)
                for section, value in state.items()
                if value != self.original_state[section]
            },
        )


__all__ = [
    "WorkerExecutionContext",
]

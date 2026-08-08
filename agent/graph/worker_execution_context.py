"""행동 요청 하나를 실행하는 동안 변경되는 작업자 상태를 관리한다."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent.runtime.worker_contracts import (
    ActionEvent,
    ActionRequest,
    WorkerState,
    build_action_event,
)
from agent.graph.worker_execution_policy import (
    compact_action_args,
    state_snapshot_for_action,
)
from agent.utils.logger import logger
from agent.vision.target_snapshot import (
    build_action_target_snapshot,
    marker_by_id,
)


@dataclass
class WorkerExecutionContext:
    """행동 실행 중 필요한 변경 가능 상태를 한곳에 모은다."""

    state: WorkerState
    action_request: ActionRequest
    prior_events: list[ActionEvent]
    new_actions: list[dict[str, Any]] = field(default_factory=list)
    new_events: list[ActionEvent] = field(default_factory=list)
    current_jobs: dict[str, Any] = field(default_factory=dict)
    is_finished: bool = False
    error_count: int = 0
    current_url: str = ""
    current_url_stale: bool = True
    pending_human_approval: bool = False
    human_approval_request: dict[str, Any] = field(default_factory=dict)
    current_markers: list[dict[str, Any]] = field(default_factory=list)
    ui_context: str = ""
    marked_image: str = ""
    job_card_queue: list[dict[str, Any]] = field(default_factory=list)
    job_results_memory: dict[str, Any] = field(default_factory=dict)
    job_results_availability: dict[str, Any] = field(default_factory=dict)
    job_detail_buffer: dict[str, Any] = field(default_factory=dict)
    job_detail_coverage: dict[str, Any] = field(default_factory=dict)
    job_detail_followup: dict[str, Any] = field(default_factory=dict)
    return_to_job_results: dict[str, Any] = field(default_factory=dict)
    screen_changed: bool = False
    transition_request: dict[str, Any] = field(default_factory=dict)
    next_pending_action: ActionRequest | None = None

    @classmethod
    def from_state(
        cls,
        state: WorkerState,
        action_request: ActionRequest,
    ) -> "WorkerExecutionContext":
        """그래프 상태를 행동 실행 전용 문맥으로 변환한다."""

        return cls(
            state=state,
            action_request=action_request,
            prior_events=[
                dict(event)
                for event in (state.get("action_events", []) or [])
                if isinstance(event, dict)
            ],
            current_jobs=dict(state.get("extracted_jd", {}) or {}),
            is_finished=bool(state.get("is_finished", False)),
            error_count=int(state.get("error_count", 0) or 0),
            current_url=str(state.get("current_url") or ""),
            current_url_stale=bool(state.get("current_url_stale", True)),
            pending_human_approval=bool(
                state.get("pending_human_approval", False)
            ),
            human_approval_request=dict(
                state.get("human_approval_request", {}) or {}
            ),
            current_markers=[
                dict(item)
                for item in (state.get("current_markers", []) or [])
                if isinstance(item, dict)
            ],
            ui_context=str(state.get("ui_context") or ""),
            marked_image=str(state.get("marked_image") or ""),
            job_card_queue=[
                dict(item)
                for item in (state.get("job_card_queue", []) or [])
                if isinstance(item, dict)
            ],
            job_results_memory=dict(
                state.get("job_results_memory", {}) or {}
            ),
            job_results_availability=dict(
                state.get("job_results_availability", {}) or {}
            ),
            job_detail_buffer=dict(
                state.get("job_detail_buffer", {}) or {}
            ),
            job_detail_coverage=dict(
                state.get("job_detail_coverage", {}) or {}
            ),
            job_detail_followup=dict(
                state.get("job_detail_followup", {}) or {}
            ),
            return_to_job_results=dict(
                state.get("return_to_job_results", {}) or {}
            ),
        )

    def state_for_dispatch(self) -> WorkerState:
        """현재까지 반영된 값으로 상태 행동용 읽기 전용 상태를 만든다."""

        return {
            **self.state,
            "extracted_jd": self.current_jobs,
            "current_url": self.current_url,
            "job_detail_buffer": self.job_detail_buffer,
            "job_detail_coverage": self.job_detail_coverage,
            "job_detail_followup": self.job_detail_followup,
        }

    def observe_job_detail_fields(
        self,
        action_name: str,
        args: dict[str, Any],
    ) -> None:
        """기존 추론 호출이 판독한 상세 필드 근거를 상태에 누적한다."""

        if action_name not in {
            "click_marker",
            "scroll",
            "finish_detail_reading",
        }:
            return
        from agent.runtime.worker_state import job_detail_key_from_state
        from agent.runtime.detail_runtime import is_job_detail_context
        from agent.runtime.job_field_contract import (
            merge_job_detail_coverage,
        )

        page_role = str(
            args.get("page_role")
            or self.state.get("current_page_role")
            or ""
        )
        if not is_job_detail_context(
            self.current_url,
            page_role=page_role,
        ):
            return
        dispatch_state = self.state_for_dispatch()
        self.job_detail_coverage = merge_job_detail_coverage(
            self.job_detail_coverage,
            args,
            state=dispatch_state,
            current_url=self.current_url,
            detail_key=job_detail_key_from_state(dispatch_state),
        )

    def next_action_sequence(self) -> int:
        return len(self.prior_events) + len(self.new_events)

    def marker_bbox(self, marker_id: int) -> list[int]:
        marker = marker_by_id(self.current_markers, marker_id)
        if marker:
            return marker["bbox"]
        raise ValueError(
            f"Marker ID {marker_id} not found in current screen."
        )

    def before_snapshot(self) -> dict[str, Any]:
        return state_snapshot_for_action(self.state, self.current_url)

    def transition_step(
        self,
        action_sequence: int,
        action_name: str,
        args: dict[str, Any],
        tool_call_id: str = "",
    ) -> dict[str, Any]:
        """전환 기록에서 행동과 선택 근거를 함께 볼 수 있게 만든다."""

        step: dict[str, Any] = {
            "seq": action_sequence,
            "action": action_name,
            "decision_capture_id": str(
                self.action_request.metadata.get("decision_capture_id") or ""
            ),
            "args": compact_action_args(action_name, args),
            "page_role": (
                args.get("page_role")
                or self.state.get("current_page_role", "")
            ),
            "target_role": (
                args.get("target_role")
                or args.get("target_role_candidate")
                or ""
            ),
            "component": (
                args.get("target_component")
                or args.get("component_candidate")
                or ""
            ),
            "expected_after": args.get("expected_after") or "",
        }
        if tool_call_id:
            step["tool_call_id"] = tool_call_id
        if self.action_request.source == "reflex":
            trace = dict(self.state.get("reflex_trace", {}) or {})
            call_trace = (
                (trace.get("tool_calls") or {}).get(tool_call_id)
                if tool_call_id
                else None
            )
            if isinstance(call_trace, dict):
                step.update(
                    {
                        "recipe_key": trace.get("recipe_key", ""),
                        "recipe_seq": call_trace.get("seq"),
                        "replay_mode": call_trace.get("replay_mode", ""),
                        "match_mode": call_trace.get("match_mode", ""),
                        "target_text": call_trace.get("target_text", ""),
                        "marker_id": call_trace.get("marker_id"),
                        "phash": call_trace.get("phash", {}),
                    }
                )
        return {
            key: value
            for key, value in step.items()
            if value not in (None, "", {}, [])
        }

    def set_transition_request(
        self,
        action_sequence: int,
        action_name: str,
        args: dict[str, Any],
        source: str,
        tool_call_id: str = "",
    ) -> None:
        recipe_key = ""
        if source == "reflex":
            recipe_key = str(
                (self.state.get("reflex_trace", {}) or {}).get("recipe_key")
                or ""
            )
        request_metadata = dict(self.action_request.metadata or {})
        before_state = (
            dict(request_metadata.get("before_state") or {})
            if isinstance(request_metadata.get("before_state"), dict)
            else {}
        )
        self.transition_request = {
            "action_seq": action_sequence,
            "action": action_name,
            "from_capture_id": str(
                self.action_request.metadata.get("decision_capture_id")
                or self.state.get("current_capture_id")
                or ""
            ),
            "source": source,
            "recipe_key": recipe_key,
            "recipe_transition_index": request_metadata.get(
                "transition_index"
            ),
            "recipe_transition_count": request_metadata.get(
                "transition_count"
            ),
            "expected_after_state": dict(
                request_metadata.get("expected_after_state") or {}
            ),
            "before_page_role": str(before_state.get("page_role") or ""),
            "transition_actions": list(
                request_metadata.get("transition_actions") or []
            ),
            "step": self.transition_step(
                action_sequence,
                action_name,
                args,
                tool_call_id,
            ),
            "before_url": self.current_url
            or self.state.get("current_url", "")
            or "",
            "before_screenshot": (
                str(self.state.get("current_screenshot") or "")
            ),
            "started_at": time.time(),
        }

    def enrich_result(
        self,
        result: dict[str, Any],
        action_name: str,
        args: dict[str, Any],
        before_snapshot: dict[str, Any],
        *,
        screen_change_expected: bool = False,
        tool_call_id: str = "",
        tool_call_metadata: dict[str, Any] | None = None,
        action_source: str = "",
    ) -> dict[str, Any]:
        result["args"] = compact_action_args(action_name, args)
        result["action_source"] = (
            action_source or self.action_request.source
        )
        if tool_call_id:
            result["tool_call_id"] = tool_call_id
        if tool_call_metadata:
            result["execution_metadata"] = dict(tool_call_metadata)
        result["before_url"] = before_snapshot.get("url", "")
        result["before_screenshot"] = before_snapshot.get("screenshot", "")
        result["before_marked_image"] = before_snapshot.get(
            "marked_image",
            "",
        )
        result["decision_capture_id"] = str(
            self.action_request.metadata.get("decision_capture_id")
            or before_snapshot.get("capture_id")
            or ""
        )
        result["screen_change_expected"] = screen_change_expected
        target = build_action_target_snapshot(
            self.state,
            action_name,
            args,
        )
        if target:
            result["target"] = target
        if self.action_request.source == "reflex":
            trace = dict(self.state.get("reflex_trace", {}) or {})
            if trace:
                result["reflex_recipe_key"] = trace.get("recipe_key", "")
                call_trace = (
                    (trace.get("tool_calls") or {}).get(tool_call_id)
                    if tool_call_id
                    else None
                )
                if call_trace:
                    result["reflex_match"] = dict(call_trace)
        if result.get("action") != action_name:
            result["requested_action"] = action_name
        return result

    def append_action_event(
        self,
        action_name: str,
        args: dict[str, Any],
        enriched_result: dict[str, Any],
        before_snapshot: dict[str, Any],
        after_context: dict[str, Any],
        action_sequence: int,
        *,
        record_ui: bool = False,
    ) -> None:
        """실행 결과와 학습 증거를 같은 행동 이벤트에 기록한다."""

        from agent.recipe.feedback import record_action_episode
        from agent.recipe.record import record_ui_step

        record_state = {
            "goal": self.state.get("goal", ""),
            "current_capture_id": str(
                before_snapshot.get("capture_id") or ""
            ),
            "current_markers": list(
                self.state.get("current_markers", []) or []
            ),
            "current_url": before_snapshot.get("url", ""),
            "current_page_role": self.state.get("current_page_role", ""),
            "screen_signature": dict(
                self.state.get("screen_signature", {}) or {}
            ),
            "current_screenshot": str(
                self.state.get("current_screenshot") or ""
            ),
            "marked_image": self.state.get("marked_image", ""),
        }
        recipe_steps: list[dict[str, Any]] = []
        if record_ui:
            record_ui_step(
                recipe_steps,
                record_state,
                action_name,
                args,
                action_sequence,
            )
        feedback: list[dict[str, Any]] = []
        record_action_episode(
            feedback,
            record_state,
            self.action_request,
            action_name,
            args,
            enriched_result,
            before_snapshot,
            after_context,
            action_sequence,
        )
        self.new_events.append(
            build_action_event(
                action_sequence,
                enriched_result,
                recipe_step=recipe_steps[0] if recipe_steps else None,
                feedback_episode=feedback[0] if feedback else None,
            )
        )

    def append_guard_result(
        self,
        action_name: str,
        args: dict[str, Any],
        before_snapshot: dict[str, Any],
        *,
        status: str,
        reason: str,
        message: str,
        step_started: float,
        increments_error: bool = False,
        observation_required: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        if observation_required:
            self.current_url_stale = True
            self.screen_changed = True
        result: dict[str, Any] = {
            "status": status,
            "action": action_name,
            "result": message if status != "error" else None,
            "error": message if status == "error" else None,
            "reason": reason,
        }
        if observation_required:
            result["observation_required"] = True
        if details:
            result["guard"] = dict(details)

        action_sequence = self.next_action_sequence()
        if observation_required:
            self.set_transition_request(
                action_sequence,
                action_name,
                args,
                "guard",
            )
        enriched = self.enrich_result(
            result,
            action_name,
            args,
            before_snapshot,
        )
        self.new_actions.append(enriched)
        self.append_action_event(
            action_name,
            args,
            enriched,
            before_snapshot,
            self.after_context(screen_changed=observation_required),
            action_sequence,
        )
        if increments_error:
            self.error_count += 1
        logger.warning(message, action=action_name, reason=reason)
        logger.debug(
            "Action guard completed",
            duration_sec=round(time.perf_counter() - step_started, 6),
        )

    def require_human_approval(
        self,
        action_name: str,
        args: dict[str, Any],
        reason: str,
        before_snapshot: dict[str, Any],
        step_started: float,
    ) -> None:
        self.pending_human_approval = True
        self.human_approval_request = {
            "status": "needs_human_approval",
            "reason": reason,
            "action": action_name,
            "args": compact_action_args(action_name, args),
            "current_url": self.current_url,
            "message": (
                "Autonomous execution stopped before a sensitive or "
                "irreversible step."
            ),
        }
        self.append_guard_result(
            action_name,
            args,
            before_snapshot,
            status="skipped",
            reason=reason,
            message="Skipped sensitive action; human confirmation is required.",
            step_started=step_started,
        )

    def after_context(
        self,
        *,
        screen_changed: bool,
    ) -> dict[str, Any]:
        return {
            "current_url": self.current_url,
            "current_url_stale": self.current_url_stale,
            "screen_changed": screen_changed,
            "extracted_jd": self.current_jobs,
            "is_finished": self.is_finished,
        }

    def build_state_update(self) -> dict[str, Any]:
        return {
            "pending_action": self.next_pending_action,
            "action_events": [*self.prior_events, *self.new_events],
            "extracted_jd": self.current_jobs,
            "is_finished": self.is_finished,
            "error_count": self.error_count,
            "current_url": self.current_url,
            "current_url_stale": self.current_url_stale,
            "current_markers": self.current_markers,
            "ui_context": self.ui_context,
            "marked_image": self.marked_image,
            "screen_signature": dict(
                self.state.get("screen_signature", {}) or {}
            ),
            "transition_request": self.transition_request,
            "transition_result": {
                **self.transition_request,
                "status": (
                    "waiting_capture"
                    if self.transition_request
                    else "idle"
                ),
                "outcome": "",
                "reason": "",
                "visual_change_detected": False,
                "visual_change_ratio": None,
                "needs_ocr": False,
            },
            "job_card_queue": self.job_card_queue,
            "job_results_memory": self.job_results_memory,
            "job_results_availability": self.job_results_availability,
            "job_detail_buffer": self.job_detail_buffer,
            "job_detail_coverage": self.job_detail_coverage,
            "job_detail_followup": self.job_detail_followup,
            "return_to_job_results": self.return_to_job_results,
            "pending_human_approval": self.pending_human_approval,
            "human_approval_request": self.human_approval_request,
        }


__all__ = ["WorkerExecutionContext"]

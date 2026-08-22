"""경험 규칙 조회, 현재 화면 적용 판단과 물리 행동 바인딩."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from agent.runtime.site_context import normalize_page_role
from agent.runtime.target_matching import roi_signature_match
from agent.runtime.transition_runtime import used_idempotent_recipe_keys_on_url
from agent.runtime.worker_contracts import ScreenMarker, WorkerState
from agent.runtime.worker_data_services import ExperienceRuleLoader
from agent.runtime.worker_state import current_frame_signature
from agent.utils.text import recipe_url_scope_matches, url_template
from agent.vision.marker_geometry import (
    marker_bbox,
    marker_center,
    ratio_rect_to_pixels,
)
from shared.schema.experience_rule_schema import (
    ExperienceRule,
    ExperienceRuleStep,
    InteractionRegionHandle,
    ReplaySession,
    ResolvedRuleAction,
    ResolvedRuleStep,
    RuleAction,
)
from shared.schema.execution_record_schema import ActionTarget
from shared.schema.skill_schema import RecipeInputName


_REJECT_REASON_PRIORITY = {
    "url_scope_mismatch": 100,
    "page_role_mismatch": 95,
    "target_unresolved": 80,
}


@dataclass
class ReflexRejectionLog:
    """후보별 탈락 사유와 대표 실패 원인을 모은다."""

    rejected_count: int = 0
    last_reason: str = ""
    reason_score: int = -1
    reason_counts: dict[str, int] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def reject(self, rule_key: str, reason: str, **trace: Any) -> None:
        resolved = str(reason or "candidate_invalid")
        self.reason_counts[resolved] = self.reason_counts.get(resolved, 0) + 1
        score = _REJECT_REASON_PRIORITY.get(resolved, 50)
        if score > self.reason_score:
            self.reason_score = score
            self.last_reason = resolved
        self.candidates.append({"recipe_key": rule_key, "reason": resolved, **trace})

    def reject_candidate(self, rule_key: str, reason: str, **trace: Any) -> None:
        self.rejected_count += 1
        self.reject(rule_key, reason, **trace)

    def trace_payload(self, candidate_count: int) -> dict[str, Any]:
        return {
            "candidate_count": candidate_count,
            "rejected_count": self.rejected_count,
            "last_reason": self.last_reason,
            "reject_reasons": dict(self.reason_counts),
            "candidate_rejections": self.candidates[:12],
        }


@dataclass(frozen=True)
class ReplayInputs:
    """경험 규칙 입력 슬롯에 연결할 현재 요청 값."""

    search_keyword: str = ""

    def value(self, name: RecipeInputName | str) -> str:
        return self.search_keyword if name == "search_keyword" else ""


@dataclass(frozen=True)
class ReflexReplayContext:
    """한 캡처에서 경험 규칙 조회와 적용에 공통으로 쓰는 값."""

    markers: list[ScreenMarker]
    inputs: ReplayInputs
    task_category: str
    site: str
    current_image_path: str
    current_page_role: str
    current_url: str
    current_url_template: str
    observation_id: str
    screen_size: list[int]
    blocked_rule_keys: set[str]
    used_rule_keys: set[str]
    replay_session: ReplaySession | None
    rule_candidates: list[tuple[str, ExperienceRule]]

    @property
    def candidate_count(self) -> int:
        return len(self.rule_candidates)

    @property
    def active_rule_key(self) -> str:
        return self.replay_session.recipe_key if self.replay_session else ""


@dataclass(frozen=True)
class ReflexCandidate:
    rule_key: str
    rule: ExperienceRule
    step: ExperienceRuleStep
    step_index: int


@dataclass(frozen=True)
class ReflexSelection:
    rule_key: str
    rule: ExperienceRule
    step: ExperienceRuleStep
    step_index: int
    resolved_step: ResolvedRuleStep
    tool_calls: list[dict[str, Any]]
    tool_call_traces: dict[str, dict[str, Any]]
    markers: list[ScreenMarker]
    resolution_mode: str


@dataclass
class TargetBindings:
    """저장된 ROI와 좌표로 복원한 규칙 행동별 물리 대상."""

    markers: list[ScreenMarker]
    by_action: dict[int, ScreenMarker] = field(default_factory=dict)
    traces: dict[int, dict[str, Any]] = field(default_factory=dict)
    blocked_reason: str = ""
    verified_target_count: int = 0


def _marker_from_reference(
    reference: ActionTarget,
    roi_signature: dict[str, Any],
    screen_size: list[int],
    marker_id: int,
) -> ScreenMarker | None:
    """저장 당시 비율 좌표를 현재 캡처의 물리 좌표로 복원한다."""

    if len(screen_size) != 2 or min(screen_size) <= 0:
        return None
    bbox = ratio_rect_to_pixels(reference.bbox_ratio, screen_size)
    if bbox == [0, 0, 0, 0]:
        center = list(
            reference.center_ratio or roi_signature.get("target_center_ratio") or []
        )
        if len(center) != 2:
            return None
        width, height = screen_size
        try:
            x = round(float(center[0]) * width)
            y = round(float(center[1]) * height)
        except (TypeError, ValueError):
            return None
        radius = max(4, round(min(width, height) * 0.01))
        bbox = [x - radius, y - radius, x + radius, y + radius]
    return {
        "id": marker_id,
        "bbox": bbox,
        "text": reference.text,
        "type": reference.marker_type or "interaction_point",
    }


def load_reflex_replay_context(
    state: WorkerState,
    load_experience_rules: ExperienceRuleLoader,
) -> ReflexReplayContext:
    observation = state["observation"]
    intent = state["request"]["collection_intent"]
    replay = state["replay"]
    raw_session = replay.get("replay_session")
    replay_session = (
        raw_session
        if isinstance(raw_session, ReplaySession)
        else ReplaySession.model_validate(raw_session)
        if raw_session
        else None
    )
    site = intent.site.strip()
    rules = (
        load_experience_rules(
            site,
            task_category=intent.task_category.strip() or None,
        )
        if site
        else []
    )
    if replay_session:
        rules = [item for item in rules if item[0] == replay_session.recipe_key]
    current_url = str(observation.get("current_url") or "")
    frame_signature = current_frame_signature(state)
    return ReflexReplayContext(
        markers=list(observation.get("current_markers", []) or []),
        inputs=ReplayInputs(search_keyword=intent.search_keyword.strip()),
        task_category=intent.task_category.strip(),
        site=site,
        current_image_path=str(observation.get("current_screenshot") or ""),
        current_page_role=normalize_page_role(observation.get("current_page_role", "")),
        current_url=current_url,
        current_url_template=url_template(current_url),
        observation_id=str(observation.get("observation_id") or ""),
        screen_size=list(frame_signature.get("size") or []),
        blocked_rule_keys={
            str(key)
            for key in (replay.get("reflex_blocked_recipe_keys") or [])
            if str(key)
        },
        used_rule_keys=used_idempotent_recipe_keys_on_url(state, current_url),
        replay_session=replay_session,
        rule_candidates=rules,
    )


def _missing_inputs(rule: ExperienceRule, inputs: ReplayInputs) -> list[str]:
    return [
        item.name for item in rule.skill_metadata.inputs if not inputs.value(item.name)
    ]


def _eligible_candidates(
    context: ReflexReplayContext,
    rejection_log: ReflexRejectionLog,
) -> list[ReflexCandidate]:
    candidates: list[ReflexCandidate] = []
    for rule_key, rule in context.rule_candidates:
        if rule_key in context.blocked_rule_keys or (
            not context.active_rule_key and rule_key in context.used_rule_keys
        ):
            reason = (
                "rule_blocked_after_effect_failure"
                if rule_key in context.blocked_rule_keys
                else "rule_already_used_on_page"
            )
            rejection_log.reject_candidate(rule_key, reason)
            continue
        if _missing_inputs(rule, context.inputs):
            rejection_log.reject_candidate(rule_key, "missing_required_inputs")
            continue
        step_index = (
            context.replay_session.current_step_index
            if context.replay_session and rule_key == context.active_rule_key
            else 0
        )
        if step_index >= len(rule.steps):
            rejection_log.reject_candidate(rule_key, "rule_step_out_of_range")
            continue
        candidates.append(
            ReflexCandidate(
                rule_key=rule_key,
                rule=rule,
                step=rule.steps[step_index],
                step_index=step_index,
            )
        )
    return candidates


def _screen_precondition_reason(
    context: ReflexReplayContext,
    step: ExperienceRuleStep,
) -> str:
    if not recipe_url_scope_matches(step.before.url_template, context.current_url):
        return "url_scope_mismatch"
    expected_role = normalize_page_role(step.before.page_role)
    if (
        expected_role
        and context.current_page_role
        and context.current_page_role != expected_role
    ):
        return "page_role_mismatch"
    return ""


def _marker_handle(
    marker: ScreenMarker,
    screen_size: list[int],
    effect_region_ratio: list[float],
) -> InteractionRegionHandle:
    if len(screen_size) != 2 or min(screen_size) <= 0:
        raise ValueError("current capture size is required to resolve a rule target")
    width, height = screen_size
    left, top, right, bottom = marker_bbox(marker)
    center_x, center_y = marker_center(marker)
    return InteractionRegionHandle(
        marker_id=int(marker["id"]),
        center_ratio=[center_x / width, center_y / height],
        bbox_ratio=[left / width, top / height, right / width, bottom / height],
        effect_region_ratio=list(effect_region_ratio),
    )


def _saved_target(
    state: WorkerState,
    context: ReflexReplayContext,
    action: RuleAction,
    marker_id: int,
) -> tuple[ScreenMarker | None, dict[str, Any]]:
    target = action.target
    if target is None or target.reference is None:
        return None, {"reason": "target_reference_missing"}
    signature = dict(target.reference_roi_signature)
    phash = roi_signature_match(
        signature,
        context.current_image_path,
        current_signature=current_frame_signature(state),
    )
    trace: dict[str, Any] = {"phash": phash}
    if not phash.get("matched"):
        trace["reason"] = str(phash.get("reason") or "roi_phash_mismatch")
        return None, trace

    marker = _marker_from_reference(
        target.reference,
        signature,
        context.screen_size,
        marker_id,
    )
    trace["match_mode"] = "saved_target_coordinate" if marker else "none"
    if marker is None:
        trace["reason"] = "target_unresolved"
    return marker, trace


def _resolved_actions(
    candidate: ReflexCandidate,
    context: ReflexReplayContext,
    markers_by_action: dict[int, ScreenMarker],
) -> list[ResolvedRuleAction]:
    actions: list[ResolvedRuleAction] = []
    for action in candidate.step.actions:
        marker = markers_by_action.get(action.source_seq)
        handle = (
            _marker_handle(
                marker,
                context.screen_size,
                candidate.step.expected_effect.target_region_ratio,
            )
            if marker is not None
            else None
        )
        param = action.param.model_copy(deep=True)
        if action.input_slot:
            param = param.model_copy(
                update={
                    "text": context.inputs.value(action.input_slot),
                    "slot_name": action.input_slot,
                }
            )
        actions.append(
            ResolvedRuleAction(
                source_seq=action.source_seq,
                action=action.action,
                target=handle,
                param=param,
                risk_level=action.risk_level,
            )
        )
    return actions


def _tool_args(
    action: ResolvedRuleAction,
    step: ExperienceRuleStep,
) -> dict[str, Any] | None:
    marker_id = action.target.marker_id if action.target else None
    trace_args = {
        "reason": step.intent,
        "page_role": step.before.page_role,
        "expected_after": step.expected_effect.description,
    }
    if action.action == "click_marker":
        if marker_id is None:
            return None
        return {"marker_id": marker_id, **trace_args, "risk_level": action.risk_level}
    if action.action == "type_in_marker":
        if marker_id is None or not action.param.text:
            return None
        args: dict[str, Any] = {
            "marker_id": marker_id,
            "text": action.param.text,
            **trace_args,
            "risk_level": action.risk_level,
        }
        if action.param.slot_name:
            args["slot_name"] = action.param.slot_name
        return args
    if action.action == "scroll":
        args = {
            "direction": action.param.direction or "down",
            "amount": action.param.amount or "page",
            **trace_args,
            "risk_level": action.risk_level,
        }
        if marker_id is not None:
            args["marker_id"] = marker_id
        return args
    if action.action == "press_key" and action.param.key:
        return {
            "key": action.param.key,
            **trace_args,
            "risk_level": action.risk_level or "safe_navigation",
        }
    return None


def _selection(
    candidate: ReflexCandidate,
    context: ReflexReplayContext,
    markers: list[ScreenMarker],
    markers_by_action: dict[int, ScreenMarker],
    traces: dict[int, dict[str, Any]],
    *,
    resolution_mode: str,
) -> ReflexSelection | None:
    resolved_step = ResolvedRuleStep(
        recipe_key=candidate.rule_key,
        step_index=candidate.step_index,
        observation_id=context.observation_id,
        actions=_resolved_actions(candidate, context, markers_by_action),
        expected_effect=candidate.step.expected_effect,
    )
    tool_calls: list[dict[str, Any]] = []
    tool_call_traces: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha1(candidate.rule_key.encode("utf-8")).hexdigest()[:12]
    for action_index, action in enumerate(resolved_step.actions):
        args = _tool_args(action, candidate.step)
        if args is None:
            return None
        call_id = f"reflex_{digest}_{candidate.step_index}_{action_index}"
        tool_calls.append({"name": action.action, "args": args, "id": call_id})
        tool_call_traces[call_id] = {
            "source_action_seq": action.source_seq,
            "action": action.action,
            "resolution_mode": resolution_mode,
            **traces.get(action.source_seq, {}),
        }
    return ReflexSelection(
        rule_key=candidate.rule_key,
        rule=candidate.rule,
        step=candidate.step,
        step_index=candidate.step_index,
        resolved_step=resolved_step,
        tool_calls=tool_calls,
        tool_call_traces=tool_call_traces,
        markers=markers,
        resolution_mode=resolution_mode,
    )


def _bind_saved_targets(
    state: WorkerState,
    context: ReflexReplayContext,
    candidate: ReflexCandidate,
) -> TargetBindings:
    bindings = TargetBindings(markers=[marker.copy() for marker in context.markers])
    next_marker_id = (
        max(
            (int(marker["id"]) for marker in bindings.markers),
            default=-1,
        )
        + 1
    )
    for action in candidate.step.actions:
        if action.target is None:
            continue
        marker, trace = _saved_target(
            state,
            context,
            action,
            next_marker_id,
        )
        bindings.traces[action.source_seq] = trace
        if marker is None:
            bindings.blocked_reason = bindings.blocked_reason or str(
                trace.get("reason") or "target_unresolved"
            )
            continue
        bindings.markers.append(marker)
        bindings.by_action[action.source_seq] = marker
        bindings.verified_target_count += 1
        next_marker_id += 1
    return bindings


def _bind_candidate(
    state: WorkerState,
    context: ReflexReplayContext,
    candidate: ReflexCandidate,
    rejection_log: ReflexRejectionLog,
) -> ReflexSelection | None:
    precondition_reason = _screen_precondition_reason(context, candidate.step)
    if precondition_reason:
        rejection_log.reject(candidate.rule_key, precondition_reason)
        return None

    bindings = _bind_saved_targets(state, context, candidate)
    if bindings.blocked_reason:
        rejection_log.reject(
            candidate.rule_key,
            bindings.blocked_reason,
            target_traces=bindings.traces,
        )
        return None
    if not bindings.verified_target_count:
        rejection_log.reject(
            candidate.rule_key,
            "target_unresolved",
            target_traces=bindings.traces,
        )
        return None
    return _selection(
        candidate,
        context,
        bindings.markers,
        bindings.by_action,
        bindings.traces,
        resolution_mode="saved_coordinate",
    )


def select_reflex_replay(
    state: WorkerState,
    context: ReflexReplayContext,
) -> tuple[ReflexSelection | None, ReflexRejectionLog]:
    """후보를 검증하고 첫 번째로 현재 화면에 적용 가능한 규칙을 반환한다."""

    rejection_log = ReflexRejectionLog()
    for candidate in _eligible_candidates(context, rejection_log):
        selection = _bind_candidate(
            state,
            context,
            candidate,
            rejection_log,
        )
        if selection is not None:
            return selection, rejection_log
        rejection_log.rejected_count += 1
    return None, rejection_log


__all__ = [
    "ReflexRejectionLog",
    "ReflexReplayContext",
    "ReflexSelection",
    "ReplayInputs",
    "load_reflex_replay_context",
    "select_reflex_replay",
]

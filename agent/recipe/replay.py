"""경험 기반 탐색 후보 조회, 화면 검증과 행동 바인딩."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from agent.recipe.matcher import is_replayable_action
from agent.recipe.phash_replay import match_step_by_screen_signature
from agent.runtime.site_context import normalize_page_role
from agent.runtime.transition_runtime import used_idempotent_recipe_keys_on_url
from agent.runtime.worker_actions import (
    RECIPE_COMMIT_ACTIONS,
    TARGET_REPLAY_ACTIONS,
)
from agent.runtime.worker_contracts import WorkerState
from agent.runtime.worker_data_services import SiteExperienceLoader
from agent.utils.text import recipe_url_scope_matches, url_template
from shared.schema.recipe_schema import (
    ExperienceTransition,
    PhysicalAction,
    ReplaySession,
    ScreenCheckpoint,
    SiteExperience,
)
from shared.schema.skill_schema import RecipeInputName


_REJECT_REASON_PRIORITY = {
    "capture_size_mismatch": 100,
    "roi_phash_distance": 90,
    "target_ratio_miss": 75,
    "url_scope_mismatch": 65,
}


@dataclass
class ReflexRejectionLog:
    """후보별 탈락 사유와 대표 실패 원인을 모은다."""

    rejected_count: int = 0
    last_reason: str = ""
    reason_score: int = -1
    reason_counts: dict[str, int] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def reject(
        self,
        recipe_key: str,
        reason: str,
        trace: dict[str, Any] | None = None,
    ) -> None:
        resolved = str(reason or "candidate_invalid")
        self.reason_counts[resolved] = self.reason_counts.get(resolved, 0) + 1
        score = _REJECT_REASON_PRIORITY.get(resolved, 50)
        if score > self.reason_score:
            self.reason_score = score
            self.last_reason = resolved
        item = {"recipe_key": recipe_key, "reason": resolved}
        if trace:
            item.update(
                {
                    "page_role": trace.get("page_role", ""),
                    "current_page_role": trace.get("current_page_role", ""),
                    "url_template": trace.get("url_template", ""),
                    "current_url_template": trace.get(
                        "current_url_template", ""
                    ),
                    "action": trace.get("action", ""),
                    "phash": dict(trace.get("phash") or {}),
                }
            )
        self.candidates.append(item)

    def reject_candidate(
        self,
        recipe_key: str,
        reason: str,
        trace: dict[str, Any] | None = None,
    ) -> None:
        self.rejected_count += 1
        self.reject(recipe_key, reason, trace)

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
    """경험 경로의 가변 슬롯에 바인딩할 현재 요청 값."""

    search_keyword: str = ""

    def value(self, name: RecipeInputName) -> str:
        if name == "search_keyword":
            return self.search_keyword
        return ""


@dataclass(frozen=True)
class ReflexReplayContext:
    """한 캡처에서 후보 조회와 검증에 공통으로 쓰는 값."""

    markers: list[dict[str, Any]]
    inputs: ReplayInputs
    task_category: str
    site: str
    current_image_path: str
    current_page_role: str
    current_url: str
    current_url_template: str
    blocked_recipe_keys: set[str]
    used_recipe_keys: set[str]
    replay_session: ReplaySession | None
    recipe_candidates: list[tuple[str, SiteExperience]]

    @property
    def candidate_count(self) -> int:
        return len(self.recipe_candidates)

    @property
    def active_recipe_key(self) -> str:
        return self.replay_session.recipe_key if self.replay_session else ""


@dataclass(frozen=True)
class ReflexCandidate:
    recipe_key: str
    recipe: SiteExperience
    transition: ExperienceTransition
    transition_index: int


@dataclass(frozen=True)
class ReflexSelection:
    recipe_key: str
    recipe: SiteExperience
    transition: ExperienceTransition
    transition_index: int
    tool_calls: list[dict[str, Any]]
    tool_call_traces: dict[str, dict[str, Any]]


def _trace_args(
    action: PhysicalAction,
    before: ScreenCheckpoint,
    transition: ExperienceTransition,
) -> dict[str, str]:
    """실행에 영향 없는 경험 추적 메타데이터를 도구 인자로 복원한다."""

    out: dict[str, str] = {}
    values = {
        "reason": action.intent or transition.intent,
        "page_role": before.page_role,
        "target_role": action.target_role,
        "target_component": action.component,
        "expected_after": transition.expected_after,
        "target_label": (
            action.target.semantic_label
            if action.target and action.target.semantic_label
            else ""
        ),
    }
    for key, value in values.items():
        if value:
            out[key] = value
    return out


def _click_action_args(
    action: PhysicalAction,
    marker_id: int | None,
    trace_args: dict[str, str],
) -> dict[str, Any] | None:
    if marker_id is None:
        return None
    args: dict[str, Any] = {"marker_id": marker_id, **trace_args}
    if action.risk_level:
        args["risk_level"] = action.risk_level
    return args


def _type_action_args(
    action: PhysicalAction,
    marker_id: int | None,
    inputs: ReplayInputs,
    trace_args: dict[str, str],
) -> dict[str, Any] | None:
    if marker_id is None:
        return None
    slot_name = action.parameter_slot()
    if action.replay_mode == "parameterized" and not slot_name:
        return None
    bound_text = inputs.value(slot_name) if slot_name else ""
    if action.replay_mode == "parameterized" and not bound_text:
        return None
    text = bound_text or action.param.text
    if not text:
        return None
    args: dict[str, Any] = {
        "marker_id": marker_id,
        "text": text,
        **trace_args,
    }
    if slot_name:
        args["slot_name"] = slot_name
    if action.risk_level:
        args["risk_level"] = action.risk_level
    return args


def _commit_action_args(
    action: PhysicalAction,
    trace_args: dict[str, str],
) -> dict[str, Any] | None:
    key = action.param.key.strip()
    if key.casefold() not in {"enter", "return"}:
        return None
    args: dict[str, Any] = {"key": key}
    args.update(
        {
            name: value
            for name, value in trace_args.items()
            if name in {"reason", "page_role", "expected_after"}
        }
    )
    args["risk_level"] = action.risk_level or "safe_navigation"
    return args


def build_reflex_action_args(
    action: PhysicalAction,
    before: ScreenCheckpoint,
    transition: ExperienceTransition,
    marker_id: int | None,
    inputs: ReplayInputs,
) -> dict[str, Any] | None:
    """저장된 행동을 검증된 물리 도구 인자로 바꾼다."""

    trace_args = _trace_args(action, before, transition)
    if action.action == "click_marker":
        return _click_action_args(action, marker_id, trace_args)
    if action.action == "type_in_marker":
        return _type_action_args(action, marker_id, inputs, trace_args)
    if action.action not in RECIPE_COMMIT_ACTIONS:
        return None
    return _commit_action_args(action, trace_args)


def _missing_required_inputs(
    recipe: SiteExperience,
    inputs: ReplayInputs,
) -> list[str]:
    return [
        item.name
        for item in recipe.skill_metadata.inputs
        if not inputs.value(item.name)
    ]


def load_reflex_replay_context(
    state: WorkerState,
    load_site_recipes: SiteExperienceLoader,
) -> ReflexReplayContext:
    """현재 작업 상태에 맞는 활성 경험 경로 후보를 조회한다."""

    observation = state["observation"]
    request = state["request"]
    replay = state["replay"]
    intent = request["collection_intent"]
    site = intent.site.strip()
    raw_session = replay.get("replay_session")
    replay_session = (
        raw_session
        if isinstance(raw_session, ReplaySession)
        else ReplaySession.model_validate(raw_session)
        if raw_session
        else None
    )
    active_recipe_key = replay_session.recipe_key if replay_session else ""
    recipe_candidates = (
        load_site_recipes(
            site,
            task_category=intent.task_category.strip() or None,
        )
        if site
        else []
    )
    if active_recipe_key:
        recipe_candidates = [
            candidate
            for candidate in recipe_candidates
            if candidate[0] == active_recipe_key
        ]
    current_url = str(observation.get("current_url") or "")
    return ReflexReplayContext(
        markers=list(observation.get("current_markers", []) or []),
        inputs=ReplayInputs(search_keyword=intent.search_keyword.strip()),
        task_category=intent.task_category.strip(),
        site=site,
        current_image_path=str(observation.get("current_screenshot") or ""),
        current_page_role=normalize_page_role(
            observation.get("current_page_role", "")
        ),
        current_url=current_url,
        current_url_template=url_template(current_url),
        blocked_recipe_keys={
            str(key)
            for key in (replay.get("reflex_blocked_recipe_keys") or [])
            if str(key)
        },
        used_recipe_keys=used_idempotent_recipe_keys_on_url(state, current_url),
        replay_session=replay_session,
        recipe_candidates=recipe_candidates,
    )


def _eligible_candidates(
    context: ReflexReplayContext,
    rejection_log: ReflexRejectionLog,
) -> list[ReflexCandidate]:
    candidates: list[ReflexCandidate] = []
    for recipe_key, recipe in context.recipe_candidates:
        if recipe_key in context.blocked_recipe_keys or (
            not context.active_recipe_key and recipe_key in context.used_recipe_keys
        ):
            reason = (
                "recipe_blocked_after_transition_failure"
                if recipe_key in context.blocked_recipe_keys
                else "recipe_already_used_on_page"
            )
            rejection_log.reject_candidate(recipe_key, reason)
            continue
        if _missing_required_inputs(recipe, context.inputs):
            rejection_log.reject_candidate(recipe_key, "missing_required_inputs")
            continue

        transition_count = len(recipe.transitions)
        transition_index = (
            context.replay_session.current_transition_index
            if context.replay_session
            and recipe_key == context.active_recipe_key
            else 0
        )
        if transition_index >= transition_count:
            rejection_log.reject_candidate(
                recipe_key,
                "recipe_transition_out_of_range",
            )
            continue
        candidates.append(
            ReflexCandidate(
                recipe_key=recipe_key,
                recipe=recipe,
                transition=recipe.transitions[transition_index],
                transition_index=transition_index,
            )
        )
    return candidates


def _step_trace(
    action: PhysicalAction,
    before: ScreenCheckpoint,
    *,
    transition_index: int,
    action_index: int,
    context: ReflexReplayContext,
) -> dict[str, Any]:
    return {
        "seq": action.source_seq,
        "transition_index": transition_index,
        "action_index": action_index,
        "action": action.action,
        "page_role": before.page_role,
        "current_page_role": context.current_page_role,
        "url_template": before.url_template,
        "current_url_template": context.current_url_template,
        "replay_mode": action.replay_mode,
        "match_mode": "none",
        "target_text": action.target.text if action.target else "",
    }


def _match_target_action_screen(
    state: WorkerState,
    context: ReflexReplayContext,
    action: PhysicalAction,
    trace: dict[str, Any],
) -> tuple[int | None, str]:
    marker_id, phash_result = match_step_by_screen_signature(
        action,
        dict(state["observation"].get("screen_signature", {}) or {}),
        context.markers,
        current_image_path=context.current_image_path,
    )
    trace["phash"] = phash_result
    trace["match_mode"] = phash_result.get("mode") or "roi_phash"
    if marker_id is None:
        return None, str(phash_result.get("reason") or "phash_check_failed")
    trace["marker_id"] = marker_id
    return marker_id, ""


def _match_action_screen(
    state: WorkerState,
    context: ReflexReplayContext,
    action: PhysicalAction,
    before: ScreenCheckpoint,
    trace: dict[str, Any],
    *,
    action_index: int,
) -> tuple[int | None, str]:
    """저장된 행동의 화면 문맥과 현재 마커 대상을 검증한다."""

    if not is_replayable_action(action, before):
        return None, "not_replayable"
    if not recipe_url_scope_matches(before.url_template, context.current_url):
        return None, "url_scope_mismatch"
    if action.action in TARGET_REPLAY_ACTIONS:
        return _match_target_action_screen(state, context, action, trace)
    if action.action in RECIPE_COMMIT_ACTIONS and action_index > 0:
        trace["match_mode"] = "grouped_recipe_action"
        return None, ""
    return None, "commit_action_without_target_input"


def _bind_candidate(
    state: WorkerState,
    context: ReflexReplayContext,
    candidate: ReflexCandidate,
    rejection_log: ReflexRejectionLog,
) -> ReflexSelection | None:
    transition = candidate.transition
    tool_calls: list[dict[str, Any]] = []
    tool_call_traces: dict[str, dict[str, Any]] = {}

    for action_index, action in enumerate(transition.actions):
        trace = _step_trace(
            action,
            transition.before,
            transition_index=candidate.transition_index,
            action_index=action_index,
            context=context,
        )
        marker_id, reject_reason = _match_action_screen(
            state,
            context,
            action,
            transition.before,
            trace,
            action_index=action_index,
        )
        if reject_reason:
            rejection_log.reject(candidate.recipe_key, reject_reason, trace)
            return None

        args = build_reflex_action_args(
            action,
            transition.before,
            transition,
            marker_id,
            context.inputs,
        )
        if args is None:
            rejection_log.reject(
                candidate.recipe_key,
                "args_build_failed",
                trace,
            )
            return None
        key_digest = hashlib.sha1(
            candidate.recipe_key.encode("utf-8")
        ).hexdigest()[:12]
        call_id = (
            f"reflex_{key_digest}_{candidate.transition_index}_{action_index}"
        )
        tool_calls.append({"name": action.action, "args": args, "id": call_id})
        trace.update({"accepted": True, "tool_call_id": call_id})
        tool_call_traces[call_id] = dict(trace)

    if not tool_calls:
        rejection_log.reject(candidate.recipe_key, "candidate_invalid")
        return None
    return ReflexSelection(
        recipe_key=candidate.recipe_key,
        recipe=candidate.recipe,
        transition=transition,
        transition_index=candidate.transition_index,
        tool_calls=tool_calls,
        tool_call_traces=tool_call_traces,
    )


def select_reflex_replay(
    state: WorkerState,
    context: ReflexReplayContext,
) -> tuple[ReflexSelection | None, ReflexRejectionLog]:
    """후보를 순서대로 검증하고 첫 번째 실행 가능한 전이를 반환한다."""

    rejection_log = ReflexRejectionLog()
    for candidate in _eligible_candidates(context, rejection_log):
        selection = _bind_candidate(state, context, candidate, rejection_log)
        if selection is not None:
            return selection, rejection_log
        rejection_log.rejected_count += 1
    return None, rejection_log


__all__ = [
    "ReflexRejectionLog",
    "ReflexReplayContext",
    "ReflexSelection",
    "ReplayInputs",
    "build_reflex_action_args",
    "load_reflex_replay_context",
    "select_reflex_replay",
]

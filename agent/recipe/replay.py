"""경험 기반 탐색 후보 조회, 화면 검증과 행동 바인딩."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.recipe.matcher import is_replayable_action
from agent.recipe.phash_replay import (
    match_step_by_screen_signature,
    screen_context_signature_match,
)
from agent.runtime.action_validation import text_input_target_rejection
from agent.runtime.worker_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
)
from agent.runtime.site_context import normalize_page_role
from agent.runtime.transition_runtime import used_idempotent_recipe_keys_on_url
from agent.runtime.worker_contracts import WorkerState
from agent.runtime.worker_data_services import SiteRecipeLoader
from agent.utils.text import recipe_url_scope_matches, url_template
from shared.schema.recipe_schema import RecipeTransition, SiteRecipe

_REJECT_REASON_PRIORITY = {
    "capture_size_mismatch": 100,
    "screen_context_signature_missing": 100,
    "roi_phash_distance": 90,
    "screen_context_phash_distance": 90,
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
                    "current_url_template": trace.get("current_url_template", ""),
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
class ReflexReplayContext:
    """한 캡처에서 후보 조회와 검증에 공통으로 쓰는 값."""

    markers: list[dict[str, Any]]
    params: dict[str, Any]
    task_category: str
    site: str
    current_image_path: str
    current_page_role: str
    current_url: str
    current_url_template: str
    blocked_recipe_keys: set[str]
    used_recipe_keys: set[str]
    active_recipe: dict[str, Any]
    active_recipe_key: str
    recipe_candidates: list[tuple[str, SiteRecipe]]

    @property
    def candidate_count(self) -> int:
        return len(self.recipe_candidates)


@dataclass(frozen=True)
class ReflexCandidate:
    recipe_key: str
    recipe: SiteRecipe
    transition: RecipeTransition
    transition_index: int


@dataclass(frozen=True)
class ReflexSelection:
    recipe_key: str
    recipe: SiteRecipe
    transition: RecipeTransition
    transition_index: int
    tool_calls: list[dict[str, Any]]
    tool_call_traces: dict[str, dict[str, Any]]


def _trace_args(step: dict[str, Any]) -> dict[str, str]:
    """실행에 영향 없는 레시피 추적 메타데이터를 도구 인자로 복원한다."""

    out: dict[str, str] = {}
    mapping = {
        "intent": "reason",
        "page_role": "page_role",
        "target_role": "target_role",
        "component": "target_component",
        "expected_after": "expected_after",
    }
    for source_key, arg_key in mapping.items():
        value = step.get(source_key)
        if value:
            out[arg_key] = str(value)
    target = step.get("target") if isinstance(step.get("target"), dict) else {}
    semantic_label = target.get("semantic_label")
    if semantic_label:
        out["target_label"] = str(semantic_label)
    return out


def _click_action_args(
    step: dict[str, Any],
    marker_id: int | None,
    trace_args: dict[str, str],
) -> dict[str, Any] | None:
    if marker_id is None:
        return None
    args: dict[str, Any] = {
        "marker_id": marker_id,
        **trace_args,
        "needs_user_confirmation": bool(step.get("needs_user_confirmation")),
    }
    if step.get("risk_level"):
        args["risk_level"] = step.get("risk_level")
    return args


def _parameter_slot_name(step: dict[str, Any], param: dict[str, Any]) -> str:
    slot_name = str(param.get("slot_name") or param.get("slot") or "")
    if slot_name:
        return slot_name
    slot_refs = step.get("slot_refs") or []
    return str(slot_refs[0]) if slot_refs else ""


def _type_action_args(
    step: dict[str, Any],
    marker_id: int | None,
    params: dict[str, Any],
    trace_args: dict[str, str],
) -> dict[str, Any] | None:
    if marker_id is None:
        return None
    param = dict(step.get("param") or {})
    slot_name = _parameter_slot_name(step, param)
    parameterized = step.get("replay_mode") == "parameterized"
    if parameterized and not slot_name:
        return None
    bound_text = params.get(slot_name) if slot_name else None
    if parameterized and slot_name and not bound_text:
        return None
    text = bound_text or param.get("text") or step.get("value")
    if not text:
        return None
    args: dict[str, Any] = {
        "marker_id": marker_id,
        "text": text,
        **trace_args,
    }
    if slot_name:
        args["slot_name"] = slot_name
    if step.get("risk_level"):
        args["risk_level"] = step.get("risk_level")
    args["needs_user_confirmation"] = bool(step.get("needs_user_confirmation"))
    return args


def _contextual_action_args(
    step: dict[str, Any],
    trace_args: dict[str, str],
) -> dict[str, Any] | None:
    action = str(step.get("action") or "")
    param = dict(step.get("param") or {})
    allowed_param_keys = {
        "press_key": {"key"},
        "go_back": set(),
        "close_current_tab": set(),
        "switch_tab": {"direction"},
    }[action]
    args = {
        key: item
        for key, item in param.items()
        if key in allowed_param_keys and item not in (None, "")
    }
    if action == "press_key" and not args.get("key"):
        return None
    if action == "switch_tab" and not args.get("direction"):
        return None
    args.update(
        {
            key: item
            for key, item in trace_args.items()
            if key in {"reason", "page_role", "expected_after"}
        }
    )
    args["risk_level"] = step.get("risk_level") or "safe_navigation"
    args["needs_user_confirmation"] = bool(step.get("needs_user_confirmation"))
    return args


def build_reflex_action_args(
    step: dict[str, Any],
    marker_id: int | None,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """저장된 행동을 검증된 물리 도구 인자로 바꾼다."""

    action = step.get("action")
    trace_args = _trace_args(step)
    if action == "click_marker":
        return _click_action_args(step, marker_id, trace_args)
    if action == "type_in_marker":
        return _type_action_args(step, marker_id, params, trace_args)
    if action not in CONTEXTUAL_REPLAY_ACTIONS:
        return None
    return _contextual_action_args(step, trace_args)


def _missing_required_inputs(
    recipe: SiteRecipe,
    params: dict[str, Any],
) -> list[str]:
    missing: list[str] = []
    for item in recipe.skill_metadata.inputs:
        value = params.get(item.name)
        if item.required and item.name and (value is None or value == ""):
            missing.append(item.name)
    return missing


def load_reflex_replay_context(
    state: WorkerState,
    load_site_recipes: SiteRecipeLoader,
) -> ReflexReplayContext:
    """현재 작업 상태에 맞는 활성 레시피 후보를 조회한다."""

    observation = state["observation"]
    request = state["request"]
    replay = state["replay"]
    intent = request["collection_intent"]
    markers = list(observation.get("current_markers", []) or [])
    params = dict(request.get("recipe_inputs") or {})
    params.setdefault("goal", request.get("goal", ""))
    task_category = intent.task_category.strip()
    site = intent.site.strip()
    current_image_path = str(observation.get("current_screenshot") or "")
    current_page_role = normalize_page_role(observation.get("current_page_role", ""))
    current_url = str(observation.get("current_url") or "")
    active_recipe = dict(replay.get("active_reflex_recipe", {}) or {})
    active_recipe_key = str(active_recipe.get("recipe_key") or "")
    recipe_candidates = (
        load_site_recipes(
            site,
            task_category=task_category or None,
        )
        if site
        else []
    )
    if active_recipe_key:
        recipe_candidates = [
            candidate
            for candidate in recipe_candidates
            if str(candidate[0]) == active_recipe_key
        ]
    return ReflexReplayContext(
        markers=markers,
        params=params,
        task_category=task_category,
        site=site,
        current_image_path=current_image_path,
        current_page_role=current_page_role,
        current_url=current_url,
        current_url_template=url_template(current_url),
        blocked_recipe_keys={
            str(key)
            for key in (replay.get("reflex_blocked_recipe_keys") or [])
            if str(key)
        },
        used_recipe_keys=used_idempotent_recipe_keys_on_url(state, current_url),
        active_recipe=active_recipe,
        active_recipe_key=active_recipe_key,
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
        if _missing_required_inputs(recipe, context.params):
            rejection_log.reject_candidate(recipe_key, "missing_required_inputs")
            continue
        if not recipe.transitions:
            rejection_log.reject_candidate(recipe_key, "empty_recipe")
            continue

        transition_count = len(recipe.transitions)
        transition_index = (
            int(context.active_recipe.get("current_transition_index") or 0)
            if recipe_key == context.active_recipe_key
            else 0
        )
        if transition_index < 0 or transition_index >= transition_count:
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
    step: dict[str, Any],
    *,
    transition_index: int,
    action_index: int,
    context: ReflexReplayContext,
) -> dict[str, Any]:
    target = step.get("target") if isinstance(step.get("target"), dict) else {}
    return {
        "seq": step.get("source_seq"),
        "transition_index": transition_index,
        "action_index": action_index,
        "action": step.get("action"),
        "page_role": step.get("page_role", ""),
        "current_page_role": context.current_page_role,
        "url_template": step.get("url_template", ""),
        "current_url_template": context.current_url_template,
        "replay_mode": step.get("replay_mode"),
        "match_mode": "none",
        "target_text": target.get("text", ""),
    }


def _match_contextual_action_screen(
    observation: dict[str, Any],
    context: ReflexReplayContext,
    step: dict[str, Any],
    trace: dict[str, Any],
    action_index: int,
) -> str:
    action = step.get("action")
    if action_index != 0 or action not in CONTEXTUAL_REPLAY_ACTIONS:
        return ""
    context_match = screen_context_signature_match(
        dict(step.get("screen_context_signature") or {}),
        dict(observation.get("screen_signature") or {}),
    )
    trace["phash"] = context_match
    trace["match_mode"] = context_match.get("mode") or "screen_context_phash"
    if not context_match.get("matched"):
        return str(context_match.get("reason") or "screen_context_mismatch")
    if not context.active_recipe_key:
        return "recipe_must_start_with_roi"
    return ""


def _match_target_action_screen(
    observation: dict[str, Any],
    context: ReflexReplayContext,
    step: dict[str, Any],
    trace: dict[str, Any],
) -> tuple[int | None, str]:
    marker_id, phash_result = match_step_by_screen_signature(
        step,
        dict(observation.get("screen_signature", {}) or {}),
        context.markers,
        current_image_path=context.current_image_path,
    )
    trace["phash"] = phash_result
    trace["match_mode"] = phash_result.get("mode") or "roi_phash"
    if marker_id is None:
        return None, str(phash_result.get("reason") or "phash_check_failed")
    if step.get("action") == "type_in_marker":
        target_rejection = text_input_target_rejection(
            context.markers,
            marker_id,
        )
        if target_rejection:
            return None, str(
                target_rejection.get("reason") or "invalid_text_input_target"
            )
    trace["marker_id"] = marker_id
    return marker_id, ""


def _match_action_screen(
    state: WorkerState,
    context: ReflexReplayContext,
    step: dict[str, Any],
    trace: dict[str, Any],
    *,
    action_index: int,
) -> tuple[int | None, str]:
    """저장된 행동의 화면 문맥과 현재 마커 대상을 검증한다."""

    observation = state["observation"]
    action = step.get("action")
    if not is_replayable_action(step):
        return None, "not_replayable"
    if not recipe_url_scope_matches(step.get("url_template", ""), context.current_url):
        return None, "url_scope_mismatch"

    contextual_rejection = _match_contextual_action_screen(
        observation,
        context,
        step,
        trace,
        action_index,
    )
    if contextual_rejection:
        return None, contextual_rejection
    if action in TARGET_REPLAY_ACTIONS:
        return _match_target_action_screen(
            observation,
            context,
            step,
            trace,
        )

    if trace["match_mode"] == "none":
        trace["match_mode"] = (
            "grouped_recipe_action" if action_index > 0 else "active_recipe_context"
        )
    return None, ""


def _bind_candidate(
    state: WorkerState,
    context: ReflexReplayContext,
    candidate: ReflexCandidate,
    rejection_log: ReflexRejectionLog,
) -> ReflexSelection | None:
    transition_data = candidate.transition.model_dump(mode="json")
    before_state = dict(transition_data.get("before") or {})
    tool_calls: list[dict[str, Any]] = []
    tool_call_traces: dict[str, dict[str, Any]] = {}

    for action_index, recipe_action in enumerate(candidate.transition.actions):
        step = recipe_action.model_dump(mode="json")
        step.update(
            {
                "page_role": before_state.get("page_role", ""),
                "url_template": before_state.get("url_template", ""),
                "screen_context_signature": before_state.get(
                    "screen_context_signature",
                    {},
                ),
                "expected_after": candidate.transition.expected_after,
                "intent": step.get("intent") or candidate.transition.intent,
            }
        )
        trace = _step_trace(
            step,
            transition_index=candidate.transition_index,
            action_index=action_index,
            context=context,
        )
        marker_id, reject_reason = _match_action_screen(
            state,
            context,
            step,
            trace,
            action_index=action_index,
        )
        if reject_reason:
            rejection_log.reject(candidate.recipe_key, reject_reason, trace)
            return None

        args = build_reflex_action_args(step, marker_id, context.params)
        if args is None:
            rejection_log.reject(
                candidate.recipe_key,
                "args_build_failed",
                trace,
            )
            return None
        call_id = (
            f"reflex_{abs(hash(candidate.recipe_key))}_"
            f"{candidate.transition_index}_{action_index}"
        )
        tool_calls.append(
            {
                "name": step.get("action"),
                "args": args,
                "id": call_id,
            }
        )
        trace.update({"accepted": True, "tool_call_id": call_id})
        tool_call_traces[call_id] = dict(trace)

    if not tool_calls:
        rejection_log.reject(candidate.recipe_key, "candidate_invalid")
        return None
    return ReflexSelection(
        recipe_key=candidate.recipe_key,
        recipe=candidate.recipe,
        transition=candidate.transition,
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
    "build_reflex_action_args",
    "load_reflex_replay_context",
    "select_reflex_replay",
]

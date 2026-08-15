"""현재 화면에 적용된 경험 규칙을 실행하고 관찰 가능한 효과를 검증한다."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.runtime import Runtime

from agent.config import get_settings
from agent.recipe.replay import (
    ReflexReplayContext,
    ReflexSelection,
    load_reflex_replay_context,
    select_reflex_replay,
)
from agent.runtime.site_context import normalize_page_role
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.runtime.worker_contracts import (
    ScreenMarker,
    TransitionRequest,
    WorkerState,
    build_action_request,
)
from agent.utils.logger import logger
from agent.utils.text import recipe_url_scope_matches
from agent.vision.frame_compare import (
    changed_pixel_ratio,
    changed_region_ratio,
    load_gray_frame,
)
from shared.schema.experience_rule_schema import (
    ExpectedEffect,
    ExperienceRuleStep,
    ReplaySession,
    RuleApplication,
    RuleAction,
)


def replay_session_from_state(state: WorkerState) -> ReplaySession | None:
    raw_session = state["replay"].get("replay_session")
    if isinstance(raw_session, ReplaySession):
        return raw_session
    return ReplaySession.model_validate(raw_session) if raw_session else None


def blocked_recipe_keys(state: WorkerState) -> list[str]:
    return [
        str(key)
        for key in (state["replay"].get("reflex_blocked_recipe_keys") or [])
        if str(key)
    ]


def replay_session_after_transition(
    state: WorkerState,
    *,
    source: str,
    status: str,
) -> ReplaySession | None:
    session = replay_session_from_state(state)
    if not session or source != "reflex":
        return session
    return session.advance() if status == "ready" else None


def blocked_recipe_keys_after(
    state: WorkerState,
    request: TransitionRequest,
    *,
    should_block: bool,
) -> list[str]:
    keys = blocked_recipe_keys(state)
    recipe_key = str(request.get("recipe_key") or "")
    if should_block and recipe_key and recipe_key not in keys:
        keys.append(recipe_key)
    return keys


def _effect_frame_ratio(
    request: TransitionRequest,
    current_image_path: str,
    effect: ExpectedEffect,
) -> float:
    before_image_path = str(request.get("before_screenshot") or "")
    if not before_image_path or not current_image_path:
        raise ValueError("effect verification requires before and after captures")
    left = load_gray_frame(before_image_path)
    right = load_gray_frame(current_image_path)
    threshold = get_settings().reflex.visual_change_pixel_threshold
    if effect.kind == "target_region_change":
        return changed_region_ratio(
            left,
            right,
            effect.target_region_ratio,
            intensity_threshold=threshold,
        )
    return changed_pixel_ratio(left, right, intensity_threshold=threshold)


def verify_replay_after_state(
    request: TransitionRequest,
    state: WorkerState,
) -> tuple[bool, str, dict[str, Any]]:
    """저장된 규칙의 기대 효과가 현재 캡처에서 실제로 발생했는지 확인한다."""

    if request.get("execution_failed"):
        return False, "rule_action_group_failed", {}
    raw_effect = request.get("expected_effect")
    if raw_effect is None:
        return False, "rule_expected_effect_missing", {}
    effect = (
        raw_effect
        if isinstance(raw_effect, ExpectedEffect)
        else ExpectedEffect.model_validate(raw_effect)
    )
    observation = state["observation"]
    current_url = str(observation.get("current_url") or "")
    current_role = normalize_page_role(observation.get("current_page_role"))

    if effect.expected_url_template and not recipe_url_scope_matches(
        effect.expected_url_template,
        current_url,
    ):
        return False, "rule_effect_url_mismatch", {"current_url": current_url}
    expected_role = normalize_page_role(effect.expected_page_role)
    if expected_role and current_role and expected_role != current_role:
        return False, "rule_effect_page_role_mismatch", {
            "expected_page_role": expected_role,
            "current_page_role": current_role,
        }

    if effect.kind == "url_change":
        before_url = str(request.get("before_url") or "")
        changed = bool(
            current_url
            and before_url
            and current_url != before_url
            and effect.expected_url_template
        )
        return changed, (
            "rule_url_change_verified" if changed else "rule_url_did_not_change"
        ), {"before_url": before_url, "current_url": current_url}

    if effect.kind == "page_change":
        before_role = normalize_page_role(request.get("before_page_role"))
        changed = bool(current_role and before_role and current_role != before_role)
        return changed, (
            "rule_page_change_verified" if changed else "rule_page_did_not_change"
        ), {"before_page_role": before_role, "current_page_role": current_role}

    try:
        ratio = _effect_frame_ratio(
            request,
            str(observation.get("current_screenshot") or ""),
            effect,
        )
    except (OSError, ValueError) as exc:
        return False, "rule_effect_frame_unavailable", {"error": str(exc)[:200]}
    minimum = get_settings().reflex.visual_change_min_ratio
    changed = ratio >= max(0.0, minimum)
    reason = (
        "rule_target_region_change_verified"
        if effect.kind == "target_region_change" and changed
        else "rule_screen_change_verified"
        if changed
        else "rule_expected_region_unchanged"
        if effect.kind == "target_region_change"
        else "rule_expected_screen_unchanged"
    )
    return changed, reason, {"visual_change_ratio": ratio}


def record_replay_outcome(
    state: WorkerState,
    request: TransitionRequest,
    *,
    status: str,
    persist_result: Callable[[str, bool], object],
) -> None:
    if str(request.get("source") or "") != "reflex":
        return
    session = replay_session_from_state(state)
    recipe_key = str(
        (session.recipe_key if session else "") or request.get("recipe_key") or ""
    )
    if not recipe_key or not session or not session.pending_is_current():
        return
    succeeded = status == "ready"
    if succeeded and not session.is_last_step():
        return
    try:
        persist_result(recipe_key, succeeded)
    except Exception as exc:
        logger.warning(
            "Experience rule outcome persistence failed",
            recipe_key=recipe_key,
            error=str(exc),
        )


def _miss_result(
    state: WorkerState,
    reason: str,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reflex_trace = {**dict(trace or {}), "hit": False, "reason": reason}
    session = replay_session_from_state(state)
    blocked = blocked_recipe_keys(state)
    if session and session.recipe_key:
        reflex_trace.update(
            {
                "recipe_key": session.recipe_key,
                "recipe_step_index": session.current_step_index,
                "recipe_step_count": session.step_count,
                "path_failed": True,
            }
        )
        if session.recipe_key not in blocked:
            blocked.append(session.recipe_key)
    return {
        "replay": {
            "reflex_trace": reflex_trace,
            "replay_session": None,
            "reflex_blocked_recipe_keys": blocked,
        }
    }


def _build_request(
    selection: ReflexSelection,
    resolver_reasoning_call_count: int,
):
    return build_action_request(
        "reflex",
        "resolved experience rule step",
        selection.tool_calls,
        metadata={
            "recipe_key": selection.rule_key,
            "step_index": selection.step_index,
            "step_count": len(selection.rule.steps),
            "source_reasoning_call_count": len(
                selection.step.source_transition_seqs
            ),
            "resolver_reasoning_call_count": resolver_reasoning_call_count,
            "before_rule_screen": selection.step.before.model_dump(mode="json"),
            "expected_effect": selection.step.expected_effect.model_dump(mode="json"),
            "resolved_step": selection.resolved_step.model_dump(mode="json"),
            "transition_actions": [
                str(action.action) for action in selection.step.actions
            ],
        },
    )


def _hit_result(
    context: ReflexReplayContext,
    selection: ReflexSelection,
    resolver_reasoning_call_count: int,
) -> dict[str, Any]:
    step_count = len(selection.rule.steps)
    handles = (
        dict(context.replay_session.interaction_handles)
        if context.replay_session
        else {}
    )
    for rule_action, resolved_action in zip(
        selection.step.actions,
        selection.resolved_step.actions,
    ):
        if rule_action.action != "scroll" or resolved_action.target is None:
            continue
        target = rule_action.target
        if target is None:
            continue
        key = "|".join(
            value.strip().casefold()
            for value in (target.component, target.role, target.description)
            if value.strip()
        )
        if key:
            handles[key] = resolved_action.target.model_copy(deep=True)
    session = ReplaySession(
        recipe_key=selection.rule_key,
        current_step_index=selection.step_index,
        pending_step_index=selection.step_index,
        step_count=step_count,
        interaction_handles=handles,
    )
    return {
        "observation": {"current_markers": selection.markers},
        "decision": {
            "pending_action": _build_request(
                selection,
                resolver_reasoning_call_count,
            )
        },
        "replay": {
            "reflex_trace": {
                "hit": True,
                "recipe_key": selection.rule_key,
                "candidate_count": context.candidate_count,
                "task_category": context.task_category,
                "actions": [call["name"] for call in selection.tool_calls],
                "tool_calls": selection.tool_call_traces,
                "resolution_mode": selection.resolution_mode,
                "recipe_step_index": selection.step_index,
                "recipe_step_count": step_count,
                "source_reasoning_call_count": len(
                    selection.step.source_transition_seqs
                ),
                "resolver_reasoning_call_count": resolver_reasoning_call_count,
            },
            "replay_session": session,
        },
    }


def _detect_target_markers(
    runtime: Runtime[WorkerDependencies],
    context: ReflexReplayContext,
    action: RuleAction,
) -> list[ScreenMarker]:
    target = action.target
    if target is None or target.reference is None or not context.current_image_path:
        return []
    crop = list(target.reference_roi_signature.get("crop_rect_ratio") or [])
    if not crop:
        return []
    marker_type = target.reference.marker_type
    try:
        perception = runtime.context.vision.get_perception()
        markers = perception.detect_target_roi(
            Path(context.current_image_path),
            crop,
            marker_type,
        )
        if markers or marker_type not in {"text", "icon"}:
            return markers
        fallback_type = "icon" if marker_type == "text" else "text"
        return perception.detect_target_roi(
            Path(context.current_image_path),
            crop,
            fallback_type,
        )
    except Exception as exc:
        logger.warning(
            "Experience target ROI detection failed",
            action=action.action,
            marker_type=marker_type,
            error=str(exc),
        )
        return []


def attempt_reflex_replay(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> dict[str, Any]:
    """후보 조회, 현재 화면 해석과 실행 요청 조립을 순서대로 수행한다."""

    started = time.perf_counter()
    logger.info("Executing Reflex Node")
    context = load_reflex_replay_context(
        state,
        runtime.context.data.load_experience_rules,
    )
    if not context.rule_candidates:
        logger.info(
            "Reflex miss: no experience rule",
            site=context.site,
            task_category=context.task_category,
        )
        return _miss_result(
            state,
            "no_rule",
            {
                "candidate_count": 0,
                "site": context.site,
                "task_category": context.task_category,
            },
        )

    resolver_reasoning_call_count = 0

    def resolve_rule_targets(
        step: ExperienceRuleStep,
        markers: list[ScreenMarker],
        image_path: str,
    ) -> RuleApplication:
        nonlocal resolver_reasoning_call_count
        resolver_reasoning_call_count += 1
        return runtime.context.resolve_experience_rule(step, markers, image_path)

    selection, rejection_log = select_reflex_replay(
        state,
        context,
        lambda action: _detect_target_markers(runtime, context, action),
        resolve_rule_targets,
    )
    if selection is None:
        trace = rejection_log.trace_payload(context.candidate_count)
        trace["resolver_reasoning_call_count"] = resolver_reasoning_call_count
        logger.info(
            "Reflex miss: no rule applied",
            candidates=context.candidate_count,
            last_reason=rejection_log.last_reason,
            reject_reasons=rejection_log.reason_counts,
        )
        return _miss_result(state, "no_rule_applied", trace)

    elapsed = time.perf_counter() - started
    logger.info(
        "Reflex hit",
        recipe_key=selection.rule_key[:24],
        actions=[call["name"] for call in selection.tool_calls],
        resolution_mode=selection.resolution_mode,
        recipe_step=f"{selection.step_index + 1}/{len(selection.rule.steps)}",
        goal=selection.rule.goal[:80],
        duration=f"{elapsed:.3f}s",
    )
    return _hit_result(context, selection, resolver_reasoning_call_count)


__all__ = [
    "attempt_reflex_replay",
    "blocked_recipe_keys_after",
    "record_replay_outcome",
    "replay_session_after_transition",
    "verify_replay_after_state",
]

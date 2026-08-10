"""자율탐색 행동과 실제 전환 기록으로 경험 기반 탐색 경로를 만든다."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.runtime.site_context import normalize_page_role
from agent.runtime.worker_actions import (
    TARGET_REPLAY_ACTIONS,
    is_supported_recipe_action_group,
)
from shared.schema.feedback_schema import RecipeCandidate, RecordedTransition


def _sequence(item: dict[str, Any]) -> int | None:
    try:
        return int(item.get("seq"))
    except (TypeError, ValueError):
        return None


def _checkpoint_payload(stored: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation_id": str(stored.get("observation_id") or ""),
        "url_template": str(stored.get("url_template") or ""),
        "page_role": str(stored.get("page_role") or ""),
        "screen_context_signature": deepcopy(
            stored.get("screen_context_signature") or {}
        ),
    }


def _checkpoint_from_step(step: dict[str, Any]) -> dict[str, Any]:
    stored = step.get("before_state")
    return _checkpoint_payload(stored) if isinstance(stored, dict) and stored else {}


def _checkpoint_from_transition(record: RecordedTransition) -> dict[str, Any]:
    return _checkpoint_payload(record.after_state) if record.after_state else {}


def _action_from_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_seq": int(step.get("seq") or 0),
        "action": str(step.get("action") or ""),
        "target": deepcopy(step.get("target")),
        "roi_signature": deepcopy(step.get("roi_signature") or {}),
        "value": step.get("value"),
        "param": deepcopy(step.get("param") or {}),
        "is_param": bool(step.get("is_param")),
        "intent": str(step.get("intent") or ""),
        "target_role": str(step.get("target_role") or ""),
        "component": str(step.get("component") or ""),
        "slot_refs": list(step.get("slot_refs") or []),
        "risk_level": str(step.get("risk_level") or ""),
        "replay_mode": str(step.get("replay_mode") or "reasoning"),
    }


def _add_action_anchor(
    checkpoint: dict[str, Any],
    action: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(checkpoint)
    if (
        action.get("action") in TARGET_REPLAY_ACTIONS
        and isinstance(action.get("target"), dict)
        and action.get("roi_signature")
    ):
        out["anchor_target"] = deepcopy(action["target"])
        out["anchor_roi_signature"] = deepcopy(action["roi_signature"])
    return out


def _states_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """같은 실행에서 기록한 관찰 식별자가 연속되는지 확인한다."""

    left_id = str(left.get("observation_id") or "")
    right_id = str(right.get("observation_id") or "")
    return bool(left_id and right_id and left_id == right_id)


def _has_verifiable_identity(
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    if after.get("anchor_target") and after.get("anchor_roi_signature"):
        return True
    before_role = normalize_page_role(before.get("page_role"))
    after_role = normalize_page_role(after.get("page_role"))
    if before_role and after_role and before_role != after_role:
        return True
    if dict(after.get("screen_context_signature") or {}).get("phash"):
        return True
    before_url = str(before.get("url_template") or "")
    after_url = str(after.get("url_template") or "")
    return bool(after_url and after_url != before_url)


def _merge_commit_actions(
    transitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """입력 뒤 Enter처럼 코드가 허용한 단일 조합만 한 전이로 묶는다."""

    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(transitions):
        current = deepcopy(transitions[index])
        if index + 1 < len(transitions):
            following = transitions[index + 1]
            actions = list(current["actions"]) + list(following["actions"])
            if is_supported_recipe_action_group(actions) and _states_match(
                current["after"],
                following["before"],
            ):
                current["actions"] = actions
                current["after"] = deepcopy(following["after"])
                current["expected_after"] = str(
                    following.get("expected_after")
                    or current.get("expected_after")
                    or ""
                )
                current["intent"] = str(
                    following.get("intent") or current.get("intent") or ""
                )
                merged.append(current)
                index += 2
                continue
        merged.append(current)
        index += 1
    for transition_index, transition in enumerate(merged):
        transition["seq"] = transition_index
    return merged


def _ordered_replay_steps(
    replay_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (
            dict(step)
            for step in replay_steps or []
            if isinstance(step, dict) and _sequence(step) is not None
        ),
        key=lambda step: int(step["seq"]),
    )


def _remove_invalid_path_prefix(
    ordered: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    while ordered and str(ordered[0].get("action") or "") not in TARGET_REPLAY_ACTIONS:
        removed = ordered.pop(0)
        issues.append(
            {
                "seq": removed.get("seq"),
                "action": removed.get("action"),
                "reason": "recipe_must_start_with_target",
            }
        )


def _transition_after_states(
    candidate: RecipeCandidate,
) -> dict[int, dict[str, Any]]:
    states: dict[int, dict[str, Any]] = {}
    for record in candidate.submission.transition_records:
        if record.action_seq is None or not record.after_state:
            continue
        states[int(record.action_seq)] = _checkpoint_from_transition(record)
    return states


def _append_remaining_issues(
    ordered: list[dict[str, Any]],
    start_position: int,
    issues: list[dict[str, Any]],
    reason: str,
) -> None:
    for remaining in ordered[start_position:]:
        issues.append(
            {
                "seq": remaining.get("seq"),
                "action": remaining.get("action"),
                "reason": reason,
            }
        )


def _build_atomic_transitions(
    candidate: RecipeCandidate,
    ordered: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    after_states = _transition_after_states(candidate)
    atomic: list[dict[str, Any]] = []
    previous_after: dict[str, Any] | None = None
    for position, step in enumerate(ordered):
        seq = int(step["seq"])
        before = _checkpoint_from_step(step)
        after = deepcopy(after_states.get(seq) or {})
        if not before or not after:
            _append_remaining_issues(
                ordered,
                position,
                issues,
                "recorded_transition_missing",
            )
            break
        if previous_after is not None and not _states_match(previous_after, before):
            _append_remaining_issues(
                ordered,
                position,
                issues,
                "state_continuity_unproven",
            )
            break
        action = _action_from_step(step)
        atomic.append(
            {
                "seq": len(atomic),
                "before": _add_action_anchor(before, action),
                "actions": [action],
                "after": after,
                "expected_after": str(step.get("expected_after") or ""),
                "intent": str(step.get("intent") or ""),
            }
        )
        previous_after = after
    return atomic


def _remove_unverifiable_tail(
    transitions: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    while transitions and not _has_verifiable_identity(
        transitions[-1]["before"],
        transitions[-1]["after"],
    ):
        removed = transitions.pop()
        first_action = removed["actions"][0]
        issues.append(
            {
                "seq": first_action.get("source_seq"),
                "action": first_action.get("action"),
                "reason": "after_state_unverifiable",
            }
        )


def build_recipe_path(
    candidate: RecipeCandidate,
    replay_steps: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """가지치기된 행동 중 실제 전환이 이어지는 구간만 경로로 만든다."""

    ordered = _ordered_replay_steps(replay_steps)
    issues: list[dict[str, Any]] = []
    if not ordered:
        return None, issues
    _remove_invalid_path_prefix(ordered, issues)
    if not ordered:
        return None, issues
    transitions = _merge_commit_actions(
        _build_atomic_transitions(candidate, ordered, issues)
    )
    _remove_unverifiable_tail(transitions, issues)
    if not transitions:
        return None, issues
    return {
        "start_state": deepcopy(transitions[0]["before"]),
        "transitions": transitions,
        "completion_state": deepcopy(transitions[-1]["after"]),
    }, issues


__all__ = ["build_recipe_path"]

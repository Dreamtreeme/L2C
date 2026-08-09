"""자율탐색 기록을 검증 가능한 경험 기반 탐색 경로로 변환한다."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.config import get_settings
from agent.runtime.worker_actions import (
    TARGET_REPLAY_ACTIONS,
    is_supported_recipe_action_group,
)
from agent.runtime.site_context import normalize_page_role
from agent.utils.text import url_template
from agent.vision.screen_signature import (
    compact_screen_context_signature,
    hamming_distance,
)


def _sequence(item: dict[str, Any]) -> int | None:
    try:
        return int(item.get("seq"))
    except (TypeError, ValueError):
        return None


def _checkpoint(
    *,
    capture_id: Any = "",
    url: Any = "",
    page_role: Any = "",
    screen_signature: Any = None,
) -> dict[str, Any]:
    context_signature = compact_screen_context_signature(
        screen_signature if isinstance(screen_signature, dict) else {}
    )
    return {
        "capture_id": str(capture_id or ""),
        "url_template": url_template(str(url or "")),
        "page_role": str(page_role or ""),
        "screen_context_signature": context_signature,
    }


def _checkpoint_from_step(step: dict[str, Any]) -> dict[str, Any]:
    stored = (
        dict(step.get("before_state") or {})
        if isinstance(step.get("before_state"), dict)
        else {}
    )
    if not stored:
        return {}
    return {
        "capture_id": str(stored.get("capture_id") or ""),
        "url_template": str(stored.get("url_template") or ""),
        "page_role": str(stored.get("page_role") or ""),
        "screen_context_signature": dict(stored.get("screen_context_signature") or {}),
    }


def _checkpoint_from_transition(
    record: dict[str, Any],
) -> dict[str, Any]:
    stored = (
        dict(record.get("after_state") or {})
        if isinstance(record.get("after_state"), dict)
        else {}
    )
    if not stored:
        return {}
    return {
        "capture_id": str(stored.get("capture_id") or ""),
        "url_template": str(stored.get("url_template") or ""),
        "page_role": str(stored.get("page_role") or ""),
        "screen_context_signature": dict(stored.get("screen_context_signature") or {}),
    }


def _feedback_before_states(
    candidate: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    payload = dict(candidate.get("payload", {}) or {})
    states: dict[int, dict[str, Any]] = {}
    for episode in payload.get("feedback_episodes", []) or []:
        if not isinstance(episode, dict):
            continue
        seq = _sequence(episode)
        if seq is None:
            continue
        observation = (
            episode.get("observation")
            if isinstance(episode.get("observation"), dict)
            else {}
        )
        before = (
            observation.get("before")
            if isinstance(observation.get("before"), dict)
            else {}
        )
        states[seq] = _checkpoint(
            capture_id=before.get("capture_id"),
            url=before.get("url"),
            page_role=before.get("page_role"),
            screen_signature=before.get("screen_signature"),
        )
    return states


def _transition_after_states(
    candidate: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    payload = dict(candidate.get("payload", {}) or {})
    states: dict[int, dict[str, Any]] = {}
    for record in payload.get("transition_records", []) or []:
        if not isinstance(record, dict):
            continue
        try:
            seq = int(record.get("action_seq"))
        except (TypeError, ValueError):
            continue
        states[seq] = _checkpoint_from_transition(record)
    return states


def _merge_checkpoint(
    primary: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """서로 다른 기록원이 가진 체크포인트 필드를 손실 없이 합친다."""

    merged = deepcopy(fallback)
    for key, value in primary.items():
        if value not in (None, "", {}, []):
            merged[key] = deepcopy(value)
    return merged


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
        "needs_user_confirmation": bool(step.get("needs_user_confirmation")),
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


def _states_match(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    """Critic이 중간 행동을 삭제해도 같은 화면이 이어지는지 확인한다."""

    left_capture = str(left.get("capture_id") or "")
    right_capture = str(right.get("capture_id") or "")
    if left_capture and right_capture and left_capture == right_capture:
        return True

    left_url = str(left.get("url_template") or "")
    right_url = str(right.get("url_template") or "")
    if left_url and right_url and left_url != right_url:
        return False

    left_signature = dict(left.get("screen_context_signature") or {})
    right_signature = dict(right.get("screen_context_signature") or {})
    distance = hamming_distance(
        str(left_signature.get("phash") or ""),
        str(right_signature.get("phash") or ""),
    )
    if distance is None:
        return False
    return distance <= get_settings().reflex.screen_context_phash_max_distance


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
    """입력과 Enter처럼 하나의 화면 전환을 만드는 행동만 합친다."""

    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(transitions):
        current = deepcopy(transitions[index])
        if index + 1 < len(transitions):
            following = transitions[index + 1]
            actions = list(current.get("actions") or []) + list(
                following.get("actions") or []
            )
            if is_supported_recipe_action_group(actions) and _states_match(
                dict(current.get("after") or {}),
                dict(following.get("before") or {}),
            ):
                current["actions"] = actions
                current["after"] = deepcopy(following.get("after") or {})
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


def _raw_steps_by_seq(candidate: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(step["seq"]): dict(step)
        for step in candidate.get("steps", []) or []
        if isinstance(step, dict) and _sequence(step) is not None
    }


def _before_checkpoint(
    step: dict[str, Any],
    feedback_before: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    seq = int(step["seq"])
    before = _checkpoint_from_step(step)
    if seq in feedback_before:
        return _merge_checkpoint(before, feedback_before[seq])
    return before


def _after_checkpoint(
    seq: int,
    raw_steps: dict[int, dict[str, Any]],
    feedback_before: dict[int, dict[str, Any]],
    transition_after: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    after = deepcopy(transition_after.get(seq) or {})
    if after:
        return after
    next_step = raw_steps.get(seq + 1)
    if not next_step:
        return {}
    after = _checkpoint_from_step(next_step)
    if seq + 1 in feedback_before:
        after = _merge_checkpoint(after, feedback_before[seq + 1])
    return after


def _append_continuity_issues(
    ordered: list[dict[str, Any]],
    start_position: int,
    issues: list[dict[str, Any]],
) -> None:
    for remaining in ordered[start_position:]:
        issues.append(
            {
                "seq": remaining.get("seq"),
                "action": remaining.get("action"),
                "reason": "state_continuity_unproven",
            }
        )


def _build_atomic_transitions(
    candidate: dict[str, Any],
    ordered: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_steps = _raw_steps_by_seq(candidate)
    feedback_before = _feedback_before_states(candidate)
    transition_after = _transition_after_states(candidate)
    atomic: list[dict[str, Any]] = []
    previous_seq: int | None = None
    previous_after: dict[str, Any] | None = None
    for position, step in enumerate(ordered):
        seq = int(step["seq"])
        before = _before_checkpoint(step, feedback_before)
        after = _after_checkpoint(
            seq,
            raw_steps,
            feedback_before,
            transition_after,
        )
        discontinuous = (
            previous_seq is not None
            and seq != previous_seq + 1
            and previous_after is not None
            and not _states_match(previous_after, before)
        )
        if discontinuous:
            _append_continuity_issues(ordered, position, issues)
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
        previous_seq = seq
        previous_after = after
    return atomic


def _anchor_following_actions(transitions: list[dict[str, Any]]) -> None:
    for index in range(len(transitions) - 1):
        next_action = transitions[index + 1]["actions"][0]
        transitions[index]["after"] = _add_action_anchor(
            transitions[index]["after"],
            next_action,
        )


def _remove_unverifiable_tail(
    transitions: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    while transitions and not _has_verifiable_identity(
        transitions[-1]["before"],
        transitions[-1]["after"],
    ):
        removed = transitions.pop()
        first_action = (removed.get("actions") or [{}])[0]
        issues.append(
            {
                "seq": first_action.get("source_seq"),
                "action": first_action.get("action"),
                "reason": "after_state_unverifiable",
            }
        )


def build_recipe_path(
    candidate: dict[str, Any],
    replay_steps: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """가지치기된 행동을 하나의 연속 상태 전이 경로로 만든다."""

    ordered = _ordered_replay_steps(replay_steps)
    issues: list[dict[str, Any]] = []
    if not ordered:
        return None, issues
    _remove_invalid_path_prefix(ordered, issues)
    if not ordered:
        return None, issues
    atomic = _build_atomic_transitions(candidate, ordered, issues)
    transitions = _merge_commit_actions(atomic)
    if not transitions:
        return None, issues
    _anchor_following_actions(transitions)
    _remove_unverifiable_tail(transitions, issues)
    if not transitions:
        return None, issues

    for index, transition in enumerate(transitions):
        transition["seq"] = index
    return {
        "start_state": deepcopy(transitions[0]["before"]),
        "transitions": transitions,
        "completion_state": deepcopy(transitions[-1]["after"]),
    }, issues


__all__ = ["build_recipe_path"]

"""자율탐색 기록을 검증 가능한 경험 기반 탐색 경로로 변환한다."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from agent.config import get_settings
from agent.recipe.replay_actions import (
    TARGET_REPLAY_ACTIONS,
    is_supported_recipe_action_group,
)
from agent.recipe.text_utils import url_template
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
    if stored:
        return {
            "capture_id": str(stored.get("capture_id") or ""),
            "url_template": str(
                stored.get("url_template")
                or url_template(str(stored.get("url") or ""))
            ),
            "page_role": str(stored.get("page_role") or ""),
            "screen_context_signature": dict(
                stored.get("screen_context_signature") or {}
            ),
        }
    return {
        "capture_id": str(step.get("decision_capture_id") or ""),
        "url_template": str(step.get("url_template") or ""),
        "page_role": str(step.get("page_role") or ""),
        "screen_context_signature": dict(
            step.get("screen_context_signature") or {}
        ),
    }


def _checkpoint_from_transition(
    record: dict[str, Any],
) -> dict[str, Any]:
    stored = (
        dict(record.get("after_state") or {})
        if isinstance(record.get("after_state"), dict)
        else {}
    )
    if stored:
        return {
            "capture_id": str(
                stored.get("capture_id")
                or record.get("to_capture_id")
                or ""
            ),
            "url_template": str(
                stored.get("url_template")
                or url_template(str(stored.get("url") or ""))
            ),
            "page_role": str(stored.get("page_role") or ""),
            "screen_context_signature": dict(
                stored.get("screen_context_signature") or {}
            ),
        }

    screenshot = str(record.get("screenshot") or "")
    screen_signature: dict[str, Any] = {}
    if screenshot and Path(screenshot).is_file():
        try:
            from agent.vision.screen_signature import compute_screen_signature

            screen_signature = compute_screen_signature(screenshot, [])
        except Exception:
            screen_signature = {}
    return _checkpoint(
        capture_id=record.get("to_capture_id"),
        url=record.get("current_url"),
        page_role=record.get("page_role"),
        screen_signature=screen_signature,
    )


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
        "needs_user_confirmation": bool(
            step.get("needs_user_confirmation")
        ),
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
        out["anchor_roi_signature"] = deepcopy(
            action["roi_signature"]
        )
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
    return (
        distance
        <= get_settings().reflex.screen_context_phash_max_distance
    )


def _has_verifiable_identity(
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    if after.get("anchor_target") and after.get("anchor_roi_signature"):
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
            if (
                is_supported_recipe_action_group(actions)
                and _states_match(
                    dict(current.get("after") or {}),
                    dict(following.get("before") or {}),
                )
            ):
                current["actions"] = actions
                current["after"] = deepcopy(following.get("after") or {})
                current["expected_after"] = str(
                    following.get("expected_after")
                    or current.get("expected_after")
                    or ""
                )
                current["intent"] = str(
                    following.get("intent")
                    or current.get("intent")
                    or ""
                )
                merged.append(current)
                index += 2
                continue
        merged.append(current)
        index += 1
    for transition_index, transition in enumerate(merged):
        transition["seq"] = transition_index
    return merged


def build_recipe_path(
    candidate: dict[str, Any],
    replay_steps: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """가지치기된 행동을 하나의 연속 상태 전이 경로로 만든다."""

    ordered = sorted(
        (
            dict(step)
            for step in replay_steps or []
            if isinstance(step, dict) and _sequence(step) is not None
        ),
        key=lambda step: int(step["seq"]),
    )
    issues: list[dict[str, Any]] = []
    if not ordered:
        return None, issues

    while (
        ordered
        and str(ordered[0].get("action") or "")
        not in TARGET_REPLAY_ACTIONS
    ):
        removed = ordered.pop(0)
        issues.append(
            {
                "seq": removed.get("seq"),
                "action": removed.get("action"),
                "reason": "recipe_must_start_with_target",
            }
        )
    if not ordered:
        return None, issues

    raw_steps = {
        int(step["seq"]): dict(step)
        for step in candidate.get("steps", []) or []
        if isinstance(step, dict) and _sequence(step) is not None
    }
    feedback_before = _feedback_before_states(candidate)
    transition_after = _transition_after_states(candidate)
    atomic: list[dict[str, Any]] = []
    previous_seq: int | None = None
    previous_after: dict[str, Any] | None = None

    for position, step in enumerate(ordered):
        seq = int(step["seq"])
        before = _checkpoint_from_step(step)
        if seq in feedback_before:
            before = _merge_checkpoint(
                before,
                feedback_before[seq],
            )

        after = deepcopy(transition_after.get(seq) or {})
        if not after:
            next_raw = raw_steps.get(seq + 1)
            if next_raw:
                after = _checkpoint_from_step(next_raw)
                if seq + 1 in feedback_before:
                    after = _merge_checkpoint(
                        after,
                        feedback_before[seq + 1],
                    )

        if (
            previous_seq is not None
            and seq != previous_seq + 1
            and previous_after is not None
            and not _states_match(previous_after, before)
        ):
            for remaining in ordered[position:]:
                issues.append(
                    {
                        "seq": remaining.get("seq"),
                        "action": remaining.get("action"),
                        "reason": "state_continuity_unproven",
                    }
                )
            break

        action = _action_from_step(step)
        before = _add_action_anchor(before, action)
        atomic.append(
            {
                "seq": len(atomic),
                "before": before,
                "actions": [action],
                "after": after,
                "expected_after": str(step.get("expected_after") or ""),
                "intent": str(step.get("intent") or ""),
            }
        )
        previous_seq = seq
        previous_after = after

    transitions = _merge_commit_actions(atomic)
    if not transitions:
        return None, issues

    for index in range(len(transitions) - 1):
        next_action = transitions[index + 1]["actions"][0]
        transitions[index]["after"] = _add_action_anchor(
            transitions[index]["after"],
            next_action,
        )

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

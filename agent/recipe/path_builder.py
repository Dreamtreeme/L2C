"""관찰된 전이에서 비평가가 남긴 행동만 경험 경로로 만든다."""

from __future__ import annotations

from typing import Literal, TypedDict

from agent.runtime.site_context import normalize_page_role
from agent.runtime.worker_actions import (
    RECIPE_COMMIT_ACTIONS,
    TARGET_REPLAY_ACTIONS,
    is_supported_recipe_action_group,
)
from shared.schema.feedback_schema import RecipeCandidate
from shared.schema.recipe_schema import (
    ExperiencePath,
    ExperienceTransition,
    PhysicalAction,
    PhysicalActionName,
    ScreenCheckpoint,
)


PathIssueReason = Literal[
    "observed_transition_missing",
    "state_continuity_unproven",
    "path_must_start_with_target",
    "after_state_unverifiable",
]


class PathBuildIssue(TypedDict):
    """경험 경로에 포함하지 못한 행동과 이유."""

    seq: int
    action: PhysicalActionName
    reason: PathIssueReason


def _issue(action: PhysicalAction, reason: PathIssueReason) -> PathBuildIssue:
    return {
        "seq": action.source_seq,
        "action": action.action,
        "reason": reason,
    }


def _has_verifiable_identity(
    before: ScreenCheckpoint,
    after: ScreenCheckpoint,
) -> bool:
    if after.has_anchor():
        return True
    before_role = normalize_page_role(before.page_role)
    after_role = normalize_page_role(after.page_role)
    if before_role and after_role and before_role != after_role:
        return True
    if after.has_context_phash():
        return True
    return bool(after.url_template and after.url_template != before.url_template)


def _ordered_actions(actions: list[PhysicalAction]) -> list[PhysicalAction]:
    return sorted(
        (action.model_copy(deep=True) for action in actions),
        key=lambda action: action.source_seq,
    )


def _append_issues(
    actions: list[PhysicalAction],
    start: int,
    issues: list[PathBuildIssue],
    reason: PathIssueReason,
) -> None:
    issues.extend(_issue(action, reason) for action in actions[start:])


def _atomic_transitions(
    candidate: RecipeCandidate,
    actions: list[PhysicalAction],
    issues: list[PathBuildIssue],
) -> list[ExperienceTransition]:
    transitions: list[ExperienceTransition] = []
    previous_after: ScreenCheckpoint | None = None
    for position, action in enumerate(actions):
        observed = candidate.transition_for_action(action.source_seq)
        if observed is None:
            _append_issues(
                actions,
                position,
                issues,
                "observed_transition_missing",
            )
            break
        if (
            previous_after is not None
            and not previous_after.same_observation_as(observed.before)
        ):
            _append_issues(
                actions,
                position,
                issues,
                "state_continuity_unproven",
            )
            break
        transitions.append(
            ExperienceTransition(
                seq=len(transitions),
                before=observed.before.with_action_anchor(action),
                actions=[action.model_copy(deep=True)],
                after=observed.after.model_copy(deep=True),
                expected_after=observed.expected_after,
                intent=observed.intent,
            )
        )
        previous_after = observed.after
    return transitions


def _link_next_action_anchors(
    transitions: list[ExperienceTransition],
) -> list[ExperienceTransition]:
    linked = [transition.model_copy(deep=True) for transition in transitions]
    for index, (current, following) in enumerate(zip(linked, linked[1:])):
        if not current.after.same_observation_as(following.before):
            continue
        if not following.before.has_anchor():
            continue
        linked[index] = current.model_copy(
            update={
                "after": current.after.model_copy(
                    deep=True,
                    update={
                        "anchor_target": following.before.anchor_target.model_copy(
                            deep=True
                        ),
                        "anchor_roi_signature": dict(
                            following.before.anchor_roi_signature
                        ),
                    },
                )
            }
        )
    return linked


def _merge_commit_actions(
    transitions: list[ExperienceTransition],
) -> list[ExperienceTransition]:
    """입력 뒤 Enter처럼 한 효과를 만드는 연속 행동을 한 전이로 묶는다."""

    merged: list[ExperienceTransition] = []
    index = 0
    while index < len(transitions):
        current = transitions[index].model_copy(deep=True)
        if index + 1 < len(transitions):
            following = transitions[index + 1]
            actions = current.actions + [
                action.model_copy(deep=True) for action in following.actions
            ]
            following_action = following.actions[0].action
            if (
                following_action in RECIPE_COMMIT_ACTIONS
                and is_supported_recipe_action_group(actions)
                and current.after.same_observation_as(following.before)
            ):
                current = current.model_copy(
                    update={
                        "actions": actions,
                        "after": following.after.model_copy(deep=True),
                        "expected_after": (
                            following.expected_after or current.expected_after
                        ),
                        "intent": following.intent or current.intent,
                    }
                )
                merged.append(current)
                index += 2
                continue
        merged.append(current)
        index += 1
    return [
        transition.model_copy(update={"seq": transition_index})
        for transition_index, transition in enumerate(merged)
    ]


def _remove_unverifiable_tail(
    transitions: list[ExperienceTransition],
    issues: list[PathBuildIssue],
) -> None:
    while transitions and not _has_verifiable_identity(
        transitions[-1].before,
        transitions[-1].after,
    ):
        removed = transitions.pop()
        issues.append(_issue(removed.actions[0], "after_state_unverifiable"))


def build_experience_path(
    candidate: RecipeCandidate,
    replay_actions: list[PhysicalAction],
) -> tuple[ExperiencePath | None, list[PathBuildIssue]]:
    """연속된 후보 행동 하나를 검증 가능한 경험 경로로 만든다."""

    actions = _ordered_actions(replay_actions)
    issues: list[PathBuildIssue] = []
    while actions and actions[0].action not in TARGET_REPLAY_ACTIONS:
        issues.append(_issue(actions.pop(0), "path_must_start_with_target"))
    if not actions:
        return None, issues

    transitions = _atomic_transitions(candidate, actions, issues)
    transitions = _link_next_action_anchors(transitions)
    transitions = _merge_commit_actions(transitions)
    _remove_unverifiable_tail(transitions, issues)
    if not transitions:
        return None, issues
    return ExperiencePath(transitions=transitions), issues


def _continuous_action_groups(
    candidate: RecipeCandidate,
    replay_actions: list[PhysicalAction],
    issues: list[PathBuildIssue],
) -> list[list[PhysicalAction]]:
    groups: list[list[PhysicalAction]] = []
    current: list[PhysicalAction] = []
    previous_after: ScreenCheckpoint | None = None
    for action in _ordered_actions(replay_actions):
        observed = candidate.transition_for_action(action.source_seq)
        if observed is None:
            if current:
                groups.append(current)
                current = []
            issues.append(_issue(action, "observed_transition_missing"))
            previous_after = None
            continue
        if (
            previous_after is not None
            and not previous_after.same_observation_as(observed.before)
        ):
            groups.append(current)
            current = []
        current.append(action)
        previous_after = observed.after
    if current:
        groups.append(current)
    return groups


def build_experience_paths(
    candidate: RecipeCandidate,
    replay_actions: list[PhysicalAction],
) -> tuple[list[ExperiencePath], list[PathBuildIssue]]:
    """가변 판단으로 끊긴 성공 기록을 각각의 경험 경로로 만든다."""

    paths: list[ExperiencePath] = []
    issues: list[PathBuildIssue] = []
    for group in _continuous_action_groups(candidate, replay_actions, issues):
        path, group_issues = build_experience_path(candidate, group)
        issues.extend(group_issues)
        if path:
            paths.append(path)
    return paths, issues


__all__ = [
    "PathBuildIssue",
    "build_experience_path",
    "build_experience_paths",
]

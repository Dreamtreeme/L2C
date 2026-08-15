"""자율탐색 경험 후보를 검토하고 활성 경로로 승격한다."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.config import get_settings
from agent.recipe.candidate_promotion import (
    PrunedTransition,
    continuous_transition_groups,
    is_preparation_transition,
    retained_candidate_path,
    reviewable_candidate_transitions,
    transitions_are_continuous,
)
from agent.recipe.store import ExperienceRuleStore
from agent.recipe.task_category import normalize_task_category
from agent.utils.text import normalize_text, url_template
from agent.utils.image_utils import image_to_base64_jpeg
from shared.schema.feedback_schema import (
    ExecutionEvent,
    RecipeCandidate,
    RecipeCandidateReview,
)
from shared.schema.execution_record_schema import (
    ObservedAction,
    ObservedTransition,
    ScreenCheckpoint,
)


CriticFn = Callable[[dict[str, Any]], dict[str, Any] | RecipeCandidateReview]
ReviewProcessMode = Literal["review", "promote"]


class RulePromotionResult(BaseModel):
    """검토된 원본 경로를 경험 규칙으로 저장한 결과."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    promoted: bool = False
    saved_count: int = 0
    rule_step_count: int = 0
    rule_action_count: int = 0
    pruned_transitions: list[PrunedTransition] = Field(default_factory=list)


def _without_empty_values(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item not in (None, "", [], {})
    }


def _compact_observation_id(value: str) -> str:
    marker = ":observation:"
    return (
        "observation:" + value.rsplit(marker, 1)[-1]
        if marker in value
        else value
    )


def _compact_checkpoint(checkpoint: ScreenCheckpoint | None) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    return _without_empty_values(
        {
            "observation_id": _compact_observation_id(checkpoint.observation_id),
            "url_template": url_template(checkpoint.url_template),
            "page_role": checkpoint.page_role,
        }
    )


def _compact_texts(values: list[Any]) -> list[str]:
    limit = get_settings().recipe.critic_evidence_text_limit
    texts: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_text(value)
        key = normalized.casefold().replace(" ", "")
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        texts.append(normalized)
        if len(texts) >= limit:
            break
    return texts


def _compact_action(action: ObservedAction) -> dict[str, Any]:
    target = action.target
    param: dict[str, Any] = {}
    if action.action == "type_in_marker":
        param["slot_name"] = action.param.slot_name
    elif action.action == "press_key":
        param["key"] = action.param.key
    return _without_empty_values(
        {
            "seq": action.source_seq,
            "action": action.action,
            "target": _without_empty_values(
                {
                    "text": target.text if target else "",
                    "semantic_label": target.semantic_label if target else "",
                    "marker_type": target.marker_type if target else "",
                    "region": target.region if target else "",
                    "center_ratio": target.center_ratio if target else [],
                }
            ),
            "roi_available": bool(action.roi_signature.get("phash")),
            "param": _without_empty_values(param),
            "slot_refs": list(action.slot_refs),
            "intent": action.intent,
            "target_role": action.target_role,
            "component": action.component,
        }
    )


def _compact_transition(
    transition: ObservedTransition,
    *,
    can_follow_seqs: list[int],
    prepares_transition_seq: int | None = None,
) -> dict[str, Any]:
    evidence = transition.evidence
    return _without_empty_values(
        {
            "seq": transition.seq,
            "can_follow_seqs": can_follow_seqs,
            "before": _compact_checkpoint(transition.before),
            "actions": [_compact_action(action) for action in transition.actions],
            "after": _compact_checkpoint(transition.after),
            "expected_after": transition.expected_after,
            "intent": transition.intent,
            "prepares_transition_seq": prepares_transition_seq,
            "evidence": _without_empty_values(
                {
                    "source": evidence.source if evidence else "",
                    "result_status": evidence.result_status if evidence else "",
                    "status": evidence.status if evidence else "",
                    "outcome": evidence.outcome if evidence else "",
                    "reason": evidence.reason if evidence else "",
                    "visual_change_ratio": (
                        evidence.visual_change_ratio if evidence else None
                    ),
                    "after_text_sample": _compact_texts(
                        list(evidence.after_marker_texts) if evidence else []
                    ),
                }
            ),
        }
    )


def _compact_trajectory_event(
    event: ExecutionEvent,
    candidate_action_seqs: set[int],
) -> dict[str, Any]:
    action = event.candidate_action
    result = dict(event.result or {})
    evidence = event.transition.evidence if event.transition else None
    return _without_empty_values(
        {
            "seq": event.seq,
            "action": (
                action.action
                if action
                else result.get("action") or result.get("requested_action")
            ),
            "candidate": event.seq in candidate_action_seqs,
            "source": result.get("action_source")
            or (evidence.source if evidence else ""),
            "before": _compact_checkpoint(event.before_checkpoint),
            "intent": event.intent or (action.intent if action else ""),
            "result": _without_empty_values(
                {
                    "status": result.get("status"),
                    "reason": result.get("reason") or result.get("error"),
                }
            ),
            "transition": (
                _without_empty_values(
                    {
                        "after": _compact_checkpoint(event.transition.after),
                        "status": evidence.status if evidence else "",
                        "outcome": evidence.outcome if evidence else "",
                        "reason": evidence.reason if evidence else "",
                    }
                )
                if event.transition
                else {}
            ),
        }
    )


def _compact_collection_request(candidate: RecipeCandidate) -> dict[str, Any]:
    intent = candidate.collection_intent
    return _without_empty_values(
        {
            "site": candidate.site,
            "task_category": normalize_task_category(intent.task_category),
            "search_keyword": intent.search_keyword,
            "count_mode": intent.count_mode,
            "target_count": intent.target_count,
            "filters": _without_empty_values(intent.filters.model_dump()),
            "purpose": intent.purpose,
            "required_fields": list(intent.required_fields),
        }
    )


def _candidate_screen_evidence(
    candidate: RecipeCandidate,
    transitions: list[ObservedTransition],
) -> list[tuple[str, Path]]:
    """비평가가 행동 순서를 직접 볼 수 있도록 전이 전후 화면을 고른다."""

    limit = get_settings().recipe.critic_evidence_image_limit
    if limit <= 0 or not transitions:
        return []
    events_by_transition = {
        event.transition.seq: event
        for event in candidate.action_events
        if event.transition is not None
    }
    evidence: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()

    def append(label: str, raw_path: Any) -> None:
        if len(evidence) >= limit:
            return
        path_text = str(raw_path or "").strip()
        if not path_text:
            return
        path = Path(path_text)
        if path in seen_paths or not path.is_file():
            return
        seen_paths.add(path)
        evidence.append((label, path))

    for index, transition in enumerate(transitions):
        event = events_by_transition.get(transition.seq)
        if index == 0 and event is not None:
            append(
                f"transition {transition.seq} before",
                event.result.get("before_marked_image")
                or event.result.get("before_screenshot"),
            )
        transition_evidence = transition.evidence
        if transition_evidence is not None:
            append(
                f"transition {transition.seq} after",
                transition_evidence.marked_image or transition_evidence.screenshot,
            )
    return evidence


def build_candidate_review_payload(
    candidate: RecipeCandidate,
) -> dict[str, Any]:
    """비평가에게 실행 순서와 재생 가능한 전이만 전달한다."""

    all_transitions = candidate.transitions
    transitions, _pruned = reviewable_candidate_transitions(candidate)
    can_follow_by_seq = {
        transition.seq: [
            following.seq
            for following in transitions[index + 1 :]
            if transitions_are_continuous(transition, following)
        ]
        for index, transition in enumerate(transitions)
    }
    following_by_seq = {
        transition.seq: all_transitions[index + 1]
        for index, transition in enumerate(all_transitions[:-1])
    }
    candidate_action_seqs = {
        action.source_seq
        for transition in transitions
        for action in transition.actions
    }
    screen_evidence = _candidate_screen_evidence(candidate, transitions)
    return {
        "run_id": candidate.run_id,
        "status": candidate.status,
        "request": _compact_collection_request(candidate),
        "trajectory": [
            _compact_trajectory_event(event, candidate_action_seqs)
            for event in candidate.action_events
        ],
        "candidate_transitions": [
            _compact_transition(
                transition,
                can_follow_seqs=can_follow_by_seq[transition.seq],
                prepares_transition_seq=(
                    following.seq
                    if (
                        (following := following_by_seq.get(transition.seq))
                        and is_preparation_transition(transition, following)
                    )
                    else None
                ),
            )
            for transition in transitions
        ],
        "required_transition_verdicts": [
            {
                "seq": transition.seq,
                "actions": [action.action for action in transition.actions],
            }
            for transition in transitions
        ],
        "screen_evidence_order": [label for label, _path in screen_evidence],
        "worker_execution": _without_empty_values(
            {
                "run_status": candidate.run_status,
                "collected_count": candidate.collected_count,
                "persisted_count": candidate.persisted_count,
            }
        ),
    }


def _coerce_review(
    raw: dict[str, Any] | RecipeCandidateReview,
) -> RecipeCandidateReview:
    return (
        raw
        if isinstance(raw, RecipeCandidateReview)
        else RecipeCandidateReview.model_validate(raw)
    )


def _fallback_review(reason: str) -> RecipeCandidateReview:
    return RecipeCandidateReview(
        decision="reject",
        reasons=[reason],
        feedback_to_worker="비평가가 전이 유지 여부를 확정하지 못했습니다.",
    )


def _llm_review_candidate(
    payload: dict[str, Any],
    screen_evidence: list[tuple[str, Path]],
) -> RecipeCandidateReview:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.llm.clients import get_structured_google_model
    from agent.llm.policy import commander_model_name
    from agent.observability.run_context import invoke_with_metrics
    from agent.prompts.trust_boundary import external_content_contract_en

    model_name = get_settings().models.recipe_critic_model or commander_model_name()
    llm = get_structured_google_model(
        model_name,
        RecipeCandidateReview,
        temperature=0.0,
        execution_role="critic",
    )
    messages = [
        SystemMessage(
            content=(
                external_content_contract_en()
                + "\nYou are the Reflex Recipe Critic. Autonomous exploration already "
                "recorded executable transition groups with before and after screens. "
                "You may only keep or drop those groups. Return exactly one "
                "transition_verdict for every required_transition_verdicts seq. "
                "Keep groups on the successful reusable route. Drop no-op, wrong-target, "
                "abandoned-branch, recovery, and variable-choice groups. Use the full "
                "trajectory and ordered screen evidence to recognize actions that were "
                "later undone. Do not rewrite actions, targets, parameters, screen "
                "evidence, or metadata. "
                "A transition with prepares_transition_seq is a successful input that "
                "prepared the following submit action even though the screen stayed stable; "
                "keep it when the successful route needs that input. "
                "The kept transitions, in seq order, must form one continuous screen path. "
                "For every kept transition except the last, the next kept seq must be listed "
                "in its can_follow_seqs. This relation already allows a later action to "
                "reconnect after dropped detours only when the recorded screen is exactly "
                "the same. If multiple disconnected reusable paths exist, keep the one that "
                "most directly supports the recurring request and drop the others. Return "
                "accept when that one path retains at least one transition. "
                "Example 1: parameterized search input followed by search submit should be "
                "kept, while clicking one job card chosen from current search results should "
                "be dropped as variable-choice. "
                "Example 2: when a wrong-menu click and recovery return to the previous "
                "screen, drop both detour groups and keep the successful actions before and "
                "after that detour."
            )
        ),
    ]
    payload_text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if screen_evidence:
        vision = get_settings().vision
        content: list[dict[str, Any]] = [{"type": "text", "text": payload_text}]
        for label, path in screen_evidence:
            encoded = image_to_base64_jpeg(
                path,
                max_dim=vision.reasoning_image_max_dim,
                quality=vision.reasoning_image_quality,
                fast=True,
            )
            content.extend(
                [
                    {"type": "text", "text": f"Screen evidence: {label}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                        },
                    },
                ]
            )
        messages.append(HumanMessage(content=content))
    else:
        messages.append(HumanMessage(content=payload_text))
    return _coerce_review(invoke_with_metrics(llm, messages, "recipe_critic"))


def _verdict_contract_errors(
    transitions: list[ObservedTransition],
    review: RecipeCandidateReview,
) -> list[str]:
    required = {transition.seq for transition in transitions}
    counts: dict[int, int] = {}
    for verdict in review.transition_verdicts:
        counts[verdict.seq] = counts.get(verdict.seq, 0) + 1
    errors = [
        f"seq={seq} verdict_count={counts.get(seq, 0)}"
        for seq in sorted(required)
        if counts.get(seq, 0) != 1
    ]
    errors.extend(
        f"unexpected seq={seq}" for seq in sorted(set(counts) - required)
    )
    if review.decision == "accept" and not any(
        verdict.keep for verdict in review.transition_verdicts
    ):
        errors.append("accepted review kept no transition")
    kept_seqs = {
        verdict.seq for verdict in review.transition_verdicts if verdict.keep
    }
    retained = [
        transition for transition in transitions if transition.seq in kept_seqs
    ]
    retained_group_count = len(continuous_transition_groups(retained))
    if review.decision == "accept" and retained_group_count != 1:
        errors.append(
            "accepted review kept "
            f"{retained_group_count} disconnected transition groups"
        )
    return errors


def review_candidate(
    candidate: RecipeCandidate,
    critic: CriticFn | None = None,
    raise_on_error: bool = False,
) -> RecipeCandidateReview:
    transitions, _pruned = reviewable_candidate_transitions(candidate)
    if not transitions:
        return _fallback_review("autonomous_replay_candidate_missing")
    try:
        payload = build_candidate_review_payload(candidate)
        raw_review = (
            critic(payload)
            if critic is not None
            else _llm_review_candidate(
                payload,
                _candidate_screen_evidence(candidate, transitions),
            )
        )
        review = _coerce_review(raw_review)
        errors = _verdict_contract_errors(transitions, review)
        if errors:
            raise ValueError(
                "critic_transition_verdict_contract_failed: "
                + "; ".join(errors[:8])
            )
        return review
    except Exception as exc:
        if raise_on_error:
            raise
        return _fallback_review(f"critic_review_failed: {str(exc)[:200]}")


def review_and_apply_candidate(
    run_id: str,
    db_path=None,
    critic: CriticFn | None = None,
    compiler=None,
    mode: str = "review",
    raise_on_critic_error: bool = False,
) -> dict[str, Any]:
    from agent.recipe.candidate_store import RecipeCandidateStore

    candidate_store = RecipeCandidateStore(db_path)
    candidate = candidate_store.get_candidate(run_id)
    if candidate is None:
        return _fallback_review(f"candidate_not_found: {run_id}").model_dump(
            mode="json"
        )

    allow_promotion = _process_mode(mode) == "promote"
    review = review_candidate(
        candidate,
        critic=critic,
        raise_on_error=raise_on_critic_error,
    )
    promotion = RulePromotionResult(
        enabled=allow_promotion,
    )
    if allow_promotion and review.decision == "accept":
        retained, pruned = retained_candidate_path(candidate, review)
        try:
            if not retained:
                raise ValueError("critic kept no continuous source path")
            from agent.application.experience_rule_compiler import (
                compile_experience_rule,
            )

            rule = compile_experience_rule(
                candidate,
                retained,
                compiler=compiler,
            )
            saved_count = ExperienceRuleStore(db_path).save_rule(
                rule,
                source_run_id=candidate.run_id,
            )
            promotion = RulePromotionResult(
                enabled=True,
                promoted=saved_count > 0,
                saved_count=saved_count,
                rule_step_count=len(rule.steps),
                rule_action_count=sum(len(step.actions) for step in rule.steps),
                pruned_transitions=pruned,
            )
        except Exception as exc:
            promotion = RulePromotionResult(
                enabled=True,
                pruned_transitions=pruned,
            )
            review = review.model_copy(
                update={
                    "decision": "reject",
                    "reasons": [
                        *review.reasons,
                        f"experience_rule_compilation_failed: {str(exc)[:200]}",
                    ],
                }
            )

    validation = {
        "review": review.model_dump(mode="json"),
        "promotion": promotion.model_dump(mode="json"),
    }
    candidate_store.update_status(
        run_id,
        "accepted" if review.decision == "accept" else "rejected",
        validation=validation,
    )
    result = review.model_dump(mode="json")
    result["run_id"] = run_id
    result["promotion"] = promotion.model_dump(mode="json")
    return result


def _process_mode(mode: str | None) -> ReviewProcessMode:
    return "promote" if (mode or "review").strip().lower() == "promote" else "review"


__all__ = [
    "CriticFn",
    "RulePromotionResult",
    "build_candidate_review_payload",
    "review_and_apply_candidate",
    "review_candidate",
]

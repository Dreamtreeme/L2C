"""자율탐색 후보를 Critic으로 검토하고 승격 결과를 저장한다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

from agent.config import get_settings
from agent.recipe.candidate_promotion import (
    CandidatePromotionResult,
    apply_candidate_promotion,
)
from agent.recipe.promotion_policy import compact_step_evidence_verdicts
from agent.recipe.task_category import normalize_task_category
from agent.runtime.worker_actions import REVIEWABLE_REPLAY_ACTIONS
from agent.utils.text import normalize_text, url_template
from shared.schema.feedback_schema import (
    ExecutionEvent,
    RecipeCandidate,
    RecipeCandidateReview,
)
from shared.schema.recipe_schema import (
    ExperienceTransition,
    PhysicalAction,
    PhysicalActionName,
    ScreenCheckpoint,
)


CriticFn = Callable[[dict[str, Any]], dict[str, Any] | RecipeCandidateReview]
ReviewProcessMode = Literal["review", "promote"]


@dataclass(frozen=True, slots=True)
class ReviewableActionSpec:
    """Critic이 빠짐없이 판정해야 하는 실행 단계."""

    seq: int
    action: PhysicalActionName

    def to_payload(self) -> dict[str, int | str]:
        return {"seq": self.seq, "action": self.action}


def _critic_evidence_text_limit() -> int:
    return get_settings().recipe.critic_evidence_text_limit


def _reviewable_action_specs(
    steps: list[PhysicalAction],
) -> list[ReviewableActionSpec]:
    """도구 계약으로 재생 방식이 확정된 단계만 Critic 검토 대상으로 삼는다."""

    specs: list[ReviewableActionSpec] = []
    for step in steps:
        if (
            step.action not in REVIEWABLE_REPLAY_ACTIONS
            or step.replay_mode not in {"fixed", "parameterized"}
        ):
            continue
        specs.append(ReviewableActionSpec(seq=step.source_seq, action=step.action))
    return specs


def _without_empty_values(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item not in (None, "", [], {})
    }


def _compact_observation_id(value: str) -> str:
    observation_id = value
    marker = ":observation:"
    if marker in observation_id:
        return "observation:" + observation_id.rsplit(marker, 1)[-1]
    return observation_id


def _compact_checkpoint(
    checkpoint: ScreenCheckpoint | None,
) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    return _without_empty_values(
        {
            "observation_id": _compact_observation_id(checkpoint.observation_id),
            "url_template": url_template(checkpoint.url_template),
            "page_role": checkpoint.page_role,
        }
    )


def _compact_target(step: PhysicalAction) -> dict[str, Any]:
    if step.target is None:
        return {}
    target = step.target
    return _without_empty_values(
        {
            "text": target.text,
            "semantic_label": target.semantic_label,
            "marker_type": target.marker_type,
            "region": target.region,
            "center_ratio": target.center_ratio,
        }
    )


def _compact_step_param(step: PhysicalAction) -> dict[str, Any]:
    if step.action == "type_in_marker":
        return _without_empty_values({"slot_name": step.param.slot_name})
    if step.action == "press_key":
        return _without_empty_values({"key": step.param.key})
    return {}


def _compact_candidate_step(
    candidate: RecipeCandidate,
    step: PhysicalAction,
) -> dict[str, Any]:
    transition = candidate.transition_for_action(step.source_seq)
    roi = dict(step.roi_signature or {})
    return _without_empty_values(
        {
            "seq": step.source_seq,
            "action": step.action,
            "replay_mode": step.replay_mode,
            "before": _compact_checkpoint(transition.before if transition else None),
            "target": _compact_target(step),
            "roi": _without_empty_values(
                {
                    "available": bool(roi.get("phash")),
                    "algorithm": roi.get("algorithm"),
                    "crop_rect_ratio": roi.get("crop_rect_ratio"),
                    "target_center_ratio": roi.get("target_center_ratio"),
                }
            ),
            "param": _compact_step_param(step),
            "slot_refs": list(step.slot_refs),
            "expected_after": transition.expected_after if transition else "",
            "intent": transition.intent if transition else step.intent,
            "target_role": step.target_role,
            "component": step.component,
            "risk_level": step.risk_level,
        }
    )


def _normalized_marker_texts(values: list[Any]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_text(value)
        key = text.casefold().replace(" ", "")
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        texts.append(text)
    return texts


def _local_marker_context(
    values: list[Any],
    step: PhysicalAction | None,
) -> list[str]:
    """전체 OCR 대신 대상 주변의 짧은 화면 문맥만 남긴다."""

    texts = _normalized_marker_texts(values)
    limit = _critic_evidence_text_limit()
    if len(texts) <= limit:
        return texts
    target = step.target if step else None
    labels = {
        normalize_text(value).casefold().replace(" ", "")
        for value in (
            target.semantic_label if target else "",
            target.text if target else "",
        )
        if normalize_text(value)
    }
    index = next(
        (
            position
            for position, text in enumerate(texts)
            if text.casefold().replace(" ", "") in labels
        ),
        None,
    )
    if index is None:
        return texts[:limit]
    start = max(0, min(index - limit // 2, len(texts) - limit))
    return texts[start : start + limit]


def _compact_transition(
    transition: ExperienceTransition | None,
    *,
    include_text: bool,
) -> dict[str, Any]:
    if transition is None:
        return {}
    evidence = transition.evidence
    match = dict(evidence.after_state_match or {}) if evidence else {}
    item = {
        "status": evidence.status if evidence else "",
        "outcome": evidence.outcome if evidence else "",
        "reason": evidence.reason if evidence else "",
        "visual_change_ratio": evidence.visual_change_ratio if evidence else None,
        "after": _without_empty_values(
            {"page_role": transition.after.page_role}
        ),
    }
    if include_text and evidence:
        item.update(
            {
                "before_observation_id": _compact_observation_id(
                    transition.before.observation_id
                ),
                "after_observation_id": _compact_observation_id(
                    transition.after.observation_id
                ),
                "phash_distance": evidence.phash_distance,
                "transition_actions": list(evidence.transition_actions),
                "after_state_match": _without_empty_values(
                    {
                        "matched": match.get("matched"),
                        "reason": match.get("reason"),
                        "mode": match.get("mode"),
                        "distance": match.get("distance"),
                    }
                ),
                "after": _compact_checkpoint(transition.after),
                "after_text_sample": _normalized_marker_texts(
                    list(evidence.after_marker_texts)
                )[: _critic_evidence_text_limit()],
            }
        )
    return _without_empty_values(item)


def _compact_trajectory_event(
    event: ExecutionEvent,
    reviewable_seqs: set[int],
) -> dict[str, Any]:
    step = event.candidate_action
    result = dict(event.result or {})
    action = (
        step.action
        if step
        else str(result.get("action") or result.get("requested_action") or "")
    )
    is_candidate = event.seq in reviewable_seqs
    before_checkpoint = _compact_checkpoint(event.before_checkpoint)
    if not is_candidate:
        before_checkpoint = _without_empty_values(
            {"page_role": before_checkpoint.get("page_role")}
        )
    evidence = event.transition.evidence if event.transition else None
    item = {
        "seq": event.seq,
        "action": action,
        "candidate": is_candidate,
        "source": result.get("action_source")
        or (evidence.source if evidence else ""),
        "before": before_checkpoint,
        "result": _without_empty_values(
            {
                "status": result.get("status"),
                "reason": result.get("reason") or result.get("error"),
            }
        ),
        "transition": _compact_transition(
            event.transition,
            include_text=is_candidate,
        ),
    }
    if is_candidate:
        item["before_text_context"] = _local_marker_context(
            list(event.before_marker_texts),
            step,
        )
    return _without_empty_values(item)


def _compact_trajectory(
    candidate: RecipeCandidate,
    reviewable_seqs: set[int],
) -> list[dict[str, Any]]:
    """성공·실패 분기를 포함한 실행 순서를 짧은 인과 기록으로 만든다."""

    return [
        _compact_trajectory_event(event, reviewable_seqs)
        for event in candidate.action_events
    ]


def _compact_worker_execution(candidate: RecipeCandidate) -> dict[str, Any]:
    """Critic에 필요한 실행 결과만 남겨 반복된 전체 상태를 제거한다."""

    keys = (
        "run_status",
        "collected_count",
        "persisted_count",
    )
    execution = {
        key: getattr(candidate, key)
        for key in keys
        if getattr(candidate, key) not in (None, "", [], {})
    }
    return execution


def _coerce_review(
    raw: dict[str, Any] | RecipeCandidateReview,
) -> RecipeCandidateReview:
    return (
        raw if isinstance(raw, RecipeCandidateReview) else RecipeCandidateReview(**raw)
    )


def _fallback_review(reason: str) -> RecipeCandidateReview:
    return RecipeCandidateReview(
        decision="revise",
        reasons=[reason],
        feedback_to_worker="Candidate review could not be completed. Re-submit with clearer worker evidence.",
    )


def _serialize_candidate_review_payload(payload: dict[str, Any]) -> str:
    """구조를 유지하면서 전송에 불필요한 JSON 공백을 제거한다."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


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


def build_candidate_review_payload(
    candidate: RecipeCandidate,
) -> dict[str, Any]:
    """전체 실행 순서와 재생 후보의 인과 증거만 Critic에게 전달한다."""

    steps = candidate.steps
    required_step_verdicts = _reviewable_action_specs(steps)
    reviewable_seqs = {item.seq for item in required_step_verdicts}
    candidate_steps = [
        _compact_candidate_step(candidate, step)
        for step in steps
        if step.source_seq in reviewable_seqs
    ]
    return {
        "run_id": candidate.run_id,
        "status": candidate.status,
        "request": _compact_collection_request(candidate),
        "trajectory": _compact_trajectory(
            candidate,
            reviewable_seqs,
        ),
        "candidate_steps": candidate_steps,
        "deterministic_step_validation": compact_step_evidence_verdicts(candidate),
        "required_step_verdicts": [
            item.to_payload() for item in required_step_verdicts
        ],
        "worker_execution": _compact_worker_execution(candidate),
    }


def _llm_review_candidate(payload: dict[str, Any]) -> RecipeCandidateReview:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.llm.policy import commander_model_name
    from agent.prompts.trust_boundary import external_content_contract_en

    model_name = get_settings().models.recipe_critic_model or commander_model_name()
    from agent.llm.clients import get_structured_google_model

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
                + "\nYou are the Reflex Recipe Critic. The recorder preserved the ordered execution trajectory and "
                "derived candidate replay modes from executed tool contracts. You have pruning authority only. "
                "Return only RecipeCandidateReview and never rewrite or synthesize executable metadata. "
                "For every item in required_step_verdicts, return exactly one step_verdict with the same seq. "
                "Use trajectory to distinguish the successful route from abandoned, recovery, or variable-choice "
                "branches. candidate_steps contains the only actions eligible for promotion. "
                "Set keep=false for wrong targets, no-op actions, abandoned or recovery branches, unstable "
                "state-dependent choices, and steps whose expected result is not supported by the evidence. "
                "A recovery action is never a reusable task path even when it eventually reveals the expected "
                "site. If the recorded intent says the visible screen is wrong, unrelated, unexpected, or must "
                "be exited before the task can continue, set keep=false for that action. Evaluate later stable "
                "task actions independently. "
                "Keep an eligible step when its after observation is the before observation of a later kept "
                "step: that continuity is evidence that it prepared the next target or state. Do not call a "
                "step redundant merely because it did not reach the final page. Prune such a bridge only when "
                "the trajectory explicitly shows that it failed, was reverted, or belonged to an abandoned "
                "branch. "
                "A successful overall run does not make every step reusable. Evaluate deterministic_step_validation "
                "first; eligible=false must always be keep=false. When execution_group_seqs is present, evaluate "
                "the listed actions as one transition: a deferred_group_effect action is valid only because the "
                "final group member verified the saved after-state. Do not reject that action merely because it "
                "did not change the screen by itself. Preserve only steps that causally contributed to success and "
                "can safely reuse the recorded action. Pruning with keep=false does not by itself require "
                "decision=revise; return decision=accept when the kept subset still forms a safe causal path. Use "
                "decision=revise only when the recorded evidence or metadata cannot produce a valid path. "
                "Do not repair a bad step, change "
                "replay_mode, create input slots, create transition contracts, or replace an action. "
                "If critic_correction is present, return one complete verdict list matching the required seq values."
            )
        ),
        HumanMessage(content=_serialize_candidate_review_payload(payload)),
    ]
    from agent.observability.run_context import invoke_with_metrics

    return _coerce_review(invoke_with_metrics(llm, messages, "recipe_critic"))


def _step_verdict_contract_errors(
    required_steps: list[ReviewableActionSpec],
    review: RecipeCandidateReview,
) -> list[str]:
    """Critic이 후보를 추가하거나 빼지 않고 모두 판정했는지 검사한다."""

    if review.decision != "accept":
        return []

    by_seq: dict[int, int] = {}
    for verdict in review.step_verdicts:
        by_seq[verdict.seq] = by_seq.get(verdict.seq, 0) + 1

    errors: list[str] = []
    required_seqs = {required.seq for required in required_steps}
    for required in required_steps:
        count = by_seq.get(required.seq, 0)
        if count == 0:
            errors.append(f"missing seq={required.seq} action={required.action}")
            continue
        if count != 1:
            errors.append(f"duplicate seq={required.seq} count={count}")
    for seq in sorted(set(by_seq) - required_seqs):
        errors.append(f"unexpected seq={seq}")
    return errors


def review_candidate(
    candidate: RecipeCandidate,
    critic: CriticFn | None = None,
    raise_on_error: bool = False,
) -> RecipeCandidateReview:
    required_steps = _reviewable_action_specs(candidate.steps)
    payload = build_candidate_review_payload(candidate)
    if not required_steps:
        return RecipeCandidateReview(
            decision="reject",
            reasons=["autonomous_replay_candidate_missing"],
            feedback_to_worker=(
                "자율탐색 단계에 fixed 또는 parameterized 재사용 후보가 없습니다."
            ),
        )
    try:
        invoke_critic = critic or _llm_review_candidate
        review = _coerce_review(invoke_critic(payload))
        errors = _step_verdict_contract_errors(required_steps, review)
        if not errors:
            return review

        corrected_payload = dict(payload)
        corrected_payload["critic_correction"] = {
            "kind": "step_verdict_contract",
            "errors": errors,
            "instruction": (
                "Return exactly one step_verdict for every required seq. "
                "Use keep=false for steps that must not be replayed."
            ),
        }
        corrected_review = _coerce_review(invoke_critic(corrected_payload))
        corrected_errors = _step_verdict_contract_errors(
            required_steps,
            corrected_review,
        )
        if corrected_errors:
            return _fallback_review(
                "critic_step_verdict_contract_failed: "
                + "; ".join(corrected_errors[:8])
            )
        return corrected_review
    except Exception as exc:
        if raise_on_error:
            raise
        return _fallback_review(f"critic_review_failed: {str(exc)[:200]}")


def _status_for_review(review: RecipeCandidateReview) -> str:
    if review.decision == "accept":
        return "accepted"
    if review.decision == "reject":
        return "rejected"
    return "revise"


def review_and_apply_candidate(
    run_id: str,
    db_path=None,
    critic: CriticFn | None = None,
    mode: str = "review",
    raise_on_critic_error: bool = False,
) -> dict[str, Any]:
    from agent.recipe.candidate_store import RecipeCandidateStore

    candidate_store = RecipeCandidateStore(db_path)
    candidate = candidate_store.get_candidate(run_id)
    if not candidate:
        return _fallback_review(
            f"candidate_not_found: {run_id}"
        ).model_dump(mode="json")

    normalized_mode = _process_mode(mode)
    allow_promotion = normalized_mode == "promote"
    review = review_candidate(
        candidate,
        critic=critic,
        raise_on_error=raise_on_critic_error,
    )
    promotion = CandidatePromotionResult(
        enabled=allow_promotion,
        promoted=False,
        saved_count=0,
        promoted_action_count=0,
        promoted_transition_count=0,
        promoted_path_count=0,
    )
    if allow_promotion and review.decision == "accept":
        promotion = apply_candidate_promotion(
            candidate,
            review,
            db_path=db_path,
        )

    validation = {
        "review": review.model_dump(mode="json"),
        "promotion": promotion.model_dump(mode="json"),
    }
    candidate_store.update_status(
        run_id, _status_for_review(review), validation=validation
    )
    out = review.model_dump(mode="json")
    out["run_id"] = run_id
    out["promotion"] = promotion.model_dump(mode="json")
    return out


def _process_mode(mode: str | None) -> ReviewProcessMode:
    normalized = (mode or "review").strip().lower()
    return "promote" if normalized == "promote" else "review"

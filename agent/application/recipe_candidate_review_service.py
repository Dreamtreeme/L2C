"""자율탐색 후보를 Critic으로 검토하고 승격 결과를 저장한다."""

from __future__ import annotations

import json
from typing import Any, Callable

from agent.config import get_settings
from agent.recipe.candidate_promotion import apply_candidate_promotion
from agent.recipe.promotion_policy import compact_step_evidence_verdicts
from agent.runtime.worker_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
)
from agent.recipe.task_category import normalize_task_category
from shared.schema.feedback_schema import (
    RecipeCandidate,
    RecipeCandidateReview,
    RecordedRecipeStep,
    WorkerSubmission,
)


CriticFn = Callable[[dict[str, Any]], dict[str, Any] | RecipeCandidateReview]


def _critic_evidence_text_limit() -> int:
    return get_settings().recipe.critic_evidence_text_limit


def _reviewable_action_specs(
    steps: list[RecordedRecipeStep],
) -> list[dict[str, Any]]:
    """자율탐색이 재사용 후보로 명시한 단계만 Critic 검토 대상으로 삼는다."""

    specs: list[dict[str, Any]] = []
    for step in steps:
        if (
            step.action not in REVIEWABLE_REPLAY_ACTIONS
            or step.replay_mode not in {"fixed", "parameterized"}
            or step.seq is None
        ):
            continue
        specs.append({"seq": step.seq, "action": step.action})
    return specs


def _feedback_evidence_seqs(
    worker_submission: WorkerSubmission,
    steps: list[RecordedRecipeStep],
    reviewable_seqs: set[int],
) -> set[int]:
    """후속 행동에는 바로 앞 행동의 성공 문맥도 Critic 증거로 포함한다."""

    contextual_seqs = {
        int(step.seq)
        for step in steps
        if (
            step.action in CONTEXTUAL_REPLAY_ACTIONS
            and step.seq is not None
            and step.seq in reviewable_seqs
        )
    }
    episode_seqs = sorted(
        {episode.seq for episode in worker_submission.feedback_episodes}
    )
    out = set(reviewable_seqs)
    for seq in contextual_seqs:
        previous = [item for item in episode_seqs if item < seq]
        if previous:
            out.add(previous[-1])
    return out


def _compact_transition_records(
    worker_submission: WorkerSubmission,
    target_seqs: set[int],
) -> list[dict[str, Any]]:
    """승격 가능한 대상 행동의 전환 결과만 제한된 OCR 증거와 함께 남긴다."""

    text_limit = _critic_evidence_text_limit()
    evidence: list[dict[str, Any]] = []
    keys = (
        "action_seq",
        "action",
        "expected_after",
        "source",
        "attempt",
        "elapsed_sec",
        "status",
        "outcome",
        "reason",
        "phash_distance",
        "ocr_skipped",
        "marker_count",
    )
    for observation in worker_submission.transition_records:
        seq = observation.action_seq
        if seq is None:
            continue
        if seq not in target_seqs:
            continue
        observation_data = observation.model_dump(mode="json")
        item = {
            key: observation_data.get(key)
            for key in keys
            if observation_data.get(key) not in (None, "", [], {})
        }
        item["marker_texts"] = observation.marker_texts[:text_limit]
        evidence.append(item)
    return evidence


def _compact_worker_execution(worker_submission: WorkerSubmission) -> dict[str, Any]:
    """Critic에 필요한 실행 결과만 남겨 반복된 전체 상태를 제거한다."""

    keys = (
        "run_status",
        "collected_count",
        "persisted_count",
    )
    execution = {
        key: getattr(worker_submission, key)
        for key in keys
        if getattr(worker_submission, key) not in (None, "", [], {})
    }
    execution["extracted_summary"] = dict(worker_submission.extracted_summary)
    return execution


def _compact_feedback_evidence(
    worker_submission: WorkerSubmission,
    target_seqs: set[int],
) -> list[dict[str, Any]]:
    """행동별 이전 화면과 실행 결과를 Critic이 비교할 수 있게 축약한다."""

    text_limit = _critic_evidence_text_limit()
    evidence: list[dict[str, Any]] = []
    for episode in worker_submission.feedback_episodes:
        seq = episode.seq
        if seq not in target_seqs:
            continue
        proposal = episode.proposal
        observation = episode.observation
        before = observation.before
        after = observation.after
        result = observation.result
        feedback = episode.feedback
        item = {
            "seq": seq,
            "action": proposal.action or result.get("action") or "",
            "expected_after": proposal.args.get("expected_after") or "",
            "before_url": before.get("url") or "",
            "after_url": after.get("url") or "",
            "screen_changed": bool(after.get("screen_changed", False)),
            "before_marker_texts": list(before.get("marker_texts", []) or [])[
                :text_limit
            ],
            "result_status": result.get("status") or "",
            "result_reason": result.get("reason") or "",
            "feedback_label": feedback.label,
            "feedback_reason": feedback.reason,
        }
        evidence.append(
            {
                key: value
                for key, value in item.items()
                if value not in (None, "", [], {})
            }
        )
    return evidence


def _coerce_review(raw: dict[str, Any] | RecipeCandidateReview) -> dict[str, Any]:
    review = (
        raw if isinstance(raw, RecipeCandidateReview) else RecipeCandidateReview(**raw)
    )
    return review.model_dump(mode="json")


def _fallback_review(reason: str) -> dict[str, Any]:
    return RecipeCandidateReview(
        decision="revise",
        reasons=[reason],
        feedback_to_worker="Candidate review could not be completed. Re-submit with clearer worker evidence.",
    ).model_dump(mode="json")


def _serialize_candidate_review_payload(payload: dict[str, Any]) -> str:
    """구조를 유지하면서 전송에 불필요한 JSON 공백을 제거한다."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_candidate_review_payload(
    candidate: RecipeCandidate,
) -> dict[str, Any]:
    """후보 증거(candidate evidence)를 의미 판단 없이 비평가(Critic)에게 전달한다."""
    worker_submission = candidate.submission
    steps = candidate.steps
    reviewable_seqs = {item["seq"] for item in _reviewable_action_specs(steps)}
    evidence_seqs = _feedback_evidence_seqs(
        worker_submission,
        steps,
        reviewable_seqs,
    )
    required_step_verdicts = _reviewable_action_specs(steps)
    task_category = normalize_task_category(
        candidate.submission.collection_intent.task_category
    )
    return {
        "run_id": candidate.run_id,
        "status": candidate.status,
        "site": candidate.site,
        "task_category": task_category,
        "goal": candidate.goal,
        "keyword": candidate.keyword,
        "steps": [step.model_dump(mode="json") for step in steps],
        "transition_records": _compact_transition_records(
            worker_submission,
            reviewable_seqs,
        ),
        "feedback_evidence": _compact_feedback_evidence(
            worker_submission,
            evidence_seqs,
        ),
        "deterministic_step_validation": compact_step_evidence_verdicts(candidate),
        "required_step_verdicts": required_step_verdicts,
        "worker_execution": _compact_worker_execution(worker_submission),
    }


def _llm_review_candidate(payload: dict[str, Any]) -> dict[str, Any]:
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
                + "\nYou are the Reflex Recipe Critic. Autonomous exploration already chose every action, argument, "
                "input slot, target, page context, and replay_mode. You have pruning authority only. "
                "Return only RecipeCandidateReview and never rewrite or synthesize executable metadata. "
                "For every item in required_step_verdicts, return exactly one step_verdict with the same seq. "
                "Set keep=false for wrong targets, no-op actions, abandoned or recovery branches, unstable "
                "state-dependent choices, and steps whose expected result is not supported by the evidence. "
                "A successful overall run does not make every step reusable. Evaluate deterministic_step_validation "
                "first; eligible=false must always be keep=false. When execution_group_seqs is present, evaluate "
                "the listed actions as one transition: a deferred_group_effect action is valid only because the "
                "final group member verified the saved after-state. Do not reject that action merely because it "
                "did not change the screen by itself. Preserve only steps that causally contributed to success and "
                "can safely reuse the autonomous replay proposal. Pruning with keep=false does not by itself require "
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
    payload: dict[str, Any],
    review: dict[str, Any],
) -> list[str]:
    """Critic이 후보를 추가하거나 빼지 않고 모두 판정했는지 검사한다."""

    if review.get("decision") != "accept":
        return []

    items = list(review.get("step_verdicts") or [])
    by_seq: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            seq = int(item.get("seq"))
        except (TypeError, ValueError):
            continue
        by_seq.setdefault(seq, []).append(item)

    errors: list[str] = []
    required_seqs = {
        int(required["seq"]) for required in payload.get("required_step_verdicts") or []
    }
    for required in payload.get("required_step_verdicts") or []:
        seq = int(required["seq"])
        matches = by_seq.get(seq, [])
        if not matches:
            errors.append(f"missing seq={seq} action={required.get('action') or ''}")
            continue
        if len(matches) != 1:
            errors.append(f"duplicate seq={seq} count={len(matches)}")
    for seq in sorted(set(by_seq) - required_seqs):
        errors.append(f"unexpected seq={seq}")
    return errors


def review_candidate(
    candidate: RecipeCandidate,
    critic: CriticFn | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    payload = build_candidate_review_payload(candidate)
    if not payload.get("required_step_verdicts"):
        return RecipeCandidateReview(
            decision="reject",
            reasons=["autonomous_replay_candidate_missing"],
            feedback_to_worker=(
                "자율탐색 단계에 fixed 또는 parameterized 재사용 후보가 없습니다."
            ),
        ).model_dump(mode="json")
    try:
        invoke_critic = critic or _llm_review_candidate
        review = _coerce_review(invoke_critic(payload))
        errors = _step_verdict_contract_errors(payload, review)
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
            corrected_payload,
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


def _status_for_review(review: dict[str, Any]) -> str:
    decision = review.get("decision")
    if decision == "accept":
        return "accepted"
    if decision == "reject":
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
        return _fallback_review(f"candidate_not_found: {run_id}")

    normalized_mode = _process_mode(mode)
    allow_promotion = normalized_mode == "promote"
    review = review_candidate(
        candidate,
        critic=critic,
        raise_on_error=raise_on_critic_error,
    )
    promotion = {
        "enabled": allow_promotion,
        "promoted": False,
        "saved_count": 0,
        "promoted_action_count": 0,
        "promoted_transition_count": 0,
        "skipped_steps": [],
    }
    if allow_promotion and review.get("decision") == "accept":
        promotion = apply_candidate_promotion(
            candidate,
            review,
            db_path=db_path,
        )

    validation = {
        "review": review,
        "promotion": promotion,
    }
    candidate_store.update_status(
        run_id, _status_for_review(review), validation=validation
    )
    out = dict(review)
    out["run_id"] = run_id
    out["promotion"] = promotion
    return out


def _process_mode(mode: str | None) -> str:
    normalized = (mode or "review").strip().lower()
    return normalized if normalized in {"review", "promote"} else "review"

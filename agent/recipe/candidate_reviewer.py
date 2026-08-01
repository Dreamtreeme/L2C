"""자율탐색이 제안한 Reflex 단계를 제거만 하는 Critic 게이트."""

from __future__ import annotations

import json
from typing import Any, Callable

from agent.config import get_settings
from agent.recipe.candidate_promotion import apply_candidate_promotion
from agent.recipe.promotion_policy import compact_step_evidence_verdicts
from agent.recipe.replay_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    REVIEWABLE_REPLAY_ACTIONS,
)
from agent.recipe.task_category import task_category_from_candidate
from agent.utils.model_dump import dump_model
from shared.schema.feedback_schema import RecipeCandidateReview


CriticFn = Callable[[dict[str, Any]], dict[str, Any] | RecipeCandidateReview]


def _critic_evidence_text_limit() -> int:
    return get_settings().recipe.critic_evidence_text_limit


def _reviewable_action_specs(
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """자율탐색이 재사용 후보로 명시한 단계만 Critic 검토 대상으로 삼는다."""

    specs: list[dict[str, Any]] = []
    for step in steps or []:
        if (
            not isinstance(step, dict)
            or step.get("action") not in REVIEWABLE_REPLAY_ACTIONS
            or step.get("replay_mode") not in {"fixed", "parameterized"}
        ):
            continue
        try:
            seq = int(step.get("seq"))
        except (TypeError, ValueError):
            continue
        specs.append({"seq": seq, "action": str(step.get("action") or "")})
    return specs


def _reviewable_action_seqs(steps: list[dict[str, Any]]) -> set[int]:
    return {item["seq"] for item in _reviewable_action_specs(steps)}


def _feedback_evidence_seqs(
    worker_submission: dict[str, Any],
    steps: list[dict[str, Any]],
    reviewable_seqs: set[int],
) -> set[int]:
    """후속 행동에는 바로 앞 행동의 성공 문맥도 Critic 증거로 포함한다."""

    contextual_seqs = {
        int(step["seq"])
        for step in steps
        if (
            isinstance(step, dict)
            and step.get("action") in CONTEXTUAL_REPLAY_ACTIONS
            and str(step.get("seq", "")).lstrip("-").isdigit()
            and int(step["seq"]) in reviewable_seqs
        )
    }
    episode_seqs = sorted(
        {
            int(episode["seq"])
            for episode in worker_submission.get("feedback_episodes", []) or []
            if (
                isinstance(episode, dict)
                and str(episode.get("seq", "")).lstrip("-").isdigit()
            )
        }
    )
    out = set(reviewable_seqs)
    for seq in contextual_seqs:
        previous = [item for item in episode_seqs if item < seq]
        if previous:
            out.add(previous[-1])
    return out


def _compact_transition_records(
    worker_submission: dict[str, Any],
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
    for observation in worker_submission.get("transition_records", []) or []:
        if not isinstance(observation, dict):
            continue
        try:
            seq = int(observation.get("action_seq"))
        except (TypeError, ValueError):
            continue
        if seq not in target_seqs:
            continue
        item = {
            key: observation.get(key)
            for key in keys
            if observation.get(key) not in (None, "", [], {})
        }
        item["marker_texts"] = list(observation.get("marker_texts", []) or [])[:text_limit]
        evidence.append(item)
    return evidence


def _compact_worker_execution(worker_submission: dict[str, Any]) -> dict[str, Any]:
    """Critic에 필요한 실행 결과만 남겨 반복된 전체 상태를 제거한다."""

    keys = (
        "run_status",
        "is_finished",
        "hit_recursion_limit",
        "collected_count",
        "target_count",
        "persisted_count",
        "worker_notes",
    )
    execution = {
        key: worker_submission.get(key)
        for key in keys
        if worker_submission.get(key) not in (None, "", [], {})
    }
    execution["extracted_summary"] = dict(worker_submission.get("extracted_summary", {}) or {})
    return execution


def _compact_feedback_evidence(
    worker_submission: dict[str, Any],
    target_seqs: set[int],
) -> list[dict[str, Any]]:
    """행동별 이전 화면과 실행 결과를 Critic이 비교할 수 있게 축약한다."""

    text_limit = _critic_evidence_text_limit()
    evidence: list[dict[str, Any]] = []
    for episode in worker_submission.get("feedback_episodes", []) or []:
        if not isinstance(episode, dict):
            continue
        try:
            seq = int(episode.get("seq"))
        except (TypeError, ValueError):
            continue
        if seq not in target_seqs:
            continue
        proposal = episode.get("proposal") if isinstance(episode.get("proposal"), dict) else {}
        observation = episode.get("observation") if isinstance(episode.get("observation"), dict) else {}
        before = observation.get("before") if isinstance(observation.get("before"), dict) else {}
        after = observation.get("after") if isinstance(observation.get("after"), dict) else {}
        result = observation.get("result") if isinstance(observation.get("result"), dict) else {}
        feedback = episode.get("feedback") if isinstance(episode.get("feedback"), dict) else {}
        item = {
            "seq": seq,
            "action": proposal.get("action") or result.get("action") or "",
            "expected_after": proposal.get("expected_after") or "",
            "before_url": before.get("url") or "",
            "after_url": after.get("url") or "",
            "screen_changed": bool(after.get("screen_changed", False)),
            "before_marker_texts": list(before.get("marker_texts", []) or [])[:text_limit],
            "result_status": result.get("status") or "",
            "result_reason": result.get("reason") or "",
            "feedback_label": feedback.get("label") or "",
            "feedback_reason": feedback.get("reason") or "",
        }
        evidence.append({key: value for key, value in item.items() if value not in (None, "", [], {})})
    return evidence


def _coerce_review(raw: dict[str, Any] | RecipeCandidateReview) -> dict[str, Any]:
    if isinstance(raw, RecipeCandidateReview):
        return dump_model(raw)
    return dump_model(RecipeCandidateReview(**(raw or {})))


def _fallback_review(reason: str) -> dict[str, Any]:
    return dump_model(
        RecipeCandidateReview(
            decision="revise",
            reasons=[reason],
            feedback_to_worker="Candidate review could not be completed. Re-submit with clearer worker evidence.",
            confidence=0.0,
        )
    )


def _serialize_candidate_review_payload(payload: dict[str, Any]) -> str:
    """구조를 유지하면서 전송에 불필요한 JSON 공백을 제거한다."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_candidate_review_payload(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """후보 증거(candidate evidence)를 의미 판단 없이 비평가(Critic)에게 전달한다."""
    worker_submission = dict(candidate.get("payload", {}) or {})
    steps = [step for step in candidate.get("steps", []) or [] if isinstance(step, dict)]
    reviewable_seqs = _reviewable_action_seqs(steps)
    evidence_seqs = _feedback_evidence_seqs(
        worker_submission,
        steps,
        reviewable_seqs,
    )
    required_step_verdicts = _reviewable_action_specs(steps)
    task_category = task_category_from_candidate(candidate)
    return {
        "candidate_id": candidate.get("candidate_id", "") or "",
        "status": candidate.get("status", "") or "",
        "site": candidate.get("site", "") or "",
        "task_category": task_category,
        "goal": candidate.get("goal", "") or "",
        "keyword": candidate.get("keyword", "") or "",
        "steps": steps,
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
        "skill_metadata_evidence": dict(worker_submission.get("skill_metadata_evidence", {}) or {}),
        "worker_execution": _compact_worker_execution(worker_submission),
        "commander_review": dict(candidate.get("review", {}) or {}),
    }


def _llm_review_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.application.model_policy import commander_model_name
    from agent.prompts.trust_boundary import external_content_contract_en

    model_name = commander_model_name("VISION_RECIPE_CRITIC_MODEL")
    from agent.application.model_clients import get_structured_google_model

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
    from agent.application.run_context import invoke_with_metrics

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
        int(required["seq"])
        for required in payload.get("required_step_verdicts") or []
    }
    for required in payload.get("required_step_verdicts") or []:
        seq = int(required["seq"])
        matches = by_seq.get(seq, [])
        if not matches:
            errors.append(
                f"missing seq={seq} action={required.get('action') or ''}"
            )
            continue
        if len(matches) != 1:
            errors.append(f"duplicate seq={seq} count={len(matches)}")
    for seq in sorted(set(by_seq) - required_seqs):
        errors.append(f"unexpected seq={seq}")
    return errors


def review_candidate(
    candidate: dict[str, Any],
    critic: CriticFn | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    payload = build_candidate_review_payload(candidate)
    if not payload.get("required_step_verdicts"):
        return dump_model(
            RecipeCandidateReview(
                decision="reject",
                reasons=["autonomous_replay_candidate_missing"],
                feedback_to_worker=(
                    "자율탐색 단계에 fixed 또는 parameterized 재사용 후보가 없습니다."
                ),
                confidence=1.0,
            )
        )
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
    candidate_id: str,
    db_path=None,
    critic: CriticFn | None = None,
    mode: str = "review",
    raise_on_critic_error: bool = False,
) -> dict[str, Any]:
    from agent.recipe.candidate_store import RecipeCandidateStore

    candidate_store = RecipeCandidateStore(db_path)
    candidate = candidate_store.get_candidate(candidate_id)
    if not candidate:
        return _fallback_review(f"candidate_not_found: {candidate_id}")

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
    candidate_store.update_status(candidate_id, _status_for_review(review), validation=validation)
    out = dict(review)
    out["candidate_id"] = candidate_id
    out["promotion"] = promotion
    return out


def _process_mode(mode: str | None) -> str:
    normalized = (mode or "review").strip().lower()
    return normalized if normalized in {"review", "promote"} else "review"

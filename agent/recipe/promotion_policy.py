"""활성 Reflex 승격 전에 명백한 실행 결함을 차단한다."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


_TARGET_ACTIONS = {"click_marker", "type_in_marker"}
_BLOCKING_FEEDBACK_LABELS = {"wrong_target", "no_effect", "loop_risk", "error"}
_BLOCKING_RESULT_STATUSES = {"error", "skipped"}
_BLOCKING_TRANSITION_REASONS = {
    "no_screen_change",
    "reflex_no_screen_change",
    "transition_timeout",
}
_CODE_MANAGED_TRANSITION_SOURCES = {
    "page_policy": "managed_by_page_policy",
    "card_queue": "managed_by_card_queue",
    "duplicate_job_policy": "managed_by_duplicate_policy",
    "screen_policy": "managed_by_screen_policy",
    "reflex": "already_managed_by_reflex",
}


def _seq(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def evaluate_candidate_step_evidence(candidate: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """대상 행동별 증거를 검사해 코드가 확정할 수 있는 차단 사유를 반환한다."""

    payload = dict(candidate.get("payload", {}) or {})
    feedback_by_seq: dict[int, list[dict[str, Any]]] = defaultdict(list)
    transitions_by_seq: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for episode in payload.get("feedback_episodes", []) or []:
        if not isinstance(episode, dict):
            continue
        seq = _seq(episode.get("seq"))
        if seq is not None:
            feedback_by_seq[seq].append(episode)

    for observation in payload.get("transition_observations", []) or []:
        if not isinstance(observation, dict):
            continue
        seq = _seq(observation.get("action_seq"))
        if seq is not None:
            transitions_by_seq[seq].append(observation)

    verdicts: dict[int, dict[str, Any]] = {}
    for step in candidate.get("steps", []) or []:
        if not isinstance(step, dict) or step.get("action") not in _TARGET_ACTIONS:
            continue
        seq = _seq(step.get("seq"))
        if seq is None:
            continue

        feedback_items = feedback_by_seq.get(seq, [])
        transition_items = transitions_by_seq.get(seq, [])
        feedback_labels: list[str] = []
        transition_sources: list[str] = []
        reasons: list[str] = []

        for episode in feedback_items:
            feedback = episode.get("feedback") if isinstance(episode.get("feedback"), dict) else {}
            observation = episode.get("observation") if isinstance(episode.get("observation"), dict) else {}
            result = observation.get("result") if isinstance(observation.get("result"), dict) else {}
            label = str(feedback.get("label") or "").strip()
            result_status = str(result.get("status") or "").strip()
            if label:
                feedback_labels.append(label)
            if label in _BLOCKING_FEEDBACK_LABELS:
                reasons.append(f"feedback_{label}")
            if result_status in _BLOCKING_RESULT_STATUSES:
                reasons.append(f"action_{result_status}")

        for observation in transition_items:
            source = str(observation.get("source") or "").strip()
            reason = str(observation.get("reason") or "").strip()
            if source:
                transition_sources.append(source)
            managed_reason = _CODE_MANAGED_TRANSITION_SOURCES.get(source)
            if managed_reason:
                reasons.append(managed_reason)
            if reason in _BLOCKING_TRANSITION_REASONS:
                reasons.append(reason)

        if not feedback_items and not transition_items:
            reasons.append("action_evidence_missing")

        verdicts[seq] = {
            "seq": seq,
            "action": str(step.get("action") or ""),
            "eligible": not reasons,
            "blocking_reasons": list(dict.fromkeys(reasons)),
            "feedback_labels": list(dict.fromkeys(feedback_labels)),
            "transition_sources": list(dict.fromkeys(transition_sources)),
        }
    return verdicts


def compact_step_evidence_verdicts(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Critic 입력과 검증 로그에 사용할 정렬된 판정 목록을 만든다."""

    verdicts = evaluate_candidate_step_evidence(candidate)
    return [verdicts[seq] for seq in sorted(verdicts)]


__all__ = ["compact_step_evidence_verdicts", "evaluate_candidate_step_evidence"]

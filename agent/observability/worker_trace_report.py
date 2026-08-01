"""작업자 제출물의 캡처, 판단, 행동, 전환 경로를 조립한다."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _sequence(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _capture_suffix(capture_id: str) -> str:
    if not capture_id:
        return "캡처 없음"
    marker = ":capture:"
    if marker not in capture_id:
        return capture_id
    return f"capture:{capture_id.rsplit(marker, 1)[1]}"


def _text_preview(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _target_summary(
    recorded_step: dict[str, Any],
    proposal: dict[str, Any],
) -> dict[str, str]:
    recorded_target = _as_dict(recorded_step.get("target"))
    proposed_target = _as_dict(proposal.get("target"))
    label = str(
        recorded_target.get("semantic_label")
        or proposal.get("target_label")
        or proposed_target.get("target_label")
        or ""
    )
    text = str(recorded_target.get("text") or proposed_target.get("text") or "")
    return {"label": label, "text": text}


def _step_value(
    recorded_step: dict[str, Any],
    proposal: dict[str, Any],
) -> Any:
    if recorded_step.get("value") is not None:
        return recorded_step.get("value")
    args = _as_dict(proposal.get("args"))
    for key in ("text", "key", "direction"):
        if args.get(key) is not None:
            return args[key]
    return None


def _latest_transition(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    if not observations:
        return {}
    return max(
        enumerate(observations),
        key=lambda item: (_sequence(item[1].get("attempt")) or 0, item[0]),
    )[1]


def build_worker_trace(submission: dict[str, Any]) -> dict[str, Any]:
    """저장된 작업자 제출물을 행동 순번 기준 실행 경로로 변환한다."""

    payload = _as_dict(submission.get("payload")) or submission
    recorded_by_seq: dict[int, dict[str, Any]] = {}
    feedback_by_seq: dict[int, dict[str, Any]] = {}
    transitions_by_seq: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ignored_records = 0

    for item in payload.get("recorded_steps", []) or []:
        step = _as_dict(item)
        seq = _sequence(step.get("seq"))
        if seq is None:
            ignored_records += 1
            continue
        recorded_by_seq[seq] = step

    for item in payload.get("feedback_episodes", []) or []:
        episode = _as_dict(item)
        seq = _sequence(episode.get("seq"))
        if seq is None:
            ignored_records += 1
            continue
        feedback_by_seq[seq] = episode

    for item in payload.get("transition_records", []) or []:
        observation = _as_dict(item)
        seq = _sequence(observation.get("action_seq"))
        if seq is None:
            ignored_records += 1
            continue
        transitions_by_seq[seq].append(observation)

    all_sequences = sorted(
        set(recorded_by_seq) | set(feedback_by_seq) | set(transitions_by_seq)
    )
    steps: list[dict[str, Any]] = []
    capture_ids: set[str] = set()
    capture_mismatches = 0

    for seq in all_sequences:
        recorded = recorded_by_seq.get(seq, {})
        episode = feedback_by_seq.get(seq, {})
        proposal = _as_dict(episode.get("proposal"))
        feedback = _as_dict(episode.get("feedback"))
        before = _as_dict(_as_dict(episode.get("observation")).get("before"))
        result = _as_dict(_as_dict(episode.get("observation")).get("result"))
        transition_attempts = transitions_by_seq.get(seq, [])
        transition = _latest_transition(transition_attempts)

        decision_capture_id = str(
            recorded.get("decision_capture_id")
            or before.get("capture_id")
            or transition.get("from_capture_id")
            or ""
        )
        from_capture_id = str(
            transition.get("from_capture_id") or decision_capture_id
        )
        to_capture_id = str(transition.get("to_capture_id") or "")
        for capture_id in (
            decision_capture_id,
            from_capture_id,
            to_capture_id,
        ):
            if capture_id:
                capture_ids.add(capture_id)

        capture_consistent = not (
            decision_capture_id
            and transition.get("from_capture_id")
            and decision_capture_id != from_capture_id
        )
        if not capture_consistent:
            capture_mismatches += 1

        action = str(
            recorded.get("action")
            or proposal.get("action")
            or transition.get("action")
            or ""
        )
        target = _target_summary(recorded, proposal)
        steps.append(
            {
                "seq": seq,
                "action": action,
                "action_source": str(result.get("action_source") or ""),
                "decision_capture_id": decision_capture_id,
                "from_capture_id": from_capture_id,
                "to_capture_id": to_capture_id,
                "capture_consistent": capture_consistent,
                "target_label": target["label"],
                "target_text": target["text"],
                "value": _step_value(recorded, proposal),
                "intent": str(recorded.get("intent") or proposal.get("reason") or ""),
                "feedback_label": str(feedback.get("label") or ""),
                "feedback_reason": str(feedback.get("reason") or ""),
                "transition_status": str(transition.get("status") or ""),
                "transition_outcome": str(transition.get("outcome") or ""),
                "transition_reason": str(transition.get("reason") or ""),
                "transition_attempt": _sequence(transition.get("attempt")),
                "transition_attempt_count": len(transition_attempts),
                "elapsed_sec": transition.get("elapsed_sec"),
                "before_screenshot": str(before.get("screenshot") or ""),
                "after_screenshot": str(transition.get("screenshot") or ""),
                "recorded_for_recipe": bool(recorded),
            }
        )

    return {
        "submission_id": str(submission.get("submission_id") or ""),
        "run_id": str(payload.get("run_id") or submission.get("run_id") or ""),
        "site": str(payload.get("site") or submission.get("site") or ""),
        "goal": str(payload.get("goal") or submission.get("goal") or ""),
        "run_status": str(
            payload.get("run_status") or submission.get("run_status") or ""
        ),
        "review_decision": str(submission.get("review_decision") or ""),
        "step_count": len(steps),
        "recorded_action_count": len(recorded_by_seq),
        "transition_count": sum(len(items) for items in transitions_by_seq.values()),
        "capture_count": len(capture_ids),
        "capture_mismatch_count": capture_mismatches,
        "ignored_record_count": ignored_records,
        "steps": steps,
    }


def render_worker_trace(trace: dict[str, Any]) -> str:
    """실행 경로를 터미널에서 읽기 쉬운 한국어 텍스트로 표시한다."""

    lines = [
        f"실행 ID: {trace.get('run_id') or '-'}",
        f"제출물 ID: {trace.get('submission_id') or '-'}",
        f"사이트: {trace.get('site') or '-'}",
        (
            f"상태: {trace.get('run_status') or '-'}"
            f" / 검토 {trace.get('review_decision') or '-'}"
        ),
        (
            f"행동: {trace.get('step_count', 0)}개"
            f" / 레시피 기록 {trace.get('recorded_action_count', 0)}개"
            f" / 전환 관찰 {trace.get('transition_count', 0)}개"
            f" / 캡처 {trace.get('capture_count', 0)}개"
        ),
    ]
    if trace.get("goal"):
        lines.append(f"목표: {_text_preview(trace['goal'])}")
    lines.append("")

    for step in trace.get("steps", []) or []:
        target = step.get("target_label") or step.get("target_text") or ""
        action = str(step.get("action") or "알 수 없는 행동")
        action_label = f"{action} [{target}]" if target else action
        source_capture = _capture_suffix(
            str(step.get("from_capture_id") or step.get("decision_capture_id") or "")
        )
        target_capture = (
            _capture_suffix(str(step.get("to_capture_id") or ""))
            if step.get("to_capture_id")
            else "전환 관찰 없음"
        )
        lines.append(
            f"[{int(step.get('seq') or 0):04d}] "
            f"{source_capture} -- {action_label} --> {target_capture}"
        )
        details = []
        if step.get("feedback_label"):
            details.append(f"피드백={step['feedback_label']}")
        if step.get("transition_status"):
            details.append(f"전환={step['transition_status']}")
        if step.get("transition_reason"):
            details.append(f"근거={step['transition_reason']}")
        if not step.get("capture_consistent", True):
            details.append("캡처 연결 불일치")
        if details:
            lines.append(f"       {' / '.join(details)}")

    if trace.get("ignored_record_count"):
        lines.append("")
        lines.append(
            f"순번이 없어 제외된 기록: {trace['ignored_record_count']}개"
        )
    return "\n".join(lines)


__all__ = ["build_worker_trace", "render_worker_trace"]

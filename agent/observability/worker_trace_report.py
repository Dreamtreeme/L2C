"""작업자 제출물의 관찰, 행동, 전환 경로를 조립한다."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from shared.schema.feedback_schema import (
    FeedbackEpisode,
    RecordedRecipeStep,
    RecordedTransition,
    StoredWorkerSubmission,
)


def _observation_suffix(observation_id: str) -> str:
    if not observation_id:
        return "관찰 없음"
    marker = ":observation:"
    if marker not in observation_id:
        return observation_id
    return f"observation:{observation_id.rsplit(marker, 1)[1]}"


def _text_preview(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _target_summary(
    recorded_step: RecordedRecipeStep | None,
    episode: FeedbackEpisode | None,
) -> dict[str, str]:
    recorded_target = (
        recorded_step.target if recorded_step and recorded_step.target else {}
    )
    proposal = episode.proposal if episode else None
    proposed_target = proposal.target if proposal and proposal.target else {}
    label = str(
        recorded_target.get("semantic_label")
        or (proposal.target_label if proposal else "")
        or proposed_target.get("target_label")
        or ""
    )
    text = str(recorded_target.get("text") or proposed_target.get("text") or "")
    return {"label": label, "text": text}


def _step_value(
    recorded_step: RecordedRecipeStep | None,
    episode: FeedbackEpisode | None,
) -> Any:
    if recorded_step and recorded_step.value is not None:
        return recorded_step.value
    args = episode.proposal.args if episode else {}
    for key in ("text", "key", "direction"):
        if args.get(key) is not None:
            return args[key]
    return None


def _latest_transition(
    observations: list[RecordedTransition],
) -> RecordedTransition | None:
    if not observations:
        return None
    return max(enumerate(observations), key=lambda item: (item[1].attempt, item[0]))[1]


def build_worker_trace(submission: StoredWorkerSubmission) -> dict[str, Any]:
    """저장된 작업자 제출물을 행동 순번 기준 실행 경로로 변환한다."""

    payload = submission.payload
    recorded_by_seq: dict[int, RecordedRecipeStep] = {}
    feedback_by_seq: dict[int, FeedbackEpisode] = {}
    transitions_by_seq: dict[int, list[RecordedTransition]] = defaultdict(list)
    ignored_records = 0

    for step in payload.recorded_steps:
        if step.seq is None:
            ignored_records += 1
        else:
            recorded_by_seq[step.seq] = step
    for episode in payload.feedback_episodes:
        feedback_by_seq[episode.seq] = episode
    for transition in payload.transition_records:
        if transition.action_seq is None:
            ignored_records += 1
        else:
            transitions_by_seq[transition.action_seq].append(transition)

    sequences = sorted(
        set(recorded_by_seq) | set(feedback_by_seq) | set(transitions_by_seq)
    )
    steps: list[dict[str, Any]] = []
    observation_ids: set[str] = set()

    for seq in sequences:
        recorded = recorded_by_seq.get(seq)
        episode = feedback_by_seq.get(seq)
        transition_attempts = transitions_by_seq.get(seq, [])
        transition = _latest_transition(transition_attempts)
        before = episode.observation.before if episode else {}
        result = episode.observation.result if episode else {}

        before_observation_id = str(
            (transition.before_observation_id if transition else "")
            or (
                recorded.before_state.get("observation_id")
                if recorded
                else ""
            )
            or before.get("observation_id")
            or ""
        )
        after_observation_id = str(
            transition.after_observation_id if transition else ""
        )
        observation_ids.update(
            observation_id
            for observation_id in (
                before_observation_id,
                after_observation_id,
            )
            if observation_id
        )

        proposal = episode.proposal if episode else None
        feedback = episode.feedback if episode else None
        action = str(
            (recorded.action if recorded else "")
            or (proposal.action if proposal else "")
            or (transition.action if transition else "")
            or ""
        )
        target = _target_summary(recorded, episode)
        steps.append(
            {
                "seq": seq,
                "action": action,
                "action_source": str(result.get("action_source") or ""),
                "before_observation_id": before_observation_id,
                "after_observation_id": after_observation_id,
                "target_label": target["label"],
                "target_text": target["text"],
                "value": _step_value(recorded, episode),
                "intent": str(
                    (recorded.intent if recorded else "")
                    or (proposal.reason if proposal else "")
                    or ""
                ),
                "feedback_label": feedback.label if feedback else "",
                "feedback_reason": feedback.reason if feedback else "",
                "transition_status": transition.status if transition else "",
                "transition_outcome": transition.outcome if transition else "",
                "transition_reason": transition.reason if transition else "",
                "transition_attempt": transition.attempt if transition else None,
                "transition_attempt_count": len(transition_attempts),
                "elapsed_sec": transition.elapsed_sec if transition else None,
                "before_screenshot": str(before.get("screenshot") or ""),
                "after_screenshot": transition.screenshot if transition else "",
                "recorded_for_recipe": recorded is not None,
            }
        )

    return {
        "submission_id": submission.submission_id,
        "run_id": payload.run_id or submission.run_id,
        "site": payload.collection_intent.site,
        "goal": payload.goal,
        "run_status": payload.run_status,
        "step_count": len(steps),
        "recorded_action_count": len(recorded_by_seq),
        "transition_count": sum(len(items) for items in transitions_by_seq.values()),
        "observation_count": len(observation_ids),
        "ignored_record_count": ignored_records,
        "steps": steps,
    }


def render_worker_trace(trace: dict[str, Any]) -> str:
    """실행 경로를 터미널에서 읽기 쉬운 한국어 텍스트로 표시한다."""

    lines = [
        f"실행 ID: {trace.get('run_id') or '-'}",
        f"제출물 ID: {trace.get('submission_id') or '-'}",
        f"사이트: {trace.get('site') or '-'}",
        f"상태: {trace.get('run_status') or '-'}",
        (
            f"행동: {trace.get('step_count', 0)}개"
            f" / 레시피 기록 {trace.get('recorded_action_count', 0)}개"
            f" / 전환 관찰 {trace.get('transition_count', 0)}개"
            f" / 화면 관찰 {trace.get('observation_count', 0)}개"
        ),
    ]
    if trace.get("goal"):
        lines.append(f"목표: {_text_preview(trace['goal'])}")
    lines.append("")

    for step in trace.get("steps", []):
        target = step.get("target_label") or step.get("target_text") or ""
        action = str(step.get("action") or "알 수 없는 행동")
        action_label = f"{action} [{target}]" if target else action
        source_observation = _observation_suffix(
            str(step.get("before_observation_id") or "")
        )
        target_observation = (
            _observation_suffix(str(step.get("after_observation_id") or ""))
            if step.get("after_observation_id")
            else "전환 관찰 없음"
        )
        lines.append(
            f"[{int(step.get('seq') or 0):04d}] "
            f"{source_observation} -- {action_label} --> {target_observation}"
        )
        details = []
        if step.get("feedback_label"):
            details.append(f"피드백={step['feedback_label']}")
        if step.get("transition_status"):
            details.append(f"전환={step['transition_status']}")
        if step.get("transition_reason"):
            details.append(f"근거={step['transition_reason']}")
        if details:
            lines.append(f"       {' / '.join(details)}")

    if trace.get("ignored_record_count"):
        lines.extend(
            ["", f"순번이 없어 제외된 기록: {trace['ignored_record_count']}개"]
        )
    return "\n".join(lines)


__all__ = ["build_worker_trace", "render_worker_trace"]

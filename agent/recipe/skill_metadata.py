"""스킬 메타데이터 증거 빌더(skill metadata evidence builder).

자율 탐색 결과를 스킬처럼 검토할 수 있게 정리한다. 이 모듈은 후보가 좋은지
판단하지 않고, 지휘자/비평가(Commander/Critic)가 판단할 구조화된 근거만 만든다.
"""

from __future__ import annotations

from typing import Any

from agent.recipe.text_utils import normalize_text
from agent.recipe.task_category import normalize_task_category


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _proposal_for_seq(feedback_episodes: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    proposals: dict[int, dict[str, Any]] = {}
    for episode in feedback_episodes or []:
        if not isinstance(episode, dict):
            continue
        seq = episode.get("seq")
        proposal = episode.get("proposal")
        if isinstance(seq, int) and isinstance(proposal, dict):
            proposals[seq] = proposal
    return proposals


def _slot_refs(step: dict[str, Any]) -> list[str]:
    return _unique(
        [str(item) for item in (step.get("slot_refs") or [])]
    )


def _input_slots(
    recorded_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for step in recorded_steps or []:
        if not isinstance(step, dict):
            continue
        param = (
            step.get("param")
            if isinstance(step.get("param"), dict)
            else {}
        )
        for name in _slot_refs(step):
            if not name:
                continue
            slots.append(
                {
                    "name": name,
                    "description": normalize_text(step.get("intent")),
                    "observed_value": (
                        param.get("text")
                        if step.get("action") == "type_in_marker"
                        else step.get("value")
                    ),
                    "required": True,
                    "source": "recorded_step",
                }
            )

    deduped: dict[str, dict[str, Any]] = {}
    for slot in slots:
        deduped.setdefault(str(slot.get("name") or ""), slot)
    return [slot for name, slot in deduped.items() if name]


def _step_intents(recorded_steps: list[dict[str, Any]], feedback_episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proposals = _proposal_for_seq(feedback_episodes)
    intents: list[dict[str, Any]] = []
    for step in recorded_steps or []:
        if not isinstance(step, dict):
            continue
        seq = int(step.get("seq") or 0)
        proposal = proposals.get(seq, {})
        intent = normalize_text(step.get("intent") or proposal.get("reason") or proposal.get("llm_thought"))
        intents.append(
            {
                "seq": seq,
                "action": step.get("action", ""),
                "intent": intent,
                "target_role": normalize_text(step.get("target_role") or proposal.get("target_role_candidate")),
                "component": normalize_text(step.get("component") or proposal.get("component_candidate")),
                "expected_after": normalize_text(step.get("expected_after") or proposal.get("expected_after")),
                "replay_mode": normalize_text(
                    step.get("replay_mode")
                ).casefold()
                or "reasoning",
                "slot_refs": _slot_refs(step),
            }
        )
    return intents


def _verification(feedback_episodes: list[dict[str, Any]], extracted_summary: dict[str, Any]) -> dict[str, list[str]]:
    success_signals: list[str] = []
    failure_signals: list[str] = []
    fallback_conditions: list[str] = [
        "현재 화면의 OCR 마커가 기록된 대상과 매칭되지 않음(marker match miss)",
        "행동 후 OpenCV 화면 변화가 확인되지 않음(screen change miss)",
    ]

    if extracted_summary.get("has_data"):
        success_signals.append("수집 결과가 존재함(extracted_summary.has_data)")
    if extracted_summary.get("job_count"):
        success_signals.append("채용공고 개수가 증가함(extracted_summary.job_count)")

    for episode in feedback_episodes or []:
        feedback = episode.get("feedback") if isinstance(episode, dict) else {}
        if not isinstance(feedback, dict):
            continue
        label = feedback.get("label")
        reason = normalize_text(feedback.get("reason"))
        if label == "success" and reason:
            success_signals.append(reason)
        elif label in {"wrong_target", "no_effect", "loop_risk", "error"}:
            failure_signals.append(reason or str(label))

    return {
        "success_signals": _unique(success_signals),
        "failure_signals": _unique(failure_signals),
        "fallback_conditions": _unique(fallback_conditions),
    }


def build_skill_metadata_evidence(
    *,
    goal: str,
    site: str,
    keyword: str,
    target_count: int,
    task_category: str = "",
    recorded_steps: list[dict[str, Any]],
    feedback_episodes: list[dict[str, Any]],
    extracted_summary: dict[str, Any],
) -> dict[str, Any]:
    """작업 제출(WorkerSubmission)에 넣을 스킬 메타데이터 증거를 만든다."""

    step_intents = _step_intents(recorded_steps, feedback_episodes)
    return {
        "when_to_use": goal,
        "goal_pattern": goal,
        "goal": goal,
        "site": site,
        "task_category": normalize_task_category(task_category),
        "keyword": keyword,
        "target_count": target_count,
        "inputs": _input_slots(recorded_steps),
        "step_intents": step_intents,
        "actions": [step.get("action", "") for step in recorded_steps or [] if isinstance(step, dict)],
        "target_roles": _unique([item.get("target_role", "") for item in step_intents]),
        "components": _unique([item.get("component", "") for item in step_intents]),
        "verification": _verification(feedback_episodes, extracted_summary),
        "notes": (
            "자율탐색이 재사용 후보를 선언하고 비평가(Critic)는 "
            "실패하거나 불안정한 단계만 제거한다."
        ),
    }

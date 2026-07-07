"""스킬 메타데이터 증거 빌더(skill metadata evidence builder).

자율탐색 결과를 스킬처럼 검토할 수 있게 정리한다. 이 모듈은 후보가 좋은지
판단하지 않고, 지휘자/비평가(Commander/Critic)가 판단할 구조화된 근거만 만든다.
"""

from __future__ import annotations

from typing import Any

from agent.recipe.state_key import normalize_text
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


def _slot_refs(proposal: dict[str, Any], step: dict[str, Any]) -> list[str]:
    refs = list(step.get("slot_refs") or [])
    for candidate in proposal.get("parameter_candidates") or []:
        if isinstance(candidate, dict):
            refs.append(str(candidate.get("slot_candidate") or ""))
    return _unique(refs)


def _input_slots(feedback_episodes: list[dict[str, Any]], keyword: str, target_count: int) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for episode in feedback_episodes or []:
        proposal = episode.get("proposal") if isinstance(episode, dict) else {}
        if not isinstance(proposal, dict):
            continue
        for candidate in proposal.get("parameter_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            name = normalize_text(candidate.get("slot_candidate"))
            if not name:
                continue
            slots.append(
                {
                    "name": name,
                    "description": normalize_text(candidate.get("reason")),
                    "observed_value": candidate.get("value"),
                    "required": name in {"query", "keyword", "target_count"},
                    "source": "parameter_candidates",
                }
            )

    if keyword and not any(slot.get("name") == "query" for slot in slots):
        slots.append(
            {
                "name": "query",
                "description": "사용자 질의에서 추출된 검색어(search keyword)",
                "observed_value": keyword,
                "required": True,
                "source": "worker_keyword",
            }
        )
    if target_count > 0 and not any(slot.get("name") == "target_count" for slot in slots):
        slots.append(
            {
                "name": "target_count",
                "description": "사용자가 요청한 수집 개수(target count)",
                "observed_value": target_count,
                "required": False,
                "source": "worker_target_count",
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
                "fixed": step.get("fixed") if step.get("fixed") is not None else proposal.get("fixed_candidate"),
                "slot_refs": _slot_refs(proposal, step),
            }
        )
    return intents


def _verification(feedback_episodes: list[dict[str, Any]], extracted_summary: dict[str, Any]) -> dict[str, list[str]]:
    success_signals: list[str] = []
    failure_signals: list[str] = []
    fallback_conditions: list[str] = [
        "현재 화면의 OCR 마커가 기록된 대상과 매칭되지 않음(marker match miss)",
        "행동 후 전환 계약(transition contract)이 제한 시간 안에 충족되지 않음",
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
        "goal": goal,
        "site": site,
        "task_category": normalize_task_category(task_category),
        "keyword": keyword,
        "target_count": target_count,
        "inputs": _input_slots(feedback_episodes, keyword, target_count),
        "step_intents": step_intents,
        "actions": [step.get("action", "") for step in recorded_steps or [] if isinstance(step, dict)],
        "target_roles": _unique([item.get("target_role", "") for item in step_intents]),
        "components": _unique([item.get("component", "") for item in step_intents]),
        "verification": _verification(feedback_episodes, extracted_summary),
        "notes": "코드는 증거만 포장하고 재사용 가능성은 비평가(Critic)가 판단한다.",
    }

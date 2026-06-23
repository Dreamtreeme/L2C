"""반사 행동 이후 OCR 결과를 전환 계약과 비교한다."""

from __future__ import annotations

from typing import Any

from agent.recipe.state_key import normalize_text
from shared.schema.recipe_schema import TransitionContract, TransitionCue


def marker_texts(markers: list[dict[str, Any]]) -> list[str]:
    """현재 OCR 마커에서 비교 가능한 텍스트만 정규화한다."""
    seen: set[str] = set()
    out: list[str] = []
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        text = normalize_text(marker.get("text"))
        key = _key(text)
        if len(key) < 2 or key in seen or text.startswith("상호작용 가능한 요소"):
            continue
        seen.add(key)
        out.append(text)
    return out


def _key(value: Any) -> str:
    return normalize_text(value).casefold().replace(" ", "")


def _contains(texts: list[str], value: Any) -> bool:
    needle = _key(value)
    return bool(needle) and any(needle in _key(text) for text in texts)


def cue_matches(cue: TransitionCue | dict[str, Any], texts: list[str], params: dict[str, Any]) -> bool:
    """단일 구조화 단서를 현재 OCR 텍스트와 비교한다."""
    if not isinstance(cue, TransitionCue):
        cue = TransitionCue(**cue)

    if cue.kind == "text_any":
        return any(_contains(texts, value) for value in cue.values)
    if cue.kind == "text_all":
        return bool(cue.values) and all(_contains(texts, value) for value in cue.values)
    if cue.kind == "slot_text":
        return bool(cue.slot) and _contains(texts, params.get(cue.slot))
    if cue.kind == "min_text_markers":
        return len(texts) >= cue.min_count
    return False


def _all_match(cues, texts: list[str], params: dict[str, Any]) -> bool:
    return bool(cues) and all(cue_matches(cue, texts, params) for cue in cues)


def evaluate_transition(
    contract: TransitionContract | dict[str, Any] | None,
    markers: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    elapsed_sec: float = 0.0,
) -> dict[str, Any]:
    """전환 상태를 ready, pending, unknown 중 하나로 판정한다."""
    if not contract:
        return {"status": "unknown", "outcome": "", "reason": "transition_contract_missing"}
    if not isinstance(contract, TransitionContract):
        contract = TransitionContract(**contract)
    if not contract.common_ready_cues and not contract.outcomes:
        return {"status": "unknown", "outcome": "", "reason": "transition_contract_empty"}

    texts = marker_texts(markers)
    params = dict(params or {})
    common_ready = not contract.common_ready_cues or _all_match(contract.common_ready_cues, texts, params)
    matched_outcome = ""
    if common_ready:
        if not contract.outcomes:
            return {"status": "ready", "outcome": "", "reason": "common_ready_cues_matched"}
        for outcome in contract.outcomes:
            if _all_match(outcome.cues, texts, params):
                matched_outcome = outcome.name
                break
        if matched_outcome:
            return {"status": "ready", "outcome": matched_outcome, "reason": "outcome_cues_matched"}

    if elapsed_sec >= contract.timeout_sec:
        return {"status": "unknown", "outcome": "", "reason": "transition_timeout"}

    loading_match = any(cue_matches(cue, texts, params) for cue in contract.loading_cues)
    return {
        "status": "pending",
        "outcome": "",
        "reason": "loading_cue_matched" if loading_match else "ready_cues_not_observed",
    }

"""E2E summary를 LangSmith 성과·실패 지표로 변환한다."""

from __future__ import annotations

from typing import Any

from agent.observability.stages import stage_for_component
from agent.observability.reflex_paths import summarize_reflex_paths


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def build_e2e_observability(summary: dict[str, Any]) -> dict[str, Any]:
    """한 E2E 실행의 최종 성과와 내부 복구 실패를 분리해 계산한다."""

    status = str(summary.get("status") or "failed")
    quality = dict(summary.get("quality") or {})
    metrics = dict(summary.get("metrics") or {})
    result = dict(summary.get("result") or {})
    steps = [item for item in (metrics.get("steps") or []) if isinstance(item, dict)]
    llm = dict(metrics.get("llm") or {})
    llm_calls = [item for item in (llm.get("calls") or []) if isinstance(item, dict)]
    totals = dict(llm.get("totals") or {})

    passed = bool(status == "completed" and quality.get("passed"))
    if passed:
        outcome = "success"
    elif status == "completed":
        outcome = "partial"
    elif status == "cancelled":
        outcome = "cancelled"
    else:
        outcome = "failed"

    failed_steps = [item for item in steps if item.get("success") is False]
    terminal_failure_stage = ""
    terminal_failure_code = ""
    if not passed:
        if failed_steps:
            failed_step = failed_steps[-1]
            component = str(failed_step.get("component") or "")
            terminal_failure_stage = str(
                failed_step.get("stage") or stage_for_component(component)
            )
            terminal_failure_code = str(
                failed_step.get("failure_code") or "step_failed"
            )
        elif outcome == "partial":
            terminal_failure_stage = "quality_gate"
            terminal_failure_code = "quality_not_passed"
        else:
            context_outcome = dict(metrics.get("outcome") or {})
            terminal_failure_stage = str(
                context_outcome.get("failure_stage") or "run"
            )
            terminal_failure_code = str(
                context_outcome.get("failure_code") or "run_failed"
            )

    persisted_count = _as_int(
        result.get("persisted_count") or quality.get("persisted_count")
    )
    total_tokens = _as_int(totals.get("total_tokens"))
    estimated_cost = (llm.get("cost") or {}).get("estimated_total")
    cost_value = None if estimated_cost is None else _as_float(estimated_cost)

    ocr_steps = [item for item in steps if item.get("component") == "ocr_request"]
    ocr_timeout_count = sum(
        1 for item in ocr_steps if item.get("failure_code") == "ocr_timeout"
    )
    reflex_hits = sum(
        1
        for item in steps
        if item.get("component") == "graph:reflex"
        and item.get("action_source") == "reflex"
    )
    queue_replay_hits = sum(
        1
        for item in steps
        if item.get("component") == "graph:selection"
        and item.get("action_source") == "job_card_queue"
    )
    reasoning_calls = sum(
        1
        for item in llm_calls
        if str(item.get("component") or "").startswith("vision_reasoning")
    )
    reflex_paths = summarize_reflex_paths(steps)

    return {
        "outcome": outcome,
        "e2e_success": int(passed),
        "terminal_failure_stage": terminal_failure_stage,
        "terminal_failure_code": terminal_failure_code,
        "execution_time_sec": round(_as_float(summary.get("execution_time_sec")), 6),
        "target_fulfillment": quality.get("target_fulfillment"),
        "persistence_rate": _as_float(quality.get("persistence_rate")),
        "persisted_count": persisted_count,
        "llm_call_count": len(llm_calls),
        "reasoning_call_count": reasoning_calls,
        "total_tokens": total_tokens,
        "tokens_per_persisted_item": (
            round(total_tokens / persisted_count, 6) if persisted_count else None
        ),
        "estimated_cost_usd": cost_value,
        "cost_per_persisted_item_usd": (
            round(cost_value / persisted_count, 9)
            if cost_value is not None and persisted_count
            else None
        ),
        "wasted_tokens": 0 if passed else total_tokens,
        "ocr_request_count": len(ocr_steps),
        "ocr_timeout_count": ocr_timeout_count,
        "recovered_failure_count": len(failed_steps) if passed else 0,
        "recovery_success": (
            int(passed) if failed_steps else None
        ),
        "reflex_hits": reflex_hits,
        **reflex_paths,
        "queue_replay_hits": queue_replay_hits,
        "internal_failure_codes": sorted(
            {
                str(item.get("failure_code") or "step_failed")
                for item in failed_steps
            }
        ),
    }


def build_langsmith_feedback(observability: dict[str, Any]) -> list[dict[str, Any]]:
    """대시보드에서 집계할 수치와 실패 범주를 feedback 레코드로 만든다."""

    numeric_keys = (
        "e2e_success",
        "execution_time_sec",
        "target_fulfillment",
        "persistence_rate",
        "persisted_count",
        "llm_call_count",
        "reasoning_call_count",
        "total_tokens",
        "tokens_per_persisted_item",
        "estimated_cost_usd",
        "cost_per_persisted_item_usd",
        "wasted_tokens",
        "ocr_request_count",
        "ocr_timeout_count",
        "recovered_failure_count",
        "recovery_success",
        "reflex_hits",
        "reflex_step_hit_count",
        "reflex_step_completed_count",
        "reflex_source_reasoning_replaced_count",
        "reflex_reasoning_call_reduction",
        "reflex_path_started_count",
        "reflex_path_completed_count",
        "reflex_path_failed_count",
        "reflex_path_mid_failure_count",
        "reflex_path_fallback_count",
        "reflex_path_incomplete_count",
        "reflex_path_completion_rate",
        "queue_replay_hits",
    )
    feedback = [
        {"key": key, "score": observability[key]}
        for key in numeric_keys
        if observability.get(key) is not None
    ]
    feedback.append({"key": "e2e_outcome", "value": observability.get("outcome", "")})
    if observability.get("terminal_failure_stage"):
        feedback.append(
            {
                "key": "terminal_failure_stage",
                "value": observability["terminal_failure_stage"],
            }
        )
    if observability.get("terminal_failure_code"):
        feedback.append(
            {
                "key": "terminal_failure_code",
                "value": observability["terminal_failure_code"],
            }
        )
    return feedback


__all__ = ["build_e2e_observability", "build_langsmith_feedback"]

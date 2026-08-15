"""구조화된 E2E 실행 요약에서 시간 분포와 Reflex 효과를 계산한다."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agent.observability.reflex_paths import summarize_reflex_paths


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * ratio) - 1)
    return ordered[index]


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "total": 0.0,
            "avg": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    total = sum(values)
    return {
        "count": len(values),
        "total": round(total, 3),
        "avg": round(total / len(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def _step_duration(item: dict[str, Any]) -> float | None:
    try:
        return float(item.get("duration_sec") or 0.0)
    except (TypeError, ValueError):
        return None


def _collect_step_metrics(
    steps: list[Any],
) -> tuple[
    dict[str, list[float]],
    int,
    int,
    Counter[str],
]:
    durations: dict[str, list[float]] = defaultdict(list)
    reflex_hits = 0
    queue_replay_hits = 0
    reasoning_modes: Counter[str] = Counter()
    for item in steps:
        if not isinstance(item, dict):
            continue
        duration = _step_duration(item)
        if duration is None:
            continue
        component = str(item.get("component") or "unknown").removeprefix("graph:")
        durations[component].append(duration)
        if component == "execution":
            for action_name in item.get("action_names") or []:
                if str(action_name):
                    durations[f"action ({action_name})"].append(duration)
        action_source = str(item.get("action_source") or "")
        reflex_hits += int(component == "reflex" and action_source == "reflex")
        queue_replay_hits += int(
            component == "selection" and action_source == "job_card_queue"
        )
        if component == "reasoning":
            reasoning_modes[str(item.get("reasoning_mode") or "general")] += 1
    return (
        durations,
        reflex_hits,
        queue_replay_hits,
        reasoning_modes,
    )


def _collect_llm_durations(calls: list[Any]) -> dict[str, list[float]]:
    durations: dict[str, list[float]] = defaultdict(list)
    for call in calls:
        if not isinstance(call, dict):
            continue
        duration = _step_duration(call)
        if duration is not None:
            durations[str(call.get("component") or "unknown")].append(duration)
    return durations


def _action_stats(durations: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    return {
        component.removeprefix("action (").removesuffix(")"): _stats(values)
        for component, values in sorted(durations.items())
        if component.startswith("action (")
    }


def profile_summary(path: Path) -> dict[str, Any]:
    """실행 중 수집된 구조화 지표만 집계한다."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    (
        durations,
        reflex_hits,
        queue_replay_hits,
        reasoning_modes,
    ) = _collect_step_metrics(metrics.get("steps") or [])
    llm_durations = _collect_llm_durations(
        (metrics.get("llm") or {}).get("calls") or []
    )
    reflex_paths = summarize_reflex_paths(metrics.get("steps") or [])

    return {
        "path": str(path),
        "source": "summary",
        "run_id": payload.get("run_id", ""),
        "status": payload.get("status", ""),
        "execution_time_sec": payload.get("execution_time_sec"),
        "nodes": {name: _stats(values) for name, values in sorted(durations.items())},
        "actions": _action_stats(durations),
        "llm_calls": {name: _stats(values) for name, values in sorted(llm_durations.items())},
        "llm_usage": (metrics.get("llm") or {}).get("totals", {}),
        "llm_cost": (metrics.get("llm") or {}).get("cost", {}),
        "quality": payload.get("quality", {}),
        "reflex_hits": reflex_hits,
        **reflex_paths,
        "queue_replay_hits": queue_replay_hits,
        "reasoning_modes": dict(reasoning_modes),
    }


def profile_path(path: Path) -> dict[str, Any]:
    """지원하는 구조화 실행 요약을 읽는다."""

    if not path.name.endswith(".summary.json"):
        raise ValueError("성능 분석 입력은 구조화된 .summary.json 파일이어야 합니다.")
    return profile_summary(path)


def _print_stats_group(name: str, values: dict[str, dict[str, Any]]) -> None:
    print(f"{name}:")
    for item_name, stats in values.items():
        print(
            f"  {item_name}: count={stats['count']} total={stats['total']}s "
            f"avg={stats['avg']}s p50={stats['p50']}s p95={stats['p95']}s max={stats['max']}s"
        )


def _print_report(report: dict[str, Any]) -> None:
    print(f"# {report['path']}")
    print(f"source: {report.get('source', '')}")
    print(f"execution_time_sec: {report.get('execution_time_sec')}")
    _print_stats_group("nodes", report.get("nodes", {}))
    _print_stats_group("actions", report.get("actions", {}))
    if report.get("llm_calls"):
        _print_stats_group("llm_calls", report["llm_calls"])
    print(f"reflex_hits: {report.get('reflex_hits', 0)}")
    print(
        "reflex_reasoning_call_reduction: "
        f"{report.get('reflex_reasoning_call_reduction', 0)}"
    )
    print(
        "reflex_path_completed_count: "
        f"{report.get('reflex_path_completed_count', 0)}"
    )
    print(
        "reflex_path_fallback_count: "
        f"{report.get('reflex_path_fallback_count', 0)}"
    )
    print(f"queue_replay_hits: {report.get('queue_replay_hits', 0)}")
    print(f"reasoning_modes: {report.get('reasoning_modes', {})}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile runtime metrics from structured E2E summaries."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    for index, path in enumerate(args.artifacts):
        if index:
            print()
        _print_report(profile_path(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

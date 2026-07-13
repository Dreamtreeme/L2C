"""E2E 로그 또는 실행 요약에서 시간 분포와 Reflex 효과를 요약한다."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NODE_PATTERNS = {
    "perception": (
        re.compile(r"Perception Node completed in ([0-9.]+) seconds"),
        re.compile(r"Perception Node completed.*duration_sec[\"']?\s*[:=]\s*([0-9.]+)"),
    ),
    "reasoning": (
        re.compile(r"Reasoning Node completed in ([0-9.]+) seconds"),
        re.compile(r"Reasoning Node completed.*duration_sec[\"']?\s*[:=]\s*([0-9.]+)"),
    ),
    "action_total": (
        re.compile(r"Action Node completed all chained tools in ([0-9.]+) seconds"),
        re.compile(r"Action Node completed all chained tools.*duration_sec[\"']?\s*[:=]\s*([0-9.]+)"),
    ),
    "reflex": (re.compile(r"Reflex hit.*duration=([0-9.]+)s"),),
    "ocr_request": (
        re.compile(r"PaddleOCR worker request completed.*duration=([0-9.]+)s"),
    ),
    "ocr_startup": (
        re.compile(r"PaddleOCR worker ready.*startup=([0-9.]+)s"),
    ),
}
ACTION_PATTERN = re.compile(r"Action Node \[([^\]]+)\] completed in ([0-9.]+) seconds")
PHASH_MISS_PATTERN = re.compile(r"pHash replay check failed.*reason=([^\s]+)")
REFLEX_MISS_PATTERN = re.compile(r"Reflex miss: ([^\[]+)")
QUEUE_REPLAY_HIT_PATTERN = re.compile(r"Result card queue replay prepared")
EXECUTION_TIME_PATTERN = re.compile(r"EXECUTION_TIME_SEC=([0-9.]+)")


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


def _read_log_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    return data.decode("utf-8", errors="replace")


def _append_pattern_duration(line: str, patterns: tuple[re.Pattern[str], ...], values: list[float]) -> bool:
    for pattern in patterns:
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)))
            return True
    return False


def _seconds(value: Any) -> float | None:
    try:
        return float(str(value).strip().removesuffix("s"))
    except (TypeError, ValueError):
        return None


def _structured_payload(line: str) -> dict[str, Any]:
    stripped = line.lstrip()
    if not stripped.startswith("{"):
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _structured_durations(line: str) -> dict[str, float]:
    payload = _structured_payload(line)
    if not payload:
        return {}
    event = str(payload.get("event") or "")
    failed_ocr_runtime = (
        event == "Runtime step completed"
        and str(payload.get("component") or "") == "ocr_request"
        and payload.get("success") is False
    )
    if event == "PaddleOCR worker request failed" or failed_ocr_runtime:
        duration = _seconds(payload.get("duration_sec"))
        return {
            "ocr_request": duration,
            "ocr_request_failed": duration,
        } if duration is not None else {}
    if event == "SoM analysis stages completed":
        result = {}
        for name, key in (
            ("som_ocr", "ocr_duration_sec"),
            ("som_yolo", "yolo_duration_sec"),
        ):
            duration = _seconds(payload.get(key))
            if duration is not None:
                result[name] = duration
        return result
    mapping = {
        "Perception Node completed": ("perception", payload.get("duration_sec")),
        "Reasoning Node completed": ("reasoning", payload.get("duration_sec")),
        "Action Node completed all chained tools": ("action_total", payload.get("duration_sec")),
        "Reflex hit": ("reflex", payload.get("duration") or payload.get("duration_sec")),
        "PaddleOCR worker request completed": ("ocr_request", payload.get("duration")),
        "PaddleOCR worker ready": ("ocr_startup", payload.get("startup")),
    }
    item = mapping.get(event)
    if item is None:
        return {}
    name, raw_duration = item
    duration = _seconds(raw_duration)
    return {name: duration} if duration is not None else {}


def profile_log(path: Path) -> dict[str, Any]:
    durations: dict[str, list[float]] = defaultdict(list)
    action_durations: dict[str, list[float]] = defaultdict(list)
    phash_miss_reasons: Counter[str] = Counter()
    reflex_miss_reasons: Counter[str] = Counter()
    reflex_hits = 0
    queue_replay_hits = 0
    perception_modes: Counter[str] = Counter()
    reasoning_modes: Counter[str] = Counter()
    execution_time_sec: float | None = None

    for line in _read_log_text(path).splitlines():
        structured_payload = _structured_payload(line)
        structured_event = str(structured_payload.get("event") or "")
        structured = _structured_durations(line)
        for name, duration in structured.items():
            durations[name].append(duration)
            if name == "reflex":
                reflex_hits += 1
        for name, patterns in NODE_PATTERNS.items():
            if name in structured:
                continue
            if _append_pattern_duration(line, patterns, durations[name]) and name == "reflex":
                reflex_hits += 1
        action_match = ACTION_PATTERN.search(line)
        if action_match:
            action_durations[action_match.group(1)].append(float(action_match.group(2)))
        phash_match = PHASH_MISS_PATTERN.search(line)
        if phash_match:
            phash_miss_reasons[phash_match.group(1)] += 1
        if structured_event.startswith("Reflex miss: "):
            reflex_miss_reasons[structured_event.removeprefix("Reflex miss: ").strip()] += 1
        else:
            miss_match = REFLEX_MISS_PATTERN.search(line)
            if miss_match:
                reflex_miss_reasons[miss_match.group(1).strip()] += 1
        if structured_event == "Result card queue replay prepared" or QUEUE_REPLAY_HIT_PATTERN.search(line):
            queue_replay_hits += 1
        if structured_event == "SoM analysis stages completed":
            perception_modes[str(structured_payload.get("mode") or "unknown")] += 1
        if structured_event == "Reasoning Node completed":
            reasoning_modes[str(structured_payload.get("reasoning_mode") or "general")] += 1
        execution_match = EXECUTION_TIME_PATTERN.search(line)
        if execution_match:
            execution_time_sec = float(execution_match.group(1))

    reasoning_total = sum(durations["reasoning"])
    reflex_total = sum(durations["reflex"])
    avoided_reasoning_estimate = 0.0
    if durations["reasoning"] and reflex_hits:
        avoided_reasoning_estimate = reflex_hits * (reasoning_total / len(durations["reasoning"]))

    return {
        "path": str(path),
        "source": "log",
        "execution_time_sec": execution_time_sec,
        "nodes": {name: _stats(values) for name, values in sorted(durations.items())},
        "actions": {name: _stats(values) for name, values in sorted(action_durations.items())},
        "reflex_hits": reflex_hits,
        "queue_replay_hits": queue_replay_hits,
        "perception_modes": dict(perception_modes),
        "reasoning_modes": dict(reasoning_modes),
        "reflex_misses": dict(reflex_miss_reasons),
        "phash_miss_reasons": dict(phash_miss_reasons),
        "estimated_reasoning_seconds_avoided": round(avoided_reasoning_estimate, 3),
        "reflex_seconds_spent": round(reflex_total, 3),
    }


def profile_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    steps = metrics.get("steps") or []
    durations: dict[str, list[float]] = defaultdict(list)
    reflex_hits = 0
    queue_replay_hits = 0
    perception_modes: Counter[str] = Counter()
    reasoning_modes: Counter[str] = Counter()
    for item in steps:
        if not isinstance(item, dict):
            continue
        component = str(item.get("component") or "unknown")
        if component.startswith("graph:"):
            component = component.removeprefix("graph:")
        try:
            durations[component].append(float(item.get("duration_sec") or 0.0))
        except (TypeError, ValueError):
            continue
        if component == "reflex" and item.get("hit") is True:
            reflex_hits += 1
        if component == "perception" and item.get("queue_replay_hit") is True:
            queue_replay_hits += 1
        if component == "perception":
            perception_modes[str(item.get("analysis_mode") or "full")] += 1
        if component == "reasoning":
            reasoning_modes[str(item.get("reasoning_mode") or "general")] += 1

    llm_durations: dict[str, list[float]] = defaultdict(list)
    for call in (metrics.get("llm") or {}).get("calls") or []:
        if not isinstance(call, dict):
            continue
        try:
            llm_durations[str(call.get("component") or "unknown")].append(
                float(call.get("duration_sec") or 0.0)
            )
        except (TypeError, ValueError):
            continue

    return {
        "path": str(path),
        "source": "summary",
        "run_id": payload.get("run_id", ""),
        "status": payload.get("status", ""),
        "execution_time_sec": payload.get("execution_time_sec"),
        "nodes": {name: _stats(values) for name, values in sorted(durations.items())},
        "actions": {
            name.removeprefix("action (").removesuffix(")"): stats
            for name, stats in (
                (component, _stats(values))
                for component, values in sorted(durations.items())
                if component.startswith("action (")
            )
        },
        "llm_calls": {name: _stats(values) for name, values in sorted(llm_durations.items())},
        "llm_usage": (metrics.get("llm") or {}).get("totals", {}),
        "llm_cost": (metrics.get("llm") or {}).get("cost", {}),
        "quality": payload.get("quality", {}),
        "reflex_hits": reflex_hits,
        "queue_replay_hits": queue_replay_hits,
        "perception_modes": dict(perception_modes),
        "reasoning_modes": dict(reasoning_modes),
    }


def profile_path(path: Path) -> dict[str, Any]:
    if path.name.endswith(".summary.json"):
        return profile_summary(path)
    return profile_log(path)


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
    print(f"queue_replay_hits: {report.get('queue_replay_hits', 0)}")
    print(f"perception_modes: {report.get('perception_modes', {})}")
    print(f"reasoning_modes: {report.get('reasoning_modes', {})}")
    if "estimated_reasoning_seconds_avoided" in report:
        print(f"estimated_reasoning_seconds_avoided: {report['estimated_reasoning_seconds_avoided']}s")
        print(f"reflex_seconds_spent: {report['reflex_seconds_spent']}s")
    for key in ("reflex_misses", "phash_miss_reasons"):
        if report.get(key):
            print(f"{key}:")
            for reason, count in report[key].items():
                print(f"  {reason}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile Reflex and runtime metrics from E2E artifacts.")
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    for index, path in enumerate(args.artifacts):
        if index:
            print()
        _print_report(profile_path(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

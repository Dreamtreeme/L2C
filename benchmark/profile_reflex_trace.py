"""E2E 로그에서 Reflex/pHash/reasoning 시간 비중을 요약한다."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


NODE_PATTERNS = {
    "perception": re.compile(r"Perception Node completed in ([0-9.]+) seconds"),
    "reasoning": re.compile(r"Reasoning Node completed in ([0-9.]+) seconds"),
    "action_total": re.compile(r"Action Node completed all chained tools in ([0-9.]+) seconds"),
    "reflex": re.compile(r"Reflex hit.*duration=([0-9.]+)s"),
}
ACTION_PATTERN = re.compile(r"Action Node \[([^\]]+)\] completed in ([0-9.]+) seconds")
PHASH_MISS_PATTERN = re.compile(r"pHash replay check failed.*reason=([^\s]+)")
REFLEX_MISS_PATTERN = re.compile(r"Reflex miss: ([^\[]+)")
FAST_PERCEPTION_PATTERN = re.compile(r"Perception Node completed in ([0-9.]+) seconds via pHash ROI fast path")
FAST_HIT_PATTERN = re.compile(r"Reflex pHash fast path hit")
FAST_MISS_PATTERN = re.compile(r"Reflex pHash fast path miss.*reason=([^\s]+)")
ROI_OCR_PATTERN = re.compile(r"ROI OCR complete")


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "total": 0.0, "avg": 0.0}
    total = sum(values)
    return {"count": len(values), "total": round(total, 3), "avg": round(total / len(values), 3)}


def _read_log_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    return data.decode("utf-8", errors="replace")


def profile_log(path: Path) -> dict:
    durations: dict[str, list[float]] = defaultdict(list)
    action_durations: dict[str, list[float]] = defaultdict(list)
    phash_miss_reasons: Counter[str] = Counter()
    reflex_miss_reasons: Counter[str] = Counter()
    fast_miss_reasons: Counter[str] = Counter()
    reflex_hits = 0
    fast_hits = 0
    roi_ocr_calls = 0

    for line in _read_log_text(path).splitlines():
        for name, pattern in NODE_PATTERNS.items():
            match = pattern.search(line)
            if match:
                durations[name].append(float(match.group(1)))
                if name == "reflex":
                    reflex_hits += 1
        fast_perception_match = FAST_PERCEPTION_PATTERN.search(line)
        if fast_perception_match:
            durations["perception_phash_roi"].append(float(fast_perception_match.group(1)))
        if FAST_HIT_PATTERN.search(line):
            fast_hits += 1
        fast_miss_match = FAST_MISS_PATTERN.search(line)
        if fast_miss_match:
            fast_miss_reasons[fast_miss_match.group(1)] += 1
        if ROI_OCR_PATTERN.search(line):
            roi_ocr_calls += 1
        action_match = ACTION_PATTERN.search(line)
        if action_match:
            action_durations[action_match.group(1)].append(float(action_match.group(2)))
        phash_match = PHASH_MISS_PATTERN.search(line)
        if phash_match:
            phash_miss_reasons[phash_match.group(1)] += 1
        miss_match = REFLEX_MISS_PATTERN.search(line)
        if miss_match:
            reflex_miss_reasons[miss_match.group(1).strip()] += 1

    reasoning_total = sum(durations["reasoning"])
    reflex_total = sum(durations["reflex"])
    avoided_reasoning_estimate = 0.0
    if durations["reasoning"] and reflex_hits:
        avoided_reasoning_estimate = reflex_hits * (reasoning_total / len(durations["reasoning"]))

    return {
        "path": str(path),
        "nodes": {name: _stats(values) for name, values in sorted(durations.items())},
        "actions": {name: _stats(values) for name, values in sorted(action_durations.items())},
        "reflex_hits": reflex_hits,
        "reflex_misses": dict(reflex_miss_reasons),
        "phash_miss_reasons": dict(phash_miss_reasons),
        "phash_roi_fast_hits": fast_hits,
        "phash_roi_fast_misses": dict(fast_miss_reasons),
        "roi_ocr_calls": roi_ocr_calls,
        "estimated_reasoning_seconds_avoided": round(avoided_reasoning_estimate, 3),
        "reflex_seconds_spent": round(reflex_total, 3),
    }


def _print_report(report: dict) -> None:
    print(f"# {report['path']}")
    print("nodes:")
    for name, stats in report["nodes"].items():
        print(f"  {name}: count={stats['count']} total={stats['total']}s avg={stats['avg']}s")
    print("actions:")
    for name, stats in report["actions"].items():
        print(f"  {name}: count={stats['count']} total={stats['total']}s avg={stats['avg']}s")
    print(f"reflex_hits: {report['reflex_hits']}")
    print(f"phash_roi_fast_hits: {report['phash_roi_fast_hits']}")
    print(f"roi_ocr_calls: {report['roi_ocr_calls']}")
    print(f"estimated_reasoning_seconds_avoided: {report['estimated_reasoning_seconds_avoided']}s")
    print(f"reflex_seconds_spent: {report['reflex_seconds_spent']}s")
    if report["reflex_misses"]:
        print("reflex_misses:")
        for reason, count in report["reflex_misses"].items():
            print(f"  {reason}: {count}")
    if report["phash_miss_reasons"]:
        print("phash_miss_reasons:")
        for reason, count in report["phash_miss_reasons"].items():
            print(f"  {reason}: {count}")
    if report["phash_roi_fast_misses"]:
        print("phash_roi_fast_misses:")
        for reason, count in report["phash_roi_fast_misses"].items():
            print(f"  {reason}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile Reflex/pHash timing from E2E logs.")
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    for index, path in enumerate(args.logs):
        if index:
            print()
        _print_report(profile_log(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

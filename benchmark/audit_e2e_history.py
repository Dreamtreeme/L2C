"""E2E 실행 기록을 비교 가능한 증거 집합으로 분류한다."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REQUIRED_IDENTITY_FIELDS = (
    "git_commit",
    "config_fingerprint",
    "scenario_id",
    "site",
    "run_mode",
)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _median(values: Iterable[float]) -> float | None:
    items = list(values)
    return round(float(statistics.median(items)), 6) if items else None


def _range_summary(values: Iterable[float]) -> dict[str, float | None]:
    items = list(values)
    return {
        "min": round(min(items), 6) if items else None,
        "median": _median(items),
        "max": round(max(items), 6) if items else None,
    }


def _load_run(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "invalid_json"
    if payload.get("schema_version") != 2 or not payload.get("site"):
        return None, "non_collection_summary"

    missing_fields = [
        field for field in REQUIRED_IDENTITY_FIELDS if not payload.get(field)
    ]
    if missing_fields:
        classification = "incomplete_identity"
    elif payload.get("git_dirty"):
        classification = "development"
    else:
        classification = "release"

    observability = dict(payload.get("observability") or {})
    quality = dict(payload.get("quality") or {})
    run = {
        "path": str(path),
        "classification": classification,
        "missing_identity_fields": missing_fields,
        "started_at": str(payload.get("started_at") or ""),
        "git_commit": str(payload.get("git_commit") or ""),
        "git_dirty": bool(payload.get("git_dirty")),
        "config_fingerprint": str(payload.get("config_fingerprint") or ""),
        "scenario_id": str(payload.get("scenario_id") or ""),
        "site": str(payload.get("site") or ""),
        "run_mode": str(payload.get("run_mode") or ""),
        "query": str(payload.get("query") or ""),
        "target_count": _as_int(payload.get("target_count")) or 0,
        "recipe_version": str(payload.get("recipe_version") or ""),
        "status": str(payload.get("status") or ""),
        "passed": bool(quality.get("passed")),
        "execution_time_sec": _as_float(payload.get("execution_time_sec")),
        "total_tokens": _as_int(observability.get("total_tokens")),
        "estimated_cost_usd": _as_float(observability.get("estimated_cost_usd")),
        "reflex_hits": _as_int(observability.get("reflex_hits")),
        "ocr_timeout_count": _as_int(observability.get("ocr_timeout_count")),
    }
    return run, classification


def _group_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return (
        run["classification"],
        run["git_commit"],
        run["config_fingerprint"],
        run["scenario_id"],
        run["site"],
        run["run_mode"],
        run["query"],
        run["target_count"],
        run["recipe_version"],
    )


def _summarize_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    first = runs[0]
    execution_times = [
        value
        for value in (run["execution_time_sec"] for run in runs)
        if value is not None
    ]
    total_tokens = [
        float(value)
        for value in (run["total_tokens"] for run in runs)
        if value is not None
    ]
    costs = [
        value
        for value in (run["estimated_cost_usd"] for run in runs)
        if value is not None
    ]
    reflex_hits = [
        float(value)
        for value in (run["reflex_hits"] for run in runs)
        if value is not None
    ]
    return {
        "classification": first["classification"],
        "git_commit": first["git_commit"],
        "config_fingerprint": first["config_fingerprint"],
        "scenario_id": first["scenario_id"],
        "site": first["site"],
        "run_mode": first["run_mode"],
        "query": first["query"],
        "target_count": first["target_count"],
        "recipe_version": first["recipe_version"],
        "count": len(runs),
        "passed_count": sum(bool(run["passed"]) for run in runs),
        "failed_count": sum(not bool(run["passed"]) for run in runs),
        "execution_time_sec": _range_summary(execution_times),
        "total_tokens": _range_summary(total_tokens),
        "estimated_cost_usd": _range_summary(costs),
        "reflex_hits": _range_summary(reflex_hits),
        "ocr_timeout_count": sum(
            int(run["ocr_timeout_count"] or 0) for run in runs
        ),
        "paths": [run["path"] for run in runs],
    }


def build_history_audit(root: Path) -> dict[str, Any]:
    """디렉터리 아래 summary를 읽고 출처 수준과 비교 단위로 묶는다."""

    paths = sorted(root.rglob("*.summary.json"))
    runs: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for path in paths:
        run, reason = _load_run(path)
        if run is None:
            skipped[reason] += 1
            continue
        runs.append(run)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[_group_key(run)].append(run)

    groups = [_summarize_group(items) for items in grouped.values()]
    groups.sort(
        key=lambda item: (
            item["classification"] != "release",
            -int(item["count"]),
            item["site"],
            item["scenario_id"],
        )
    )
    classifications = Counter(run["classification"] for run in runs)
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "root": str(root.resolve()),
        "discovered_summary_count": len(paths),
        "collection_run_count": len(runs),
        "classification_counts": dict(sorted(classifications.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "repeated_group_count": sum(group["count"] >= 2 for group in groups),
        "singleton_group_count": sum(group["count"] == 1 for group in groups),
        "groups": groups,
    }


def _display_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def render_markdown(
    audit: dict[str, Any],
    *,
    minimum_group_size: int = 1,
) -> str:
    """작은 표본에 p95를 붙이지 않고 원시 범위를 보여 주는 보고서를 만든다."""

    counts = dict(audit.get("classification_counts") or {})
    skipped = dict(audit.get("skipped_counts") or {})
    lines = [
        "# E2E 실행 기록 감사",
        "",
        f"- 생성 시각: `{audit.get('generated_at', '')}`",
        f"- 탐색한 summary: `{audit.get('discovered_summary_count', 0)}`개",
        f"- 수집 E2E 기록: `{audit.get('collection_run_count', 0)}`개",
        f"- 기준 커밋 후보(clean): `{counts.get('release', 0)}`개",
        f"- 개발 중 기록(dirty): `{counts.get('development', 0)}`개",
        f"- 식별자 불완전: `{counts.get('incomplete_identity', 0)}`개",
        f"- 비교 가능한 반복 그룹: `{audit.get('repeated_group_count', 0)}`개",
        f"- 수집 E2E가 아닌 summary: `{skipped.get('non_collection_summary', 0)}`개",
        "",
        "## 해석 기준",
        "",
        "- `release`: 깨끗한 작업 트리에서 실행되어 최종 기준값 후보로 사용할 수 있다.",
        "- `development`: 변경 파일이 있는 상태의 실행이다. 성능 기준값이 아니라 회귀·트러블슈팅 증거로 사용한다.",
        "- 같은 분류, 커밋, 설정 fingerprint, 시나리오, 사이트, 실행 모드, 질의, 목표 수만 한 그룹으로 묶는다.",
        "- 작은 표본에는 p95를 계산하지 않고 성공 건수와 실행시간 최소·중앙·최대값을 그대로 표시한다.",
        "",
        "## 비교 그룹",
        "",
        "| 분류 | 커밋 | 설정 | 시나리오 | 모드 | 성공/전체 | 실행시간 최소/중앙/최대(초) | 토큰 중앙 | 비용 중앙($) | Reflex 중앙 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    visible_groups = [
        group
        for group in audit.get("groups", [])
        if int(group.get("count") or 0) >= max(1, minimum_group_size)
    ]
    for group in visible_groups:
        duration = dict(group.get("execution_time_sec") or {})
        token = dict(group.get("total_tokens") or {})
        cost = dict(group.get("estimated_cost_usd") or {})
        reflex = dict(group.get("reflex_hits") or {})
        duration_text = "/".join(
            _display_number(duration.get(name))
            for name in ("min", "median", "max")
        )
        lines.append(
            "| {classification} | `{commit}` | `{config}` | {scenario} | {mode} | "
            "{passed}/{count} | {duration} | {tokens} | {cost} | {reflex} |".format(
                classification=group.get("classification", ""),
                commit=str(group.get("git_commit") or "")[:8],
                config=group.get("config_fingerprint", ""),
                scenario=group.get("scenario_id", ""),
                mode=group.get("run_mode", ""),
                passed=group.get("passed_count", 0),
                count=group.get("count", 0),
                duration=duration_text,
                tokens=_display_number(token.get("median"), 0),
                cost=_display_number(cost.get("median"), 4),
                reflex=_display_number(reflex.get("median"), 1),
            )
        )
    if not visible_groups:
        lines.append("| - | - | - | - | - | 0/0 | - | - | - | - |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="기존 E2E summary를 비교 가능한 증거 집합으로 분류합니다."
    )
    parser.add_argument("root", nargs="?", default="logs", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--minimum-group-size", type=int, default=1)
    args = parser.parse_args()

    audit = build_history_audit(args.root)
    markdown = render_markdown(
        audit,
        minimum_group_size=max(1, args.minimum_group_size),
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

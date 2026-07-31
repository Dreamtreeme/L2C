"""자동 E2E 요약과 사람 판정표를 결합해 엄격 성공률을 계산한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

from benchmark.quality_eval import evaluate_collection_summary


ManualResult = Literal["pass", "fail", "unavailable"]
RunOutcome = Literal["success", "partial", "failure"]


class JobManualJudgement(BaseModel):
    url: str
    semantic_match: bool
    company_name: ManualResult
    job_title: ManualResult
    responsibilities: ManualResult
    requirements: ManualResult


class RunManualJudgement(BaseModel):
    run_id: str
    summary_path: str
    site: str
    query: str
    search_conditions_correct: bool
    count_handling_correct: bool
    no_out_of_scope_actions: bool
    wrong_target_count: int = Field(0, ge=0)
    no_effect_action_count: int = Field(0, ge=0)
    recovery_succeeded: bool | None = None
    human_review_sec: float = Field(0.0, ge=0)
    jobs: list[JobManualJudgement] = Field(default_factory=list)
    notes: str = ""


class EvaluationManifest(BaseModel):
    schema_version: int = 1
    commit_sha: str
    model_contract: dict[str, Any]
    environment_contract: dict[str, Any]
    runs: list[RunManualJudgement]


def _manual_contract_passed(judgement: RunManualJudgement) -> bool:
    if not (
        judgement.search_conditions_correct
        and judgement.count_handling_correct
        and judgement.no_out_of_scope_actions
        and judgement.jobs
    ):
        return False
    return all(
        job.semantic_match
        and job.company_name == "pass"
        and job.job_title == "pass"
        and job.responsibilities in {"pass", "unavailable"}
        and job.requirements in {"pass", "unavailable"}
        for job in judgement.jobs
    )


def _normalized_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    if parts.scheme.casefold() not in {"http", "https"} or not parts.netloc:
        return ""
    return urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            parts.path.rstrip("/"),
            "",
            "",
        )
    )


def _manual_coverage_passed(
    result: dict[str, Any],
    judgement: RunManualJudgement,
    resolved_count: int,
) -> bool:
    """자동 실행이 해결한 모든 공고를 사람 판정표가 한 번씩 포함하는지 확인한다."""

    judged_urls = [_normalized_url(job.url) for job in judgement.jobs]
    if (
        resolved_count <= 0
        or len(judged_urls) != resolved_count
        or any(not url for url in judged_urls)
        or len(set(judged_urls)) != len(judged_urls)
    ):
        return False
    validation = dict(result.get("persistence_validation") or {})
    persisted_urls = {
        _normalized_url(item.get("url"))
        for item in validation.get("persisted_items", []) or []
        if isinstance(item, dict) and _normalized_url(item.get("url"))
    }
    return persisted_urls.issubset(set(judged_urls))


def evaluate_manual_run(
    summary: dict[str, Any],
    judgement: RunManualJudgement,
) -> dict[str, Any]:
    """자동 저장 계약과 사람의 의미·본문 판정을 별도 표시한 뒤 결론을 낸다."""

    result = summary.get("result")
    if not isinstance(result, dict):
        result = summary
    automatic = evaluate_collection_summary(result)
    manual_coverage_passed = _manual_coverage_passed(
        result,
        judgement,
        automatic["resolved_count"],
    )
    manual_passed = (
        manual_coverage_passed
        and _manual_contract_passed(judgement)
    )
    if automatic["passed"] and manual_passed:
        outcome: RunOutcome = "success"
    elif automatic["resolved_count"] > 0 and manual_passed:
        outcome = "partial"
    else:
        outcome = "failure"
    return {
        "run_id": judgement.run_id,
        "site": judgement.site,
        "query": judgement.query,
        "outcome": outcome,
        "automatic_contract_passed": bool(automatic["passed"]),
        "manual_contract_passed": manual_passed,
        "automatic": automatic,
        "manual": {
            "search_conditions_correct": judgement.search_conditions_correct,
            "count_handling_correct": judgement.count_handling_correct,
            "no_out_of_scope_actions": judgement.no_out_of_scope_actions,
            "wrong_target_count": judgement.wrong_target_count,
            "no_effect_action_count": judgement.no_effect_action_count,
            "recovery_succeeded": judgement.recovery_succeeded,
            "human_review_sec": judgement.human_review_sec,
            "job_count": len(judgement.jobs),
            "coverage_passed": manual_coverage_passed,
        },
    }


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return round(ordered[index], 3)


def evaluate_manifest(
    manifest: EvaluationManifest,
    *,
    base_dir: Path,
) -> dict[str, Any]:
    runs = []
    execution_times = []
    for judgement in manifest.runs:
        summary_path = Path(judgement.summary_path)
        if not summary_path.is_absolute():
            summary_path = base_dir / summary_path
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluated = evaluate_manual_run(summary, judgement)
        evaluated["summary_path"] = str(summary_path)
        execution_time = summary.get("execution_time_sec")
        if execution_time is not None:
            execution_times.append(float(execution_time))
            evaluated["execution_time_sec"] = float(execution_time)
        runs.append(evaluated)

    outcomes = Counter(item["outcome"] for item in runs)
    total = len(runs)
    return {
        "schema_version": 1,
        "commit_sha": manifest.commit_sha,
        "model_contract": manifest.model_contract,
        "environment_contract": manifest.environment_contract,
        "run_count": total,
        "strict_success_rate": (
            round(outcomes["success"] / total, 6)
            if total
            else 0.0
        ),
        "partial_rate": (
            round(outcomes["partial"] / total, 6)
            if total
            else 0.0
        ),
        "failure_rate": (
            round(outcomes["failure"] / total, 6)
            if total
            else 0.0
        ),
        "outcome_counts": dict(outcomes),
        "execution_time_sec": {
            "min": min(execution_times) if execution_times else None,
            "median": (
                round(median(execution_times), 3)
                if execution_times
                else None
            ),
            "p95": _percentile(execution_times, 0.95),
            "max": max(execution_times) if execution_times else None,
        },
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="사람 판정표와 E2E summary로 엄격 성공률을 계산합니다.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = EvaluationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    report = evaluate_manifest(
        manifest,
        base_dir=manifest_path.parent,
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EvaluationManifest",
    "JobManualJudgement",
    "RunManualJudgement",
    "evaluate_manifest",
    "evaluate_manual_run",
]

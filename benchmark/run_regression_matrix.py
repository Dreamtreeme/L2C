"""물리 입력 충돌 없이 다중 사이트 E2E 행렬을 순차 실행하고 같은 지표로 요약한다."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.observability.reflex_paths import summarize_reflex_paths
from benchmark.quality_eval import evaluate_expected_source_urls


def _expand_scenarios(
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """같은 회차의 시나리오가 연이어 실행되도록 반복을 확장한다."""

    expanded = []
    repeats = [max(1, int(raw.get("repeat") or 1)) for raw in scenarios]
    for repeat_index in range(1, max(repeats, default=0) + 1):
        for raw, repeat in zip(scenarios, repeats, strict=True):
            if repeat_index > repeat:
                continue
            base_id = str(raw["id"])
            scenario = dict(raw)
            scenario["base_id"] = base_id
            scenario["repeat_index"] = repeat_index
            scenario["repeat_total"] = repeat
            scenario["id"] = f"{base_id}-r{repeat_index}" if repeat > 1 else base_id
            expanded.append(scenario)
    return expanded


def _git_execution_contract() -> dict[str, Any]:
    def run_git(*args: str, strip: bool = True) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT_DIR,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if strip else completed.stdout.rstrip("\r\n")

    status = run_git("status", "--porcelain", strip=False)
    return {
        "commit_sha": run_git("rev-parse", "HEAD"),
        "worktree_clean": not bool(status),
        "changed_paths": [line[3:] for line in status.splitlines() if len(line) >= 4],
    }


def _runtime_execution_contract() -> dict[str, Any]:
    from agent.llm.policy import (
        commander_model_name,
        lightweight_model_name,
        worker_reasoning_model_name,
    )
    from agent.config import get_settings

    settings = get_settings()
    environment_keys = (
        "COMMANDER_MODEL",
        "VISION_WORKER_REASONING_MODEL",
        "VISION_LIGHTWEIGHT_MODEL",
        "CHROME_WINDOW_WIDTH",
        "CHROME_WINDOW_HEIGHT",
        "VISION_BROWSER_WINDOW_WIDTH",
        "VISION_BROWSER_WINDOW_HEIGHT",
        "PADDLEOCR_USE_GPU",
        "PADDLE_OCR_MAX_DIM",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "resolved": {
            "commander_model": commander_model_name(),
            "worker_reasoning_model": worker_reasoning_model_name(),
            "lightweight_model": lightweight_model_name(),
            "vision_window_width": settings.browser.vision_window_width,
            "vision_window_height": settings.browser.vision_window_height,
            "ocr_use_gpu": settings.ocr.use_gpu,
            "ocr_max_dim": settings.ocr.max_image_dim,
            "run_deadline_sec": settings.execution.run_deadline_sec,
        },
        "environment": {key: os.environ.get(key, "") for key in environment_keys},
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return round(ordered[index], 6)


def _metric_summary(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(payload.get("metrics") or {})
    steps = [item for item in metrics.get("steps", []) if isinstance(item, dict)]
    ocr = [
        float(item.get("duration_sec") or 0.0)
        for item in steps
        if item.get("component") == "ocr_request"
    ]
    reasoning = [item for item in steps if item.get("stage") == "reasoning"]
    llm = dict(metrics.get("llm") or {})
    totals = dict(llm.get("totals") or {})
    outcome = dict(metrics.get("outcome") or {})
    quality = dict(payload.get("quality") or {})
    experience_preconditions = dict(
        payload.get("experience_guided_preconditions") or {}
    )
    reflex_paths = summarize_reflex_paths(steps)
    return {
        "status": payload.get("status"),
        "quality_passed": bool(quality.get("passed")),
        "collected_count": int(quality.get("collected_count") or 0),
        "persisted_count": int(quality.get("persisted_count") or 0),
        "observed_existing_count": int(quality.get("observed_existing_count") or 0),
        "resolved_count": int(quality.get("resolved_count") or 0),
        "execution_time_sec": payload.get("execution_time_sec"),
        "ocr_count": len(ocr),
        "ocr_time_sec": round(sum(ocr), 6),
        "ocr_p50_sec": _percentile(ocr, 0.50),
        "ocr_p95_sec": _percentile(ocr, 0.95),
        "ocr_startup_sec": round(
            sum(
                float(item.get("duration_sec") or 0.0)
                for item in steps
                if item.get("component") == "ocr_startup"
            ),
            6,
        ),
        "reasoning_count": len(reasoning),
        "reasoning_time_sec": round(
            sum(float(item.get("duration_sec") or 0.0) for item in reasoning), 6
        ),
        "reflex_count": sum(
            item.get("stage") == "reflex" and item.get("action_source") == "reflex"
            for item in steps
        ),
        **reflex_paths,
        "queue_count": sum(
            item.get("stage") == "selection"
            and item.get("action_source") == "job_card_queue"
            for item in steps
        ),
        "experience_guided_replay_ready": bool(
            experience_preconditions.get("replay_ready", True)
        ),
        "active_experience_rule_count": int(
            experience_preconditions.get("experience_rules") or 0
        ),
        "input_tokens": int(totals.get("input_tokens") or 0),
        "output_tokens": int(totals.get("output_tokens") or 0),
        "total_tokens": int(totals.get("total_tokens") or 0),
        "estimated_cost": (llm.get("cost") or {}).get("estimated_total"),
        "last_phase": outcome.get("last_phase", ""),
        "failure_stage": outcome.get("failure_stage", ""),
        "failure_code": outcome.get("failure_code", ""),
    }


def _target_contract_passed(result: dict[str, Any]) -> bool:
    contract = result.get("target_contract")
    return isinstance(contract, dict) and contract.get("passed") is True


def _command(scenario: dict[str, Any], log_path: Path, summary_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT_DIR / "benchmark" / "run_realtime_e2e.py"),
        "--site",
        str(scenario["site"]),
        "--search-keyword",
        str(scenario["search_keyword"]),
        "--target-count",
        str(max(0, int(scenario.get("target_count") or 0))),
        "--count-mode",
        str(scenario.get("count_mode") or "unspecified"),
        "--original-query",
        str(scenario.get("original_query") or scenario["search_keyword"]),
        "--scenario-id",
        str(scenario["id"]),
        "--execution-mode",
        str(scenario["execution_mode"]),
        "--experiment-name",
        "architecture-regression",
    ]
    for expected_url in scenario.get("expected_source_urls", []):
        command.extend(("--expected-source-url", str(expected_url)))
    command.extend(
        [
            "--log",
            str(log_path),
            "--summary",
            str(summary_path),
        ]
    )
    return command


def _scenario_environment(
    scenario: dict[str, Any],
    *,
    db_path: Path | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    if db_path is not None:
        env["DB_PATH"] = str(db_path)
    # 승격 시간은 수집 실행과 분리하고 부모 프로세스의 승격 작업자로 측정한다.
    env["VISION_RECIPE_AUTO_PROMOTE"] = "0"
    execution_mode = str(scenario["execution_mode"])
    if execution_mode == "autonomous":
        env["REFLEX_ENABLED"] = "0"
    elif execution_mode == "experience_guided":
        env["REFLEX_ENABLED"] = "1"
    return env


def _clear_jobs_for_collection_run(db_path: Path) -> int:
    """격리 DB의 레시피는 보존하고 공고만 지워 반복 작업량을 맞춘다."""

    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone()
        if not table_exists:
            return 0
        removed_count = int(
            connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        )
        connection.execute("DELETE FROM jobs")
        connection.commit()
    return removed_count


def _scenario_workload_key(scenario: dict[str, Any]) -> str:
    return json.dumps(
        {
            "site": str(scenario.get("site") or "").strip().casefold(),
            "search_keyword": str(scenario.get("search_keyword") or "")
            .strip()
            .casefold(),
            "target_count": max(0, int(scenario.get("target_count") or 0)),
            "count_mode": str(scenario.get("count_mode") or "unspecified"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _paired_autonomous_failed(
    scenario: dict[str, Any],
    autonomous_contracts: dict[str, bool],
) -> bool:
    return (
        str(scenario.get("execution_mode") or "") == "experience_guided"
        and autonomous_contracts.get(_scenario_pair_key(scenario)) is False
    )


def _scenario_pair_key(scenario: dict[str, Any]) -> str:
    return (
        f"{_scenario_workload_key(scenario)}"
        f"#repeat={int(scenario.get('repeat_index') or 1)}"
    )


def _promote_autonomous_candidate(
    payload: dict[str, Any],
    *,
    db_path: Path,
) -> dict[str, Any]:
    """자율 탐색 시간과 분리해 후보를 검토하고 경험 기반 탐색에 반영한다."""

    raw_result = payload.get("result")
    result: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
    run_id = str(result.get("worker_run_id") or "")
    if not run_id:
        return {"run_id": "", "promoted": False, "reason": "run_id_missing"}

    outcome = _process_candidate_until_settled(run_id, db_path=db_path)
    validation = dict(outcome.get("validation") or {})
    review = dict(validation.get("review") or {})
    promotion = dict(validation.get("promotion") or {})
    return {
        "run_id": run_id,
        "decision": str(review.get("decision") or ""),
        "review_status": str(outcome.get("review_status") or ""),
        "review_attempts": int(outcome.get("review_attempts") or 0),
        "review_error": str(outcome.get("review_error") or ""),
        "review_metrics": dict(outcome.get("review_metrics") or {}),
        "reason": "" if promotion.get("promoted") else "candidate_not_promoted",
        **promotion,
    }


def _aggregate_review_metrics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [dict(item.get("review_metrics") or {}) for item in attempts]
    costs = [
        float(item["estimated_cost"])
        for item in metrics
        if item.get("estimated_cost") is not None
    ]
    return {
        "attempt_count": len(attempts),
        "duration_sec": round(
            sum(float(item.get("duration_sec") or 0.0) for item in metrics), 6
        ),
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in metrics),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in metrics),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in metrics),
        "estimated_cost": round(sum(costs), 10) if costs else None,
    }


def _process_candidate_until_settled(
    run_id: str,
    *,
    db_path: Path,
) -> dict[str, Any]:
    """회귀 행렬에서 특정 후보의 재시도 결과만 동기적으로 수집한다."""

    from agent.application.recipe_promotion_worker import RecipePromotionWorker
    from agent.recipe.candidate_store import RecipeCandidateStore

    store = RecipeCandidateStore(db_path)
    candidate = store.get_candidate(run_id)
    if candidate is None:
        return {
            "review_status": "not_found",
            "review_attempts": 0,
            "review_error": "",
            "validation": {},
            "review_metrics": _aggregate_review_metrics([]),
        }
    if candidate.status == "recorded":
        store.enqueue_review(run_id)

    worker = RecipePromotionWorker(db_path, retry_delay_sec=0)
    attempts: list[dict[str, Any]] = []
    for _attempt in range(worker.max_attempts):
        candidate = store.get_candidate(run_id)
        if candidate is None or candidate.status != "pending_review":
            break
        result = worker.process_one(run_id)
        if result is None:
            break
        attempts.append(dict(result))

    candidate = store.get_candidate(run_id)
    return {
        "review_status": candidate.status if candidate else "not_found",
        "review_attempts": candidate.review_attempts if candidate else 0,
        "review_error": candidate.review_error if candidate else "",
        "validation": candidate.validation if candidate else {},
        "review_metrics": _aggregate_review_metrics(attempts),
    }


def _attach_promotion_metrics(
    metric_summary: dict[str, Any],
    promotion: dict[str, Any],
) -> dict[str, Any]:
    review_metrics = dict(promotion.get("review_metrics") or {})
    collection_cost = metric_summary.get("estimated_cost")
    promotion_cost = review_metrics.get("estimated_cost")
    numeric_costs = [
        float(value) for value in (collection_cost, promotion_cost) if value is not None
    ]
    return {
        **metric_summary,
        "promotion_time_sec": float(review_metrics.get("duration_sec") or 0.0),
        "promotion_total_tokens": int(review_metrics.get("total_tokens") or 0),
        "promotion_estimated_cost": promotion_cost,
        "workflow_total_tokens": (
            int(metric_summary.get("total_tokens") or 0)
            + int(review_metrics.get("total_tokens") or 0)
        ),
        "workflow_estimated_cost": round(sum(numeric_costs), 10)
        if numeric_costs
        else None,
    }


def _experience_reuse_effectiveness(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """경험 기반 실행에서 실제로 생략한 판단과 폴백을 집계한다."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        scenario = dict(item.get("scenario") or {})
        if str(scenario.get("execution_mode") or "") != "experience_guided":
            continue
        workload = _scenario_workload_key(scenario)
        metrics = dict(item.get("metrics") or {})
        validated = all(
            (
                metrics.get("quality_passed"),
                item.get("mode_contract_passed"),
                _target_contract_passed(item),
                metrics.get("experience_guided_replay_ready"),
            )
        )
        grouped.setdefault(workload, []).append(
            {
                "repeat_index": int(scenario.get("repeat_index") or 1),
                "validated": validated,
                "reasoning_call_reduction": int(
                    metrics.get("reflex_reasoning_call_reduction") or 0
                ),
                "source_reasoning_replaced_count": int(
                    metrics.get("reflex_source_reasoning_replaced_count") or 0
                ),
                "resolver_reasoning_call_count": int(
                    metrics.get("reflex_resolver_reasoning_call_count") or 0
                ),
                "reflex_path_started_count": int(
                    metrics.get("reflex_path_started_count") or 0
                ),
                "reflex_path_completed_count": int(
                    metrics.get("reflex_path_completed_count") or 0
                ),
                "reflex_path_failed_count": int(
                    metrics.get("reflex_path_failed_count") or 0
                ),
                "reflex_path_fallback_count": int(
                    metrics.get("reflex_path_fallback_count") or 0
                ),
            }
        )

    summaries = []
    for workload, runs in grouped.items():
        validated_runs = [item for item in runs if item["validated"]]
        started = sum(item["reflex_path_started_count"] for item in runs)
        completed = sum(item["reflex_path_completed_count"] for item in runs)
        summaries.append(
            {
                "workload": json.loads(workload),
                "experience_run_count": len(runs),
                "validated_run_count": len(validated_runs),
                "validated_reasoning_call_reduction": sum(
                    item["reasoning_call_reduction"] for item in validated_runs
                ),
                "validated_source_reasoning_replaced_count": sum(
                    item["source_reasoning_replaced_count"]
                    for item in validated_runs
                ),
                "validated_resolver_reasoning_call_count": sum(
                    item["resolver_reasoning_call_count"]
                    for item in validated_runs
                ),
                "reflex_path_started_count": started,
                "reflex_path_completed_count": completed,
                "reflex_path_failed_count": sum(
                    item["reflex_path_failed_count"] for item in runs
                ),
                "reflex_path_fallback_count": sum(
                    item["reflex_path_fallback_count"] for item in runs
                ),
                "reflex_path_completion_rate": (
                    round(completed / started, 6) if started else None
                ),
                "runs": runs,
            }
        )
    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run L2C vision E2E scenarios sequentially."
    )
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--db-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _selected_matrix_scenarios(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    matrix_path = Path(args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    require_recipe_promotion = bool(matrix.get("require_recipe_promotion", False))
    scenarios = _expand_scenarios(
        [
            {
                **item,
                "require_recipe_promotion": bool(
                    item.get("require_recipe_promotion", require_recipe_promotion)
                ),
            }
            for item in matrix.get("scenarios", [])
            if isinstance(item, dict)
        ]
    )
    selected = set(args.scenario)
    if selected:
        scenarios = [
            item
            for item in scenarios
            if str(item.get("id")) in selected or str(item.get("base_id")) in selected
        ]
    if not scenarios:
        raise SystemExit("실행할 E2E 시나리오가 없습니다.")
    return matrix_path, matrix, scenarios


def _scenario_commands(
    scenarios: list[dict[str, Any]],
    output_dir: Path,
) -> list[list[str]]:
    return [
        _command(
            scenario,
            output_dir / f"{scenario['id']}.log",
            output_dir / f"{scenario['id']}.summary.json",
        )
        for scenario in scenarios
    ]


def _load_process_summary(summary_path: Path, return_code: int) -> dict[str, Any]:
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {"status": "missing_summary", "error": f"exit_code={return_code}"}


def _promotion_result(
    scenario: dict[str, Any],
    payload: dict[str, Any],
    *,
    return_code: int,
    db_path: Path,
) -> dict[str, Any]:
    if (
        not scenario.get("require_recipe_promotion")
        or str(scenario.get("execution_mode") or "") != "autonomous"
        or return_code != 0
        or payload.get("status") != "completed"
    ):
        return {}
    try:
        return _promote_autonomous_candidate(payload, db_path=db_path)
    except Exception as exc:
        return {
            "promoted": False,
            "reason": "promotion_failed",
            "error": str(exc)[:300],
        }


def _mode_contract_passed(
    scenario: dict[str, Any],
    metrics: dict[str, Any],
    promotion: dict[str, Any],
) -> bool:
    execution_mode = str(scenario["execution_mode"])
    if execution_mode == "autonomous":
        if not scenario.get("require_recipe_promotion"):
            return True
        return bool(promotion.get("promoted"))
    if execution_mode == "experience_guided":
        return (
            metrics["reflex_path_completed_count"] > 0
            and metrics["experience_guided_replay_ready"]
        )
    return True


def _run_scenario_process(
    scenario: dict[str, Any],
    command: list[str],
    db_path: Path,
) -> dict[str, Any]:
    reset_count = _clear_jobs_for_collection_run(db_path)
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        check=False,
        env=_scenario_environment(scenario, db_path=db_path),
    )
    summary_path = Path(command[-1])
    payload = _load_process_summary(summary_path, completed.returncode)
    target_contract = dict(payload.get("target_contract") or {})
    promotion = (
        _promotion_result(
            scenario,
            payload,
            return_code=completed.returncode,
            db_path=db_path,
        )
        if target_contract.get("passed") is True
        else {}
    )
    metrics = _attach_promotion_metrics(_metric_summary(payload), promotion)
    return {
        "scenario": scenario,
        "process_exit_code": completed.returncode,
        "summary_path": str(summary_path),
        "metrics": metrics,
        "promotion": promotion,
        "target_contract": target_contract,
        "job_reset_count": reset_count,
        "mode_contract_passed": _mode_contract_passed(
            scenario,
            metrics,
            promotion,
        ),
        "skipped_reason": "",
    }


def _skipped_experience_guided_result(
    scenario: dict[str, Any],
    command: list[str],
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "process_exit_code": None,
        "summary_path": str(Path(command[-1])),
        "metrics": _attach_promotion_metrics(
            _metric_summary({"status": "skipped"}),
            {},
        ),
        "promotion": {},
        "target_contract": evaluate_expected_source_urls(
            scenario.get("expected_source_urls", []),
            [],
        ),
        "job_reset_count": 0,
        "mode_contract_passed": False,
        "skipped_reason": "paired_autonomous_promotion_failed",
    }


def _result_passed(result: dict[str, Any]) -> bool:
    return bool(
        result["process_exit_code"] == 0
        and result["metrics"]["quality_passed"]
        and result["mode_contract_passed"]
        and _target_contract_passed(result)
    )


def _run_scenario_matrix(
    scenarios: list[dict[str, Any]],
    commands: list[list[str]],
    db_path: Path,
    *,
    fail_fast: bool,
) -> tuple[list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    autonomous_contracts: dict[str, bool] = {}
    exit_code = 0
    for scenario, command in zip(scenarios, commands, strict=True):
        if _paired_autonomous_failed(scenario, autonomous_contracts):
            result = _skipped_experience_guided_result(scenario, command)
        else:
            result = _run_scenario_process(scenario, command, db_path)
        results.append(result)
        if str(scenario["execution_mode"]) == "autonomous":
            autonomous_contracts[_scenario_pair_key(scenario)] = _result_passed(result)
        if not _result_passed(result):
            exit_code = 1
            if fail_fast:
                break
    return results, exit_code


def main() -> int:
    args = _parse_args()
    matrix_path, matrix, scenarios = _selected_matrix_scenarios(args)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir or ROOT_DIR / "logs" / f"regression_{stamp}"
    ).resolve()
    db_path = (
        Path(args.db_path).resolve() if args.db_path else output_dir / "regression.db"
    )
    commands = _scenario_commands(scenarios, output_dir)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "git": _git_execution_contract(),
                    "runtime": _runtime_execution_contract(),
                    "commands": commands,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    git_contract = _git_execution_contract()
    if matrix.get("require_clean_worktree") and not git_contract["worktree_clean"]:
        raise SystemExit("이 평가 행렬은 깨끗한 작업 트리에서만 실행할 수 있습니다.")

    output_dir.mkdir(parents=True, exist_ok=False)
    results, exit_code = _run_scenario_matrix(
        scenarios,
        commands,
        db_path,
        fail_fast=args.fail_fast,
    )

    aggregate = {
        "schema_version": 3,
        "matrix": str(matrix_path),
        "db_path": str(db_path),
        "created_at": datetime.now().astimezone().isoformat(),
        "git": git_contract,
        "runtime": _runtime_execution_contract(),
        "passed": all(_result_passed(item) for item in results),
        "experience_reuse_effectiveness": _experience_reuse_effectiveness(results),
        "results": results,
    }
    aggregate_path = output_dir / "matrix.summary.json"
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"MATRIX_SUMMARY={aggregate_path}")
    return exit_code if aggregate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

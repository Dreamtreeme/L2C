"""물리 입력 충돌 없이 다중 사이트 E2E 행렬을 순차 실행하고 같은 지표로 요약한다."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = Path(__file__).with_name("e2e_regression_matrix.json")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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
    action_sources = [
        str(item.get("action_source") or "")
        for item in steps
        if item.get("stage") in {"action", "selection", "reflex"}
    ]
    llm = dict(metrics.get("llm") or {})
    totals = dict(llm.get("totals") or {})
    outcome = dict(metrics.get("outcome") or {})
    quality = dict(payload.get("quality") or {})
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
        "reasoning_time_sec": round(sum(float(item.get("duration_sec") or 0.0) for item in reasoning), 6),
        "reflex_count": sum(source == "reflex" for source in action_sources),
        "queue_count": sum(source == "card_queue" for source in action_sources),
        "input_tokens": int(totals.get("input_tokens") or 0),
        "output_tokens": int(totals.get("output_tokens") or 0),
        "total_tokens": int(totals.get("total_tokens") or 0),
        "estimated_cost": (llm.get("cost") or {}).get("estimated_total"),
        "last_phase": outcome.get("last_phase", ""),
        "failure_stage": outcome.get("failure_stage", ""),
        "failure_code": outcome.get("failure_code", ""),
    }


def _command(scenario: dict[str, Any], log_path: Path, summary_path: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT_DIR / "benchmark" / "run_realtime_e2e.py"),
        "--site",
        str(scenario["site"]),
        "--query",
        str(scenario["query"]),
        "--target-count",
        str(max(0, int(scenario.get("target_count") or 0))),
        "--count-mode",
        str(scenario.get("count_mode") or "unspecified"),
        "--original-query",
        str(scenario.get("original_query") or scenario["query"]),
        "--scenario-id",
        str(scenario["id"]),
        "--run-mode",
        str(scenario.get("run_mode") or "unspecified"),
        "--experiment-name",
        "architecture-regression",
        "--log",
        str(log_path),
        "--summary",
        str(summary_path),
    ]


def _scenario_environment(
    scenario: dict[str, Any],
    *,
    db_path: Path | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    if db_path is not None:
        env["DB_PATH"] = str(db_path)
    run_mode = str(scenario.get("run_mode") or "unspecified")
    if run_mode == "cold":
        env["REFLEX_ENABLED"] = "0"
    elif run_mode == "warm":
        env["REFLEX_ENABLED"] = "1"
    return env


def _clear_jobs_for_warm_run(db_path: Path) -> int:
    """격리 DB의 레시피는 보존하고 공고만 지워 동일 작업량을 만든다."""

    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone()
        if not table_exists:
            return 0
        removed_count = int(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
        connection.execute("DELETE FROM jobs")
        connection.commit()
    return removed_count


def _promote_cold_candidate(
    payload: dict[str, Any],
    *,
    db_path: Path,
) -> dict[str, Any]:
    """cold 실행 시간과 분리해 후보를 검토하고 다음 warm 실행에 반영한다."""

    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    candidate_id = str(result.get("submission_id") or "")
    if not candidate_id:
        return {"candidate_id": "", "promoted": False, "reason": "candidate_id_missing"}

    from agent.recipe.candidate_reviewer import review_and_apply_candidate

    review = review_and_apply_candidate(
        candidate_id,
        db_path=db_path,
        mode="promote",
        raise_on_critic_error=True,
    )
    promotion = dict(review.get("promotion") or {})
    return {
        "candidate_id": candidate_id,
        "decision": str(review.get("decision") or ""),
        **promotion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run L2C vision E2E scenarios sequentially.")
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--db-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    matrix_path = Path(args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    scenarios = [item for item in matrix.get("scenarios", []) if isinstance(item, dict)]
    selected = set(args.scenario)
    if selected:
        scenarios = [item for item in scenarios if str(item.get("id")) in selected]
    if not scenarios:
        raise SystemExit("실행할 E2E 시나리오가 없습니다.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or ROOT_DIR / "logs" / f"regression_{stamp}").resolve()
    db_path = Path(args.db_path).resolve() if args.db_path else output_dir / "regression.db"
    commands = []
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        commands.append(
            _command(
                scenario,
                output_dir / f"{scenario_id}.log",
                output_dir / f"{scenario_id}.summary.json",
            )
        )
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), "commands": commands}, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    exit_code = 0
    for scenario, command in zip(scenarios, commands):
        run_mode = str(scenario.get("run_mode") or "unspecified")
        warm_reset_count = (
            _clear_jobs_for_warm_run(db_path)
            if run_mode == "warm"
            else 0
        )
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            check=False,
            env=_scenario_environment(scenario, db_path=db_path),
        )
        summary_path = Path(command[-1])
        payload = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {"status": "missing_summary", "error": f"exit_code={completed.returncode}"}
        )
        promotion: dict[str, Any] = {}
        if (
            str(scenario.get("run_mode") or "") == "cold"
            and completed.returncode == 0
            and payload.get("status") == "completed"
        ):
            try:
                promotion = _promote_cold_candidate(payload, db_path=db_path)
            except Exception as exc:
                promotion = {
                    "promoted": False,
                    "reason": "promotion_failed",
                    "error": str(exc)[:300],
                }
        metric_summary = _metric_summary(payload)
        mode_contract_passed = True
        if run_mode == "cold":
            target_count = max(0, int(scenario.get("target_count") or 0))
            mode_contract_passed = (
                metric_summary["persisted_count"] >= target_count
                and bool(promotion.get("promoted"))
            )
        elif run_mode == "warm":
            target_count = max(0, int(scenario.get("target_count") or 0))
            mode_contract_passed = (
                metric_summary["persisted_count"] >= target_count
                and metric_summary["reflex_count"] > 0
            )

        results.append(
            {
                "scenario": scenario,
                "process_exit_code": completed.returncode,
                "summary_path": str(summary_path),
                "metrics": metric_summary,
                "promotion": promotion,
                "warm_reset_count": warm_reset_count,
                "mode_contract_passed": mode_contract_passed,
            }
        )
        if completed.returncode != 0:
            exit_code = 1
            if args.fail_fast:
                break

    aggregate = {
        "schema_version": 1,
        "matrix": str(matrix_path),
        "db_path": str(db_path),
        "created_at": datetime.now().astimezone().isoformat(),
        "passed": all(
            item["process_exit_code"] == 0
            and item["metrics"]["quality_passed"]
            and item["mode_contract_passed"]
            for item in results
        ),
        "results": results,
    }
    aggregate_path = output_dir / "matrix.summary.json"
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MATRIX_SUMMARY={aggregate_path}")
    return exit_code if aggregate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

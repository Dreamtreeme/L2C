"""Realtime worker E2E runner with stable file logging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class _Tee:
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _git_revision() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
        return commit, dirty
    except OSError:
        return "", False


def _runtime_config() -> dict[str, str]:
    from agent.application.model_policy import (
        DEFAULT_COMMANDER_MODEL,
        DEFAULT_LIGHTWEIGHT_MODEL,
        DEFAULT_WORKER_REASONING_THINKING_LEVEL,
    )

    defaults = {
        "COMMANDER_MODEL": DEFAULT_COMMANDER_MODEL,
        "VISION_LIGHTWEIGHT_MODEL": DEFAULT_LIGHTWEIGHT_MODEL,
        "VISION_WORKER_REASONING_MODEL": DEFAULT_COMMANDER_MODEL,
        "VISION_WORKER_REASONING_THINKING_LEVEL": DEFAULT_WORKER_REASONING_THINKING_LEVEL,
        "VISION_DETAIL_FINAL_EXTRACTION_MODEL": DEFAULT_LIGHTWEIGHT_MODEL,
        "VISION_SEARCH_INTENT_MODEL": DEFAULT_LIGHTWEIGHT_MODEL,
        "VISION_WORKER_REVIEW_MODEL": DEFAULT_COMMANDER_MODEL,
        "VISION_LIGHTWEIGHT_MAX_OUTPUT_TOKENS": "1536",
        "SOM_OCR_MAX_DIM": "1152",
        "SOM_OCR_REQUEST_TIMEOUT_SEC": "20",
        "REFLEX_ENABLED": "",
        "VISION_RECIPE_AUTO_PROMOTE": "",
        "VISION_RECIPE_CRITIC_EVIDENCE_TEXT_LIMIT": "",
        "VISION_BROWSER_WINDOW_SIZE": "",
        "VISION_BROWSER_WINDOW_WIDTH": "",
        "VISION_BROWSER_WINDOW_HEIGHT": "",
        "VISION_REASONING_SCREEN_GUARD": "",
    }
    return {
        key: os.getenv(key, "").strip() or default
        for key, default in defaults.items()
    }


def _config_fingerprint(config: dict[str, str]) -> str:
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


def _apply_run_mode_environment(run_mode: str) -> None:
    """E2E 실행 모드가 실제 Reflex 경로를 결정하게 한다."""

    if run_mode == "cold":
        os.environ["REFLEX_ENABLED"] = "0"
    elif run_mode == "warm":
        os.environ["REFLEX_ENABLED"] = "1"


def _warm_run_preconditions(
    run_mode: str,
    site: str,
) -> dict[str, object]:
    """warm 성능 비교에 필요한 활성 레시피 상태를 실행 전에 고정합니다."""

    if run_mode != "warm":
        return {
            "required": False,
            "performance_comparable": True,
            "reasons": [],
        }
    try:
        from agent.recipe.store import RecipeStore

        counts = RecipeStore().active_counts(site)
    except Exception as exc:
        return {
            "required": True,
            "site": site,
            "performance_comparable": False,
            "reasons": ["active_recipe_lookup_failed"],
            "error": str(exc)[:200],
        }
    reasons = []
    if int(counts.get("roi_recipes") or 0) <= 0:
        reasons.append("active_roi_recipe_missing")
    return {
        "required": True,
        "site": site,
        **counts,
        "performance_comparable": not reasons,
        "reasons": reasons,
    }


def _finalize_warm_preconditions(
    preconditions: dict[str, object],
    quality: dict[str, object],
) -> dict[str, object]:
    """기존 DB 공고가 섞인 warm 실행을 성능 비교 대상에서 제외합니다."""

    out = dict(preconditions)
    if not out.get("required"):
        return out
    reasons = list(out.get("reasons") or [])
    observed_existing_count = int(
        quality.get("observed_existing_count") or 0
    )
    if observed_existing_count > 0:
        reasons.append("existing_jobs_observed")
    out["observed_existing_count"] = observed_existing_count
    out["reasons"] = list(dict.fromkeys(reasons))
    out["performance_comparable"] = not out["reasons"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run realtime_scraping E2E and tee stdout/stderr to a log file.")
    parser.add_argument("--site", default="wanted")
    parser.add_argument("--query", required=True)
    parser.add_argument("--target-count", type=int, default=0)
    parser.add_argument(
        "--count-mode",
        choices=("unspecified", "explicit", "visible_all"),
        default="unspecified",
    )
    parser.add_argument("--original-query", default="")
    parser.add_argument("--scenario-id", default="manual")
    parser.add_argument(
        "--run-mode",
        choices=("unspecified", "cold", "warm"),
        default="unspecified",
    )
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--recipe-version", default="")
    parser.add_argument("--log", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else log_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise SystemExit(f"log already exists: {log_path}")
    if summary_path.exists():
        raise SystemExit(f"summary already exists: {summary_path}")

    _apply_run_mode_environment(args.run_mode)
    commit, git_dirty = _git_revision()
    runtime_config = _runtime_config()
    config_fingerprint = _config_fingerprint(runtime_config)
    experiment_name = (
        args.experiment_name
        or os.getenv("L2C_E2E_EXPERIMENT", "manual")
        or "manual"
    )
    recipe_version = args.recipe_version or os.getenv("VISION_RECIPE_VERSION", "")
    warm_preconditions = _warm_run_preconditions(
        args.run_mode,
        args.site,
    )
    configured_models = sorted(
        {
            value
            for key, value in runtime_config.items()
            if key.endswith("MODEL") and value
        }
    )

    with log_path.open("x", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _Tee(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _Tee(original_stderr, log_file)  # type: ignore[assignment]
        vision_runtime = None
        try:
            print(
                "WARM_PRECONDITIONS="
                + json.dumps(warm_preconditions, ensure_ascii=False)
            )
            from agent.application.run_context import run_context
            from agent.application.run_contracts import RunPhase, RunStatus
            from agent.observability.langsmith_adapter import publish_langsmith_feedback
            from agent.runtime.vision_worker_runtime import VisionWorkerRuntime
            from agent.tools.realtime_scraping import build_runtime_realtime_scraping_tool
            from benchmark.e2e_observability import (
                build_e2e_observability,
                build_langsmith_feedback,
            )
            from benchmark.quality_eval import evaluate_collection_summary

            vision_runtime = VisionWorkerRuntime()
            collection_tool = build_runtime_realtime_scraping_tool(vision_runtime)
            events = []
            status = "failed"
            result = ""
            error = ""
            quality = {}
            recipe_promotion = {}
            started_at = datetime.now(timezone.utc).isoformat()
            with run_context(
                query=args.query,
                event_sink=events.append,
                prefix="e2e",
                metadata={
                    "scenario_id": args.scenario_id,
                    "experiment_name": experiment_name,
                    "site": args.site,
                    "target_count": max(0, args.target_count),
                    "count_mode": args.count_mode,
                    "run_mode": args.run_mode,
                    "git_commit": commit,
                    "git_dirty": git_dirty,
                    "config_fingerprint": config_fingerprint,
                    "recipe_version": recipe_version,
                    "models": configured_models,
                },
                tags=[
                    "e2e",
                    f"site:{args.site}",
                    f"scenario:{args.scenario_id}",
                    f"mode:{args.run_mode}",
                ],
            ) as (context, _created):
                start = time.perf_counter()
                try:
                    result = collection_tool.invoke(
                        {
                            "site": args.site,
                            "query": args.query,
                            "target_count": max(0, args.target_count),
                            "count_mode": args.count_mode,
                            "original_query": args.original_query or args.query,
                        }
                    )
                    try:
                        parsed_during_run = json.loads(result) if isinstance(result, str) else result
                    except json.JSONDecodeError:
                        parsed_during_run = {"raw": str(result)}
                    quality = evaluate_collection_summary(parsed_during_run)
                    status = "completed"
                    context.emit(
                        "run_completed",
                        RunPhase.COMPLETED,
                        "E2E 수집을 완료했습니다.",
                        status=RunStatus.COMPLETED,
                    )
                except Exception as exc:
                    error = str(exc)
                    context.emit(
                        "run_failed",
                        RunPhase.FAILED,
                        "E2E 수집에 실패했습니다.",
                        status=RunStatus.FAILED,
                        data={"error": error[:300]},
                    )
                elapsed = time.perf_counter() - start
                if status == "completed" and quality.get("passed"):
                    context.set_outcome("success")
                elif status == "completed":
                    context.set_outcome(
                        "partial",
                        failure_stage="quality_gate",
                        failure_code="quality_not_passed",
                    )
                metrics = context.snapshot()
            if status == "completed" and isinstance(parsed_during_run, dict):
                candidate_id = str(parsed_during_run.get("submission_id") or "")
                if candidate_id:
                    from agent.application.recipe_promotion_service import (
                        get_recipe_candidate_promotion_status,
                    )

                    recipe_promotion = get_recipe_candidate_promotion_status(candidate_id)
                    print(
                        "RECIPE_PROMOTION="
                        + json.dumps(recipe_promotion, ensure_ascii=False)
                    )
            if isinstance(result, str):
                print(result)
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"EXECUTION_TIME_SEC={elapsed:.3f}")
            print(f"LOG_TARGET={log_path}")

            try:
                parsed_result = json.loads(result) if isinstance(result, str) else result
            except json.JSONDecodeError:
                parsed_result = {"raw": str(result)}
            final_quality = quality or evaluate_collection_summary(
                parsed_result
            )
            warm_preconditions = _finalize_warm_preconditions(
                warm_preconditions,
                final_quality,
            )
            summary = {
                "schema_version": 2,
                "run_id": context.run_id,
                "status": status,
                "error": error,
                "started_at": started_at,
                "scenario_id": args.scenario_id,
                "experiment_name": experiment_name,
                "run_mode": args.run_mode,
                "site": args.site,
                "query": args.query,
                "target_count": max(0, args.target_count),
                "count_mode": args.count_mode,
                "original_query": args.original_query or args.query,
                "execution_time_sec": round(elapsed, 6),
                "git_commit": commit,
                "git_dirty": git_dirty,
                "config_fingerprint": config_fingerprint,
                "recipe_version": recipe_version,
                "models": configured_models,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "config": runtime_config,
                "metrics": metrics,
                "events": [event.model_dump(mode="json") for event in events],
                "quality": final_quality,
                "warm_preconditions": warm_preconditions,
                "recipe_promotion": recipe_promotion,
                "result": parsed_result,
            }
            observability = build_e2e_observability(summary)
            summary["observability"] = observability
            trace_id = str((metrics.get("langsmith") or {}).get("trace_id") or "")
            summary["langsmith_feedback"] = publish_langsmith_feedback(
                trace_id,
                build_langsmith_feedback(observability),
            )
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"SUMMARY_TARGET={summary_path}")
            if status != "completed":
                return 1
            if not summary["quality"].get("passed"):
                return 2
        finally:
            if vision_runtime is not None:
                vision_runtime.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

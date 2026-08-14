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
    from agent.config import get_settings
    from agent.llm.policy import (
        commander_model_name,
        lightweight_model_name,
        worker_reasoning_model_name,
    )

    settings = get_settings()
    commander_model = commander_model_name()
    lightweight_model = lightweight_model_name()
    defaults = {
        "COMMANDER_MODEL": commander_model,
        "VISION_LIGHTWEIGHT_MODEL": lightweight_model,
        "VISION_WORKER_REASONING_MODEL": worker_reasoning_model_name(),
        "VISION_WORKER_REASONING_THINKING_LEVEL": (
            settings.models.worker_reasoning_thinking_level
        ),
        "VISION_DETAIL_FINAL_EXTRACTION_MODEL": (
            settings.models.detail_final_extraction_model or lightweight_model
        ),
        "VISION_RECIPE_CRITIC_MODEL": (
            settings.models.recipe_critic_model or commander_model
        ),
        "VISION_LIGHTWEIGHT_MAX_OUTPUT_TOKENS": str(
            settings.models.lightweight_max_output_tokens
        ),
        "PADDLE_OCR_MAX_DIM": str(settings.ocr.max_image_dim),
        "PADDLE_OCR_REQUEST_TIMEOUT_SEC": str(settings.ocr.request_timeout_sec),
        "REFLEX_ENABLED": str(int(settings.reflex.enabled)),
        "VISION_RECIPE_AUTO_PROMOTE": str(int(settings.recipe.auto_promote)),
        "VISION_RECIPE_CRITIC_EVIDENCE_TEXT_LIMIT": str(
            settings.recipe.critic_evidence_text_limit
        ),
        "VISION_BROWSER_WINDOW_WIDTH": str(settings.browser.vision_window_width),
        "VISION_BROWSER_WINDOW_HEIGHT": str(settings.browser.vision_window_height),
        "VISION_LOADING_TIMEOUT_SEC": str(settings.vision.loading_timeout_sec),
        "VISION_LOADING_BLANK_MAX_STDDEV": str(
            settings.vision.loading_blank_max_stddev
        ),
        "VISION_LOADING_BLANK_MAX_EDGE_MEAN": str(
            settings.vision.loading_blank_max_edge_mean
        ),
        "VISION_LOADING_BLANK_MIN_DOMINANT_RATIO": str(
            settings.vision.loading_blank_min_dominant_ratio
        ),
        "VISION_LOADING_CHECK_INTERVAL_SEC": str(
            settings.vision.loading_check_interval_sec
        ),
        "VISION_LOADING_MOTION_THRESHOLD_PERCENT": str(
            settings.vision.loading_motion_threshold_percent
        ),
        "VISION_LOADING_STABLE_FRAMES": str(settings.vision.loading_stable_frames),
    }
    return {
        key: os.getenv(key, "").strip() or default for key, default in defaults.items()
    }


def _config_fingerprint(config: dict[str, str]) -> str:
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


def _apply_execution_mode_environment(execution_mode: str) -> None:
    """탐색 방식에 따라 실제 Reflex 경로를 결정한다."""

    if execution_mode == "autonomous":
        os.environ["REFLEX_ENABLED"] = "0"
    elif execution_mode == "experience_guided":
        os.environ["REFLEX_ENABLED"] = "1"


def _experience_guided_preconditions(
    execution_mode: str,
    site: str,
    *,
    db_path: Path,
) -> dict[str, object]:
    """경험 기반 탐색에 필요한 활성 레시피 상태를 실행 전에 고정한다."""

    if execution_mode != "experience_guided":
        return {
            "required": False,
            "performance_comparable": True,
            "reasons": [],
        }
    try:
        from agent.recipe.store import RecipeStore

        counts = RecipeStore(db_path).active_counts(site)
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


def _finalize_experience_guided_preconditions(
    preconditions: dict[str, object],
    quality: dict[str, object],
) -> dict[str, object]:
    """기존 DB 공고가 섞인 경험 기반 탐색을 성능 비교에서 제외한다."""

    out = dict(preconditions)
    if not out.get("required"):
        return out
    reasons = list(out.get("reasons") or [])
    observed_existing_count = int(quality.get("observed_existing_count") or 0)
    if observed_existing_count > 0:
        reasons.append("existing_jobs_observed")
    out["observed_existing_count"] = observed_existing_count
    out["reasons"] = list(dict.fromkeys(reasons))
    out["performance_comparable"] = not out["reasons"]
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run realtime_scraping E2E and tee stdout/stderr to a log file."
    )
    parser.add_argument("--site", default="wanted")
    parser.add_argument("--search-keyword", required=True)
    parser.add_argument("--target-count", type=int, default=0)
    parser.add_argument(
        "--count-mode",
        choices=("unspecified", "explicit", "visible_all"),
        default="unspecified",
    )
    parser.add_argument("--original-query", default="")
    parser.add_argument("--scenario-id", default="manual")
    parser.add_argument(
        "--execution-mode",
        choices=("autonomous", "experience_guided"),
        required=True,
    )
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--recipe-version", default="")
    parser.add_argument("--expected-source-url", action="append", default=[])
    parser.add_argument("--log", required=True)
    parser.add_argument("--summary", default="")
    return parser.parse_args()


def _artifact_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary) if args.summary else log_path.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    for path, label in ((log_path, "log"), (summary_path, "summary")):
        if path.exists():
            raise SystemExit(f"{label} already exists: {path}")
    return log_path, summary_path


def _execution_identity(args: argparse.Namespace) -> dict[str, object]:
    commit, git_dirty = _git_revision()
    runtime_config = _runtime_config()
    return {
        "commit": commit,
        "git_dirty": git_dirty,
        "runtime_config": runtime_config,
        "config_fingerprint": _config_fingerprint(runtime_config),
        "experiment_name": (
            args.experiment_name
            or os.getenv("L2C_E2E_EXPERIMENT", "manual")
            or "manual"
        ),
        "recipe_version": (
            args.recipe_version or os.getenv("VISION_RECIPE_VERSION", "")
        ),
        "configured_models": sorted(
            {
                value
                for key, value in runtime_config.items()
                if key.endswith("MODEL") and value
            }
        ),
    }


def _execute_collection(
    args: argparse.Namespace,
    worker_service: object,
    context: object,
    *,
    db_path: Path,
) -> dict[str, object]:
    from agent.application.collection_experience import record_collection_experience
    from agent.application.collection_postprocessing import postprocess_collection_batch
    from agent.application.collection_storage import store_postprocessed_collection
    from agent.graph.investigation_collection_nodes import build_collection_result
    from agent.observability.run_contracts import RunPhase
    from benchmark.quality_eval import evaluate_collection_summary
    from shared.schema.collection_intent import CollectionIntent
    from shared.schema.run_schema import RunStatus

    status = "failed"
    error = ""
    parsed_result: dict[str, object] = {}
    quality: dict[str, object] = {}
    started = time.perf_counter()
    try:
        intent = CollectionIntent(
            site=args.site,
            search_keyword=args.search_keyword,
            target_count=max(0, args.target_count),
            count_mode=args.count_mode,
            original_query=args.original_query or args.search_keyword,
        )
        batch = worker_service.run(intent)
        processed = postprocess_collection_batch(batch)
        persistence = store_postprocessed_collection(processed, db_path=db_path)
        experience = record_collection_experience(
            batch,
            persistence,
            db_path=db_path,
        )
        result = build_collection_result(batch, persistence, experience)
        parsed_result = result.model_dump(mode="json")
        quality = evaluate_collection_summary(
            parsed_result,
            expected_source_urls=args.expected_source_url,
        )
        if result.status == "failed":
            error = result.message or result.error_code
            context.emit(
                "run_failed",
                RunPhase.FAILED,
                "E2E 수집에 실패했습니다.",
                status=RunStatus.FAILED,
                data={"error": error[:300]},
            )
        else:
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
    if status == "completed" and quality.get("passed"):
        context.set_outcome("success")
    elif status == "completed":
        context.set_outcome(
            "partial",
            failure_stage="quality_gate",
            failure_code="quality_not_passed",
        )
    return {
        "status": status,
        "error": error,
        "result": parsed_result,
        "quality": quality,
        "elapsed": time.perf_counter() - started,
    }


def _recipe_promotion(
    result: dict[str, object],
    status: str,
    *,
    db_path: Path,
) -> dict[str, object]:
    if status != "completed":
        return {}
    run_id = str(result.get("worker_run_id") or "")
    if not run_id:
        return {}
    from agent.recipe.candidate_store import RecipeCandidateStore

    candidate = RecipeCandidateStore(db_path).get_candidate(run_id)
    promotion = (
        {
            "run_id": candidate.run_id,
            "status": candidate.status,
            "review_attempts": candidate.review_attempts,
            "review_error": candidate.review_error,
            "promotion": candidate.validation.get("promotion", {}),
        }
        if candidate
        else {"run_id": run_id, "status": "not_found"}
    )
    print("RECIPE_PROMOTION=" + json.dumps(promotion, ensure_ascii=False))
    return promotion


def _build_summary(
    args: argparse.Namespace,
    identity: dict[str, object],
    execution: dict[str, object],
    *,
    run_id: str,
    started_at: str,
    metrics: dict,
    events: list,
    preconditions: dict[str, object],
    promotion: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "run_id": run_id,
        "status": execution["status"],
        "error": execution["error"],
        "started_at": started_at,
        "scenario_id": args.scenario_id,
        "experiment_name": identity["experiment_name"],
        "execution_mode": args.execution_mode,
        "site": args.site,
        "search_keyword": args.search_keyword,
        "target_count": max(0, args.target_count),
        "count_mode": args.count_mode,
        "original_query": args.original_query or args.search_keyword,
        "expected_source_urls": list(args.expected_source_url),
        "execution_time_sec": round(float(execution["elapsed"]), 6),
        "git_commit": identity["commit"],
        "git_dirty": identity["git_dirty"],
        "config_fingerprint": identity["config_fingerprint"],
        "recipe_version": identity["recipe_version"],
        "models": identity["configured_models"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": identity["runtime_config"],
        "metrics": metrics,
        "events": [event.model_dump(mode="json") for event in events],
        "quality": execution["quality"],
        "target_contract": dict(execution["quality"]).get("target_contract", {}),
        "experience_guided_preconditions": preconditions,
        "recipe_promotion": promotion,
        "result": execution["result"],
    }


def _write_summary(summary: dict[str, object], summary_path: Path) -> None:
    from agent.observability.langsmith_adapter import publish_langsmith_feedback
    from benchmark.e2e_observability import (
        build_e2e_observability,
        build_langsmith_feedback,
    )

    observability = build_e2e_observability(summary)
    summary["observability"] = observability
    metrics = dict(summary.get("metrics") or {})
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


def _run_e2e(args: argparse.Namespace, log_path: Path, summary_path: Path) -> int:
    from agent.application.worker_execution_service import (
        WorkerExecutionService,
        build_worker_data_services,
    )
    from agent.config import get_settings
    from agent.graph.workflow import build_graph
    from agent.observability.run_context import run_context
    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime
    from shared.db.database import Database

    identity = _execution_identity(args)
    db_path = get_settings().paths.db_path
    preconditions = _experience_guided_preconditions(
        args.execution_mode,
        args.site,
        db_path=db_path,
    )
    print(
        "EXPERIENCE_GUIDED_PRECONDITIONS="
        + json.dumps(preconditions, ensure_ascii=False)
    )
    Database(db_path)
    vision_runtime = VisionWorkerRuntime(graph_factory=build_graph)
    try:
        worker_service = WorkerExecutionService(
            vision_runtime,
            build_worker_data_services(db_path),
        )
        events = []
        started_at = datetime.now(timezone.utc).isoformat()
        with run_context(
            query=args.original_query or args.search_keyword,
            event_sink=events.append,
            prefix="e2e",
            metadata={
                "scenario_id": args.scenario_id,
                "experiment_name": identity["experiment_name"],
                "site": args.site,
                "target_count": max(0, args.target_count),
                "count_mode": args.count_mode,
                "execution_mode": args.execution_mode,
                "git_commit": identity["commit"],
                "git_dirty": identity["git_dirty"],
                "config_fingerprint": identity["config_fingerprint"],
                "recipe_version": identity["recipe_version"],
                "models": identity["configured_models"],
            },
            tags=[
                "e2e",
                f"site:{args.site}",
                f"scenario:{args.scenario_id}",
                f"mode:{args.execution_mode}",
            ],
        ) as (context, _created):
            execution = _execute_collection(
                args,
                worker_service,
                context,
                db_path=db_path,
            )
            metrics = context.snapshot()
        result = dict(execution["result"])
        promotion = _recipe_promotion(
            result,
            str(execution["status"]),
            db_path=db_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"EXECUTION_TIME_SEC={float(execution['elapsed']):.3f}")
        print(f"LOG_TARGET={log_path}")
        final_preconditions = _finalize_experience_guided_preconditions(
            preconditions,
            dict(execution["quality"]),
        )
        summary = _build_summary(
            args,
            identity,
            execution,
            run_id=context.run_id,
            started_at=started_at,
            metrics=metrics,
            events=events,
            preconditions=final_preconditions,
            promotion=promotion,
        )
        _write_summary(summary, summary_path)
        if execution["status"] != "completed":
            return 1
        return 0 if dict(execution["quality"]).get("passed") else 2
    finally:
        vision_runtime.close()


def main() -> int:
    args = _parse_args()
    log_path, summary_path = _artifact_paths(args)

    _apply_execution_mode_environment(args.execution_mode)
    with log_path.open("x", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _Tee(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _Tee(original_stderr, log_file)  # type: ignore[assignment]
        try:
            return _run_e2e(args, log_path, summary_path)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())

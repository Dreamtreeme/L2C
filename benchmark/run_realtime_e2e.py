"""Realtime worker E2E runner with stable file logging."""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run realtime_scraping E2E and tee stdout/stderr to a log file.")
    parser.add_argument("--site", default="wanted")
    parser.add_argument("--query", required=True)
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

    with log_path.open("x", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _Tee(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _Tee(original_stderr, log_file)  # type: ignore[assignment]
        try:
            from agent.application.run_context import run_context
            from agent.application.run_contracts import RunPhase, RunStatus
            from agent.tools.realtime_scraping import realtime_scraping
            from benchmark.quality_eval import evaluate_collection_summary

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
            ) as (context, _created):
                start = time.perf_counter()
                try:
                    result = realtime_scraping.invoke(
                        {"site": args.site, "query": args.query}
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
                metrics = context.snapshot()
            if status == "completed" and isinstance(parsed_during_run, dict):
                candidate_id = str(parsed_during_run.get("submission_id") or "")
                if candidate_id:
                    from agent.application.recipe_promotion_service import (
                        auto_promotion_enabled,
                        wait_for_recipe_candidate_promotion,
                    )

                    if auto_promotion_enabled():
                        promotion_timeout = float(os.getenv("VISION_E2E_PROMOTION_TIMEOUT_SEC", "90"))
                        recipe_promotion = wait_for_recipe_candidate_promotion(
                            candidate_id,
                            timeout=promotion_timeout,
                        )
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
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT_DIR,
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
            except OSError:
                commit = ""
            try:
                parsed_result = json.loads(result) if isinstance(result, str) else result
            except json.JSONDecodeError:
                parsed_result = {"raw": str(result)}
            config_defaults = {
                "VISION_DETAIL_FINAL_EXTRACTION_MODEL": "",
                "VISION_SEARCH_INTENT_MODEL": "",
                "VISION_WORKER_REVIEW_MODEL": "",
                "SOM_OCR_MAX_DIM": "1152",
                "SOM_OCR_REQUEST_TIMEOUT_SEC": "20",
                "PADDLEOCR_IR_OPTIM": "0",
                "REFLEX_ENABLED": "",
                "VISION_RECIPE_AUTO_PROMOTE": "",
                "VISION_RECIPE_CRITIC_EVIDENCE_TEXT_LIMIT": "",
                "VISION_BROWSER_WINDOW_SIZE": "",
                "VISION_BROWSER_WINDOW_WIDTH": "",
                "VISION_BROWSER_WINDOW_HEIGHT": "",
                "VISION_REASONING_SCREEN_GUARD": "",
            }
            summary = {
                "schema_version": 1,
                "run_id": context.run_id,
                "status": status,
                "error": error,
                "started_at": started_at,
                "site": args.site,
                "query": args.query,
                "execution_time_sec": round(elapsed, 6),
                "git_commit": commit,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "config": {
                    key: os.getenv(key, default)
                    for key, default in config_defaults.items()
                },
                "metrics": metrics,
                "events": [event.model_dump(mode="json") for event in events],
                "quality": quality or evaluate_collection_summary(parsed_result),
                "recipe_promotion": recipe_promotion,
                "result": parsed_result,
            }
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
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

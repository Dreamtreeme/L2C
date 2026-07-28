"""같은 백엔드 수명에서 비전 작업자 자원이 재사용되는지 검증한다."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.run_product_chat_e2e import (  # noqa: E402
    _Tee,
    _database_jobs,
    _run_chat_request,
)


def _metric_steps(final_payload: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = dict(final_payload.get("metrics") or {})
    return [
        item
        for item in metrics.get("steps", [])
        if isinstance(item, dict)
    ]


def _request_summary(
    *,
    query: str,
    request_result: dict[str, Any],
    jobs_before: list[dict[str, Any]],
    jobs_after: list[dict[str, Any]],
    resource_snapshot: dict[str, Any],
    duration_sec: float,
) -> dict[str, Any]:
    final_payload = dict(request_result.get("final") or {})
    events = [
        item
        for item in request_result.get("events", [])
        if isinstance(item, dict)
    ]
    event_names = [str(item.get("event") or "") for item in events]
    answer = str(final_payload.get("text") or "")
    cited_ids = [int(value) for value in re.findall(r"\[job_id:(\d+)\]", answer)]
    valid_ids = {int(item["id"]) for item in jobs_after}
    before_ids = {int(item["id"]) for item in jobs_before}
    new_ids = sorted(valid_ids - before_ids)
    steps = _metric_steps(final_payload)
    ocr_startup = [
        item for item in steps if item.get("component") == "ocr_startup"
    ]
    quality = {
        "completed": final_payload.get("status") == "completed",
        "collection_started": "collection_started" in event_names,
        "collection_completed": "collection_completed" in event_names,
        "new_job_ids": new_ids,
        "citation_valid": bool(cited_ids) and set(cited_ids).issubset(valid_ids),
    }
    quality["passed"] = bool(
        not request_result.get("error")
        and quality["completed"]
        and quality["collection_started"]
        and quality["collection_completed"]
        and quality["new_job_ids"]
        and quality["citation_valid"]
    )
    return {
        "query": query,
        "execution_time_sec": round(duration_sec, 6),
        "error": str(request_result.get("error") or ""),
        "quality": quality,
        "ocr_startup_count": len(ocr_startup),
        "ocr_startup_sec": round(
            sum(float(item.get("duration_sec") or 0.0) for item in ocr_startup),
            6,
        ),
        "ocr_request_count": sum(
            item.get("component") == "ocr_request" for item in steps
        ),
        "resource_snapshot": resource_snapshot,
        "final": final_payload,
        "jobs_after": jobs_after,
    }


def _reuse_quality(runs: list[dict[str, Any]]) -> dict[str, Any]:
    snapshots = [
        dict(run.get("resource_snapshot") or {})
        for run in runs
    ]
    pids = [snapshot.get("ocr_worker_pid") for snapshot in snapshots]
    model_counts = [
        int(snapshot.get("ui_model_variant_count") or 0)
        for snapshot in snapshots
    ]
    return {
        "request_quality_passed": bool(runs) and all(
            bool((run.get("quality") or {}).get("passed")) for run in runs
        ),
        "same_ocr_worker_pid": (
            len(pids) >= 2
            and all(pid is not None for pid in pids)
            and len(set(pids)) == 1
        ),
        "first_request_started_ocr": bool(
            runs and int(runs[0].get("ocr_startup_count") or 0) > 0
        ),
        "later_requests_skipped_ocr_startup": bool(
            len(runs) >= 2
            and all(
                int(run.get("ocr_startup_count") or 0) == 0
                for run in runs[1:]
            )
        ),
        "browser_closed_after_each_request": bool(snapshots) and all(
            not snapshot.get("browser_window_bound") for snapshot in snapshots
        ),
        "reasoning_model_cache_reused": (
            len(model_counts) >= 2
            and model_counts[0] > 0
            and len(set(model_counts)) == 1
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run two product requests in one backend lifespan.")
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--conversation-prefix", default="runtime-reuse-e2e")
    args = parser.parse_args()
    if len(args.query) < 2:
        raise SystemExit("--query를 두 번 이상 지정해야 합니다.")

    db_path = Path(args.db_path).resolve()
    log_path = Path(args.log).resolve()
    summary_path = Path(args.summary).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() or summary_path.exists():
        raise SystemExit("재사용 E2E 로그 또는 요약 파일이 이미 존재합니다.")

    os.environ["DB_PATH"] = str(db_path)
    os.environ["VISION_RECIPE_AUTO_PROMOTE"] = "0"

    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    error = ""
    with log_path.open("x", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _Tee(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _Tee(original_stderr, log_file)  # type: ignore[assignment]
        try:
            from fastapi.testclient import TestClient

            from agent.web_server import app

            with TestClient(app) as client:
                runtime = client.app.state.runtime
                for index, query in enumerate(args.query, start=1):
                    jobs_before = _database_jobs(db_path)
                    request_started = time.perf_counter()
                    result = _run_chat_request(
                        client,
                        query=query,
                        conversation_id=f"{args.conversation_prefix}-{index}",
                    )
                    jobs_after = _database_jobs(db_path)
                    runs.append(
                        _request_summary(
                            query=query,
                            request_result=result,
                            jobs_before=jobs_before,
                            jobs_after=jobs_after,
                            resource_snapshot=runtime.vision_runtime.resource_snapshot(),
                            duration_sec=time.perf_counter() - request_started,
                        )
                    )
        except Exception as exc:
            error = str(exc)
            print(f"RUNTIME_REUSE_E2E_ERROR={error}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    quality = _reuse_quality(runs)
    quality["passed"] = bool(not error and quality and all(quality.values()))
    summary = {
        "schema_version": 1,
        "execution_time_sec": round(time.perf_counter() - started, 6),
        "error": error,
        "quality": quality,
        "runs": runs,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"LOG_TARGET={log_path}")
    print(f"SUMMARY_TARGET={summary_path}")
    return 0 if quality["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

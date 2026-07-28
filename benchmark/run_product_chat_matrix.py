"""실제 Chat API의 DB 조회·질문 보완·웹 수집 계약을 함께 검증한다."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
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


def _mode_check(mode: str, value: bool) -> bool:
    normalized = str(mode or "optional").strip().lower()
    if normalized == "required":
        return value
    if normalized == "forbidden":
        return not value
    if normalized == "optional":
        return True
    raise ValueError(f"지원하지 않는 검증 모드입니다: {mode}")


def _scenario_quality(
    scenario: dict[str, Any],
    request_result: dict[str, Any],
    jobs_before: list[dict[str, Any]],
    jobs_after: list[dict[str, Any]],
) -> dict[str, Any]:
    """문구가 아니라 제품 경로의 관찰 가능한 계약을 검사한다."""

    final_payload = dict(request_result.get("final") or {})
    events = [
        item
        for item in request_result.get("events", [])
        if isinstance(item, dict)
    ]
    event_names = [str(item.get("event") or "") for item in events]
    answer = str(final_payload.get("text") or "").strip()
    metrics = dict(final_payload.get("metrics") or {})
    llm_metrics = dict(metrics.get("llm") or {})
    token_totals = dict(llm_metrics.get("totals") or {})
    try:
        execution_time_sec = max(0.0, float(metrics.get("duration_sec") or 0.0))
    except (TypeError, ValueError):
        execution_time_sec = 0.0
    try:
        total_tokens = max(0, int(token_totals.get("total_tokens") or 0))
    except (TypeError, ValueError):
        total_tokens = 0
    cited_ids = sorted(
        {
            int(value)
            for value in re.findall(r"\[job_id:(\d+)\]", answer)
        }
    )
    valid_ids = {int(item["id"]) for item in jobs_after}
    collection_observed = (
        "collection_started" in event_names
        and "collection_completed" in event_names
    )
    collection_document_ids = sorted(
        {
            int(document_id)
            for event in events
            if str(event.get("event") or "") == "collection_completed"
            for document_id in (
                (event.get("data") or {}).get("document_ids", [])
                if isinstance(event.get("data"), dict)
                else []
            )
            if str(document_id).isdigit() and int(document_id) > 0
        }
    )
    clarification = dict(final_payload.get("clarification") or {})
    clarification_observed = bool(clarification) and (
        "clarification_required" in event_names
    )
    option_count = len(
        [
            item
            for item in clarification.get("options", [])
            if isinstance(item, dict)
        ]
    )
    mutation_mode = str(
        scenario.get("database_mutation") or "optional"
    ).strip().lower()
    if mutation_mode == "forbidden":
        database_mutation_valid = jobs_before == jobs_after
    elif mutation_mode == "required":
        database_mutation_valid = jobs_before != jobs_after
    elif mutation_mode == "optional":
        database_mutation_valid = True
    else:
        raise ValueError(
            f"지원하지 않는 DB 변경 검증 모드입니다: {mutation_mode}"
        )

    minimum_citations = max(0, int(scenario.get("minimum_citations") or 0))
    minimum_options = max(
        0,
        int(scenario.get("minimum_clarification_options") or 0),
    )
    maximum_execution_time_sec = max(
        0.0,
        float(scenario.get("maximum_execution_time_sec") or 0.0),
    )
    maximum_total_tokens = max(
        0,
        int(scenario.get("maximum_total_tokens") or 0),
    )
    checks = {
        "request_succeeded": not bool(request_result.get("error")),
        "status_matches": (
            str(final_payload.get("status") or "")
            == str(scenario.get("expected_status") or "completed")
        ),
        "answer_present": bool(answer),
        "collection_contract": _mode_check(
            str(scenario.get("collection") or "optional"),
            collection_observed,
        ),
        "clarification_contract": _mode_check(
            str(scenario.get("clarification") or "optional"),
            clarification_observed,
        ),
        "clarification_options": option_count >= minimum_options,
        "citation_count": len(cited_ids) >= minimum_citations,
        "citation_integrity": set(cited_ids).issubset(valid_ids),
        "citation_scope": (
            bool(cited_ids)
            and set(cited_ids).issubset(set(collection_document_ids))
            if str(scenario.get("citation_scope") or "database").strip().lower()
            == "collection"
            else True
        ),
        "database_job_count": (
            len(jobs_after)
            >= max(0, int(scenario.get("minimum_database_jobs") or 0))
        ),
        "database_mutation": database_mutation_valid,
        "execution_time_budget": (
            maximum_execution_time_sec <= 0
            or (
                execution_time_sec > 0
                and execution_time_sec <= maximum_execution_time_sec
            )
        ),
        "token_budget": (
            maximum_total_tokens <= 0
            or (total_tokens > 0 and total_tokens <= maximum_total_tokens)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "actual_status": str(final_payload.get("status") or ""),
        "collection_observed": collection_observed,
        "clarification_observed": clarification_observed,
        "clarification_option_count": option_count,
        "citation_ids": cited_ids,
        "collection_document_ids": collection_document_ids,
        "database_jobs_before": len(jobs_before),
        "database_jobs_after": len(jobs_after),
        "execution_time_sec": execution_time_sec,
        "total_tokens": total_tokens,
    }


def _selected_scenarios(
    matrix: dict[str, Any],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    scenarios = [
        dict(item)
        for item in matrix.get("scenarios", [])
        if isinstance(item, dict)
    ]
    selected = [
        item
        for item in scenarios
        if not selected_ids or str(item.get("scenario_id") or "") in selected_ids
    ]
    if selected_ids:
        found = {str(item.get("scenario_id") or "") for item in selected}
        missing = sorted(selected_ids - found)
        if missing:
            raise ValueError(f"행렬에 없는 시나리오입니다: {', '.join(missing)}")
    return selected


def _snapshot_database(source: Path, target: Path) -> None:
    """실행 중인 SQLite의 WAL 내용까지 포함해 테스트 DB를 복제한다."""

    with sqlite3.connect(source) as source_connection:
        with sqlite3.connect(target) as target_connection:
            source_connection.backup(target_connection)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run natural-language product regression scenarios in one backend lifespan."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT_DIR / "benchmark" / "product_chat_matrix.json",
    )
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--scenario", default="")
    parser.add_argument("--conversation-prefix", default="product-matrix")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    matrix_path = args.matrix.resolve()
    source_db_path = args.source_db.resolve()
    db_path = args.db_path.resolve()
    log_path = args.log.resolve()
    summary_path = args.summary.resolve()
    for target in (db_path, log_path, summary_path):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise SystemExit(f"출력 파일이 이미 존재합니다: {target}")
    if not source_db_path.exists():
        raise SystemExit(f"원본 DB를 찾을 수 없습니다: {source_db_path}")

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    selected_ids = {
        value.strip()
        for value in str(args.scenario or "").split(",")
        if value.strip()
    }
    scenarios = _selected_scenarios(matrix, selected_ids)
    if not scenarios:
        raise SystemExit("실행할 제품 회귀 시나리오가 없습니다.")

    _snapshot_database(source_db_path, db_path)
    os.environ["DB_PATH"] = str(db_path)
    os.environ["VISION_RECIPE_AUTO_PROMOTE"] = "0"

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    error = ""
    with log_path.open("x", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        if args.verbose:
            sys.stdout = _Tee(original_stdout, log_file)  # type: ignore[assignment]
            sys.stderr = _Tee(original_stderr, log_file)  # type: ignore[assignment]
        else:
            sys.stdout = log_file  # type: ignore[assignment]
            sys.stderr = log_file  # type: ignore[assignment]
        try:
            from fastapi.testclient import TestClient

            from agent.web_server import app

            with TestClient(app) as client:
                runtime = client.app.state.runtime
                for index, scenario in enumerate(scenarios, start=1):
                    scenario_started = time.perf_counter()
                    jobs_before = _database_jobs(db_path)
                    request_result = _run_chat_request(
                        client,
                        query=str(scenario.get("query") or ""),
                        conversation_id=(
                            f"{args.conversation_prefix}-{index}-"
                            f"{scenario.get('scenario_id') or 'scenario'}"
                        ),
                    )
                    jobs_after = _database_jobs(db_path)
                    quality = _scenario_quality(
                        scenario,
                        request_result,
                        jobs_before,
                        jobs_after,
                    )
                    results.append(
                        {
                            "scenario_id": str(
                                scenario.get("scenario_id") or f"scenario-{index}"
                            ),
                            "query": str(scenario.get("query") or ""),
                            "execution_time_sec": round(
                                time.perf_counter() - scenario_started,
                                6,
                            ),
                            "error": str(request_result.get("error") or ""),
                            "quality": quality,
                            "event_names": [
                                str(item.get("event") or "")
                                for item in request_result.get("events", [])
                                if isinstance(item, dict)
                            ],
                            "final": dict(request_result.get("final") or {}),
                            "resource_snapshot": (
                                runtime.vision_runtime.resource_snapshot()
                            ),
                        }
                    )
        except Exception as exc:
            error = str(exc)
            print(f"PRODUCT_MATRIX_ERROR={error}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    summary = {
        "schema_version": 1,
        "matrix": str(matrix_path),
        "source_db": str(source_db_path),
        "test_db": str(db_path),
        "execution_time_sec": round(time.perf_counter() - started, 6),
        "error": error,
        "passed": sum(
            bool((item.get("quality") or {}).get("passed"))
            for item in results
        ),
        "total": len(scenarios),
        "results": results,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "total": summary["total"],
                "execution_time_sec": summary["execution_time_sec"],
                "error": summary["error"],
                "results": [
                    {
                        "scenario_id": item["scenario_id"],
                        "execution_time_sec": item["execution_time_sec"],
                        "quality": item["quality"],
                    }
                    for item in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"LOG_TARGET={log_path}")
    print(f"SUMMARY_TARGET={summary_path}")
    passed = not error and len(results) == len(scenarios) and all(
        bool((item.get("quality") or {}).get("passed"))
        for item in results
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

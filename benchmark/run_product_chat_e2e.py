"""자연어 Chat API부터 SQLite 근거 답변까지 제품 경로를 검증한다."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, TextIO


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


def _frame_payload(line: str, label: str) -> dict[str, Any] | None:
    prefix = f"data: [{label}] "
    if not line.startswith(prefix):
        return None
    try:
        value = json.loads(line[len(prefix) :])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _database_jobs(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone()
        if not exists:
            return []
        return [
            dict(row)
            for row in connection.execute(
                "SELECT id, company_name, position, url, source_platform "
                "FROM jobs ORDER BY id"
            ).fetchall()
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete L2C product chat E2E.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--conversation-id", default="product-e2e")
    args = parser.parse_args()

    db_path = Path(args.db_path).resolve()
    log_path = Path(args.log).resolve()
    summary_path = Path(args.summary).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() or summary_path.exists():
        raise SystemExit("제품 E2E 로그 또는 요약 파일이 이미 존재합니다.")

    os.environ["DB_PATH"] = str(db_path)
    os.environ["VISION_RECIPE_AUTO_PROMOTE"] = "0"

    started = time.perf_counter()
    final_payload: dict[str, Any] = {}
    event_payloads: list[dict[str, Any]] = []
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
                with client.stream(
                    "POST",
                    "/api/chat",
                    json={
                        "query": args.query,
                        "conversation_id": args.conversation_id,
                    },
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        print(line)
                        event = _frame_payload(line, "EVENT")
                        if event is not None:
                            event_payloads.append(event)
                        final = _frame_payload(line, "FINAL")
                        if final is not None:
                            final_payload = final
                        failure = _frame_payload(line, "ERROR")
                        if failure is not None:
                            error = str(failure.get("message") or failure)
        except Exception as exc:
            error = str(exc)
            print(f"PRODUCT_E2E_ERROR={error}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    jobs = _database_jobs(db_path)
    answer = str(final_payload.get("text") or "")
    cited_ids = [int(value) for value in re.findall(r"\[job_id:(\d+)\]", answer)]
    valid_ids = {int(item["id"]) for item in jobs}
    event_names = [str(item.get("event") or "") for item in event_payloads]
    quality = {
        "completed": final_payload.get("status") == "completed",
        "collection_started": "collection_started" in event_names,
        "collection_completed": "collection_completed" in event_names,
        "persisted_job_count": len(jobs),
        "citation_count": len(cited_ids),
        "citation_valid": bool(cited_ids) and set(cited_ids).issubset(valid_ids),
    }
    quality["passed"] = bool(
        not error
        and quality["completed"]
        and quality["collection_started"]
        and quality["collection_completed"]
        and quality["persisted_job_count"] > 0
        and quality["citation_valid"]
    )
    summary = {
        "schema_version": 1,
        "query": args.query,
        "execution_time_sec": round(time.perf_counter() - started, 6),
        "error": error,
        "quality": quality,
        "event_count": len(event_payloads),
        "event_names": event_names,
        "final": final_payload,
        "jobs": jobs,
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

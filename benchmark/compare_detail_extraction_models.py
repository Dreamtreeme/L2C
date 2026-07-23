"""동일한 저장 OCR로 상세 정제 모델의 품질·시간·토큰을 비교한다."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.application.run_context import run_context
from agent.utils.model_dump import dump_model
from benchmark.bench_openai_detail import _dedupe_marker_texts, _load_dotenv


DEFAULT_SUBMISSION_ID = "worker-20260714122243-27719ba9:0"
DEFAULT_MODELS = ("openai:gpt-5.4-mini", "gemini-3.5-flash-lite")
CORE_FIELDS = ("company_name", "position", "url")
CONTENT_FIELDS = ("main_tasks", "requirements", "preferred", "benefits", "tech_stack")
TOKEN_PRICES = {
    "openai:gpt-5.4-mini": (0.75, 4.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}


def _load_reference_jobs(submission_id: str, db_path: Path) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT payload_json FROM worker_submissions WHERE submission_id=?",
            (submission_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise SystemExit(f"worker submission not found: {submission_id}")
    payload = json.loads(row["payload_json"])
    return {
        str(job.get("url") or ""): dict(job)
        for job in payload.get("semantic_evidence") or []
        if isinstance(job, dict) and str(job.get("url") or "")
    }


def _normalized_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return "".join(str(value or "").casefold().split())


def _quality_summary(parsed: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    core_present = {
        field: bool(_normalized_text(parsed.get(field)))
        for field in CORE_FIELDS
    }
    core_matches = {
        field: (
            bool(_normalized_text(reference.get(field)))
            and _normalized_text(parsed.get(field)) == _normalized_text(reference.get(field))
        )
        for field in CORE_FIELDS
    }
    content_counts = {
        field: len(parsed.get(field) or [])
        if isinstance(parsed.get(field), list)
        else int(bool(_normalized_text(parsed.get(field))))
        for field in CONTENT_FIELDS
    }
    return {
        "core_present": core_present,
        "core_matches_reference": core_matches,
        "content_item_counts": content_counts,
        "usable": all(core_present.values())
        and bool(content_counts["main_tasks"])
        and bool(content_counts["requirements"]),
    }


def _estimated_cost(model_spec: str, usage: dict[str, Any]) -> float | None:
    prices = TOKEN_PRICES.get(model_spec)
    if prices is None:
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return round((input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000, 8)


def _run_model(model_spec: str, document: dict[str, Any]) -> dict[str, Any]:
    from agent.application import detail_extraction_service

    previous = os.environ.get("VISION_DETAIL_FINAL_EXTRACTION_MODEL")
    os.environ["VISION_DETAIL_FINAL_EXTRACTION_MODEL"] = model_spec
    detail_extraction_service._detail_extraction_llm = None
    detail_extraction_service._detail_extraction_llm_key = None
    state = {
        "detail_ocr_buffer": {
            "lines": [
                {"text": line}
                for line in str(document.get("ocr_text") or "").splitlines()
                if line.strip()
            ]
        },
        "active_result_card": dict(document.get("active_result_card") or {}),
    }
    try:
        with run_context(query="detail model comparison", prefix="bench-detail") as (
            context,
            _created,
        ):
            started = time.perf_counter()
            parsed = detail_extraction_service.extract_job_from_detail_ocr_buffer(
                state,
                str(document.get("url") or ""),
            )
            duration = time.perf_counter() - started
            metrics = context.snapshot()
    finally:
        if previous is None:
            os.environ.pop("VISION_DETAIL_FINAL_EXTRACTION_MODEL", None)
        else:
            os.environ["VISION_DETAIL_FINAL_EXTRACTION_MODEL"] = previous
        detail_extraction_service._detail_extraction_llm = None
        detail_extraction_service._detail_extraction_llm_key = None

    parsed = dump_model(parsed)
    parsed.pop("raw_ocr_text", None)
    usage = dict(((metrics.get("llm") or {}).get("totals") or {}))
    return {
        "duration_sec": round(duration, 3),
        "usage": usage,
        "estimated_cost_usd": _estimated_cost(model_spec, usage),
        "parsed": parsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-id", default=DEFAULT_SUBMISSION_ID)
    parser.add_argument("--db", type=Path, default=Path("data/jobs.db"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--max-docs", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    documents = _dedupe_marker_texts(args.submission_id, args.db)[: args.max_docs]
    references = _load_reference_jobs(args.submission_id, args.db)
    report: dict[str, Any] = {
        "submission_id": args.submission_id,
        "document_count": len(documents),
        "models": {},
    }
    for model_spec in args.models:
        model_results = []
        for document in documents:
            result = _run_model(model_spec, document)
            result["url"] = document["url"]
            result["ocr_chars"] = len(document["ocr_text"])
            result["quality"] = _quality_summary(
                result["parsed"],
                references.get(document["url"], {}),
            )
            model_results.append(result)
        report["models"][model_spec] = {
            "documents": model_results,
            "usable_count": sum(
                1 for item in model_results if item["quality"]["usable"]
            ),
            "duration_sec": round(
                sum(item["duration_sec"] for item in model_results), 3
            ),
            "input_tokens": sum(
                int(item["usage"].get("input_tokens") or 0)
                for item in model_results
            ),
            "output_tokens": sum(
                int(item["usage"].get("output_tokens") or 0)
                for item in model_results
            ),
            "estimated_cost_usd": round(
                sum(float(item["estimated_cost_usd"] or 0) for item in model_results),
                8,
            ),
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

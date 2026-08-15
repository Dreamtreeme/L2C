"""인크루트 Classic 온보딩 실행과 summary 증거를 생성한다."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.observability.run_context import run_context
from classic.automation.collection import ClassicCollectionRunner
from classic.automation.sites.incruit import INCRUIT_HOMEPAGE
from classic.extractor.normalization import LLMDomJobNormalizer
from shared.db.database import Database
from shared.schema.agent_contract import DEFAULT_JOB_COLLECTION_FIELDS
from shared.schema.collection_intent import CollectionIntent


def run_incruit_classic(
    *,
    homepage: str,
    query: str,
    target_count: int,
    db_path: Path,
    model_name: str | None = None,
) -> dict[str, Any]:
    """격리 DB에 공고를 저장하고 실행 결과와 계측값을 반환한다."""

    intent = CollectionIntent(
        original_query=f"{query} 채용공고 {target_count}건",
        site="incruit",
        search_keyword=query,
        target_count=target_count,
        required_fields=list(DEFAULT_JOB_COLLECTION_FIELDS),
    )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    with run_context(
        query=query,
        prefix="classic-incruit",
        metadata={
            "site": "incruit",
            "target_count": target_count,
            "approach": "classic",
        },
        tags=["site-onboarding", "classic", "incruit"],
        deadline_sec=90 * 60,
    ) as (context, _created):
        result = ClassicCollectionRunner(
            db_path=db_path,
            normalizer=LLMDomJobNormalizer(model_name),
        ).run(homepage, intent)
        context.set_outcome(result.status)
        metrics = context.snapshot()
    runtime_sec = time.perf_counter() - started

    jobs = [
        job.model_dump(mode="json")
        for job in Database(db_path).load_jobs(result.document_ids)
    ]
    llm_metrics = dict(metrics.get("llm") or {})
    totals = dict(llm_metrics.get("totals") or {})
    cost = dict(llm_metrics.get("cost") or {}).get("estimated_total")
    return {
        "site": "incruit",
        "homepage": homepage,
        "query": query,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": round(runtime_sec, 6),
        "total_tokens": int(totals.get("total_tokens") or 0),
        "estimated_cost_usd": float(cost or 0.0),
        "metrics": metrics,
        "result": result.model_dump(mode="json"),
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="인크루트 Classic 수집 summary를 생성합니다."
    )
    parser.add_argument("--homepage", default=INCRUIT_HOMEPAGE)
    parser.add_argument("--query", required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = run_incruit_classic(
        homepage=args.homepage,
        query=args.query,
        target_count=args.count,
        db_path=args.db_path,
        model_name=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["result"], ensure_ascii=False, indent=2))
    print(f"SUMMARY_TARGET={args.output}")
    return 0 if summary["result"]["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_incruit_classic"]

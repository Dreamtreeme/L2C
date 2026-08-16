"""랠릿 Classic 수집을 격리 SQLite DB와 JSON summary로 실행한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.observability.run_context import run_context
from classic.automation.collection import ClassicCollectionRunner
from shared.schema.agent_contract import DEFAULT_JOB_COLLECTION_FIELDS
from shared.schema.collection_intent import CollectionIntent


RALLIT_HOMEPAGE = "https://www.rallit.com/"


def run_rallit_classic(
    query: str,
    *,
    db_path: Path,
    target_count: int = 2,
) -> dict[str, Any]:
    """공식 홈페이지에서 검색을 시작해 공통 수집 결과와 메트릭을 반환한다."""

    required_fields = list(DEFAULT_JOB_COLLECTION_FIELDS)
    intent = CollectionIntent(
        original_query=f"{query} 공고 {target_count}건",
        site="rallit",
        search_keyword=query,
        target_count=target_count,
        required_fields=required_fields,
    )
    runner = ClassicCollectionRunner(db_path=db_path)
    with run_context(query=query, prefix="classic-rallit") as (context, _created):
        result = runner.run(RALLIT_HOMEPAGE, intent)
        context.set_outcome(result.status)
        jobs = [
            job.model_dump(mode="json")
            for job in runner.db.load_jobs(result.document_ids)
        ]
        metrics = context.snapshot()

    return {
        "site": "rallit",
        "homepage": RALLIT_HOMEPAGE,
        "query": query,
        "collection_intent": intent.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
        "jobs": jobs,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="랠릿에서 Classic DOM 방식으로 채용공고를 수집합니다."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=2)
    args = parser.parse_args()

    payload = run_rallit_classic(
        args.query,
        db_path=args.db_path,
        target_count=args.target_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["result"]["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RALLIT_HOMEPAGE", "run_rallit_classic"]

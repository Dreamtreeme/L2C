"""Classic 공통 수집 실행기를 합성 사이트에서 검증한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark.site_onboarding_acceptance import (
    evaluate_site_onboarding_acceptance,
)
from benchmark.synthetic_job_site import (
    SyntheticCollectionAdapter,
    SyntheticJobNormalizer,
    serve_synthetic_job_site,
)
from classic.automation.collection import ClassicCollectionRunner
from shared.db.database import Database
from shared.schema.agent_contract import DEFAULT_JOB_COLLECTION_FIELDS
from shared.schema.collection_intent import CollectionIntent


def run_synthetic_classic(db_path: Path) -> dict:
    """합성 검색 결과 두 건을 저장하고 공통 품질 계약으로 판정한다."""

    required_fields = list(DEFAULT_JOB_COLLECTION_FIELDS)
    intent = CollectionIntent(
        original_query="백엔드 개발자 공고 2개",
        site="synthetic",
        search_keyword="백엔드 개발자",
        target_count=2,
        required_fields=required_fields,
    )
    with serve_synthetic_job_site() as homepage:
        result = ClassicCollectionRunner(
            db_path=db_path,
            normalizer=SyntheticJobNormalizer(),
        ).run(
            homepage,
            intent,
            adapter=SyntheticCollectionAdapter(),
        )

    jobs = [
        job.model_dump(mode="json")
        for job in Database(db_path).load_jobs(result.document_ids)
    ]
    summary = {
        "result": result.model_dump(mode="json"),
        "jobs": jobs,
    }
    quality = evaluate_site_onboarding_acceptance(
        summary,
        homepage=homepage,
        required_fields=[field.value for field in required_fields],
    )
    return {
        "passed": quality["passed"],
        **summary,
        "quality": quality,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="합성 사이트에서 Classic 공통 수집 경로를 검증합니다."
    )
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="l2c-classic-") as temp_dir:
        db_path = args.db_path or Path(temp_dir) / "synthetic.db"
        payload = run_synthetic_classic(db_path)
    output = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

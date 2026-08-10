"""버전 관리되는 로컬 검색 사전을 SQLite에 적재한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.application.search_taxonomy_import_service import (
    import_local_seed,
    taxonomy_counts,
)
from agent.application.job_taxonomy_linker import JobTaxonomyLinker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="로컬 검색 의미 사전 적재")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "jobs.db")
    parser.add_argument(
        "--local-seed",
        type=Path,
        default=ROOT / "data" / "samples" / "search_taxonomy_ko.json",
    )
    parser.add_argument("--skip-relink", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "local": import_local_seed(args.db, args.local_seed)
    }
    if not args.skip_relink:
        result["job_links"] = JobTaxonomyLinker(args.db).relink_all_jobs()
    result["database"] = taxonomy_counts(args.db)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

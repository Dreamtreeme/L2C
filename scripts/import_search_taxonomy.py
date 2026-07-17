"""검색 의미 사전을 내려받고 SQLite에 적재하는 관리 명령."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.application.search_taxonomy_import_service import (
    import_local_seed,
    import_onet_archive,
    taxonomy_counts,
)
from agent.application.search_taxonomy_service import SearchTaxonomyService


DEFAULT_ONET_URL = "https://www.onetcenter.org/dl_files/database/db_30_3_csv.zip"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="검색 의미 사전 적재")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "jobs.db")
    parser.add_argument(
        "--onet-archive",
        type=Path,
        default=ROOT / "data" / "taxonomy" / "source" / "onet_30_3_csv.zip",
    )
    parser.add_argument(
        "--local-seed",
        type=Path,
        default=ROOT / "data" / "samples" / "search_taxonomy_ko.json",
    )
    parser.add_argument("--download-onet", action="store_true")
    parser.add_argument("--skip-onet", action="store_true")
    parser.add_argument("--skip-relink", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.onet_archive.parent.mkdir(parents=True, exist_ok=True)
    if args.download_onet or (not args.skip_onet and not args.onet_archive.exists()):
        urllib.request.urlretrieve(DEFAULT_ONET_URL, args.onet_archive)

    result: dict[str, object] = {}
    if not args.skip_onet:
        result["onet"] = import_onet_archive(args.db, args.onet_archive)
    result["local"] = import_local_seed(args.db, args.local_seed)
    if not args.skip_relink:
        result["job_links"] = SearchTaxonomyService(args.db).relink_all_jobs()
    result["database"] = taxonomy_counts(args.db)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

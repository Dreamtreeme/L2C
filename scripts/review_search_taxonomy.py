"""미등록 검색어 후보를 조회하고 검토 결과를 사전에 반영한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.application.search_taxonomy_review_service import (
    SearchTaxonomyReviewService,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="검색 의미 사전 후보 검토")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "jobs.db")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="후보 목록 조회")
    list_parser.add_argument(
        "--status",
        choices=("candidate", "accepted", "rejected", "all"),
        default="candidate",
    )
    list_parser.add_argument("--limit", type=int, default=50)

    alias_parser = commands.add_parser("alias", help="기존 개념의 별칭으로 승인")
    alias_parser.add_argument("candidate_id", type=int)
    alias_parser.add_argument("concept_key")
    alias_parser.add_argument("--note", default="")

    new_parser = commands.add_parser("new", help="새 검토 개념으로 승인")
    new_parser.add_argument("candidate_id", type=int)
    new_parser.add_argument("canonical_label")
    new_parser.add_argument("--alias", action="append", default=[])
    new_parser.add_argument("--broader", default="")
    new_parser.add_argument("--note", default="")

    reject_parser = commands.add_parser("reject", help="검색 개념이 아닌 후보 거절")
    reject_parser.add_argument("candidate_id", type=int)
    reject_parser.add_argument("--note", default="")
    return parser


def main() -> int:
    args = _parser().parse_args()
    service = SearchTaxonomyReviewService(args.db)
    if args.command == "list":
        result = service.list_candidates(status=args.status, limit=args.limit)
    elif args.command == "alias":
        result = service.accept_as_alias(
            args.candidate_id,
            args.concept_key,
            note=args.note,
        )
    elif args.command == "new":
        result = service.accept_as_new_concept(
            args.candidate_id,
            args.canonical_label,
            aliases=args.alias,
            broader_concept_key=args.broader,
            note=args.note,
        )
    else:
        result = service.reject(args.candidate_id, note=args.note)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

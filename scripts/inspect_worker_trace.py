"""작업자 제출물에서 캡처와 행동의 실행 경로를 조회한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.observability.worker_trace_report import (
    build_worker_trace,
    render_worker_trace,
)
from agent.recipe.submission_store import SubmissionStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="worker_submissions의 캡처-행동-전환 경로 조회"
    )
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "jobs.db")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--submission-id", help="정확한 제출물 ID")
    selector.add_argument("--run-id", help="작업자 실행 ID")
    parser.add_argument(
        "--attempt",
        type=int,
        help="검토 시도 번호. --run-id와 함께 사용하며 생략하면 최신 시도를 조회",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="사람용 텍스트 대신 구조화 JSON 출력",
    )
    return parser


def _load_submission(
    store: SubmissionStore,
    *,
    submission_id: str = "",
    run_id: str = "",
    attempt: int | None = None,
) -> dict | None:
    if submission_id:
        return store.get_submission(submission_id)
    if run_id:
        return store.get_run_attempt(run_id, attempt)
    recent = store.list_recent(limit=1)
    return recent[0] if recent else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.attempt is not None and not args.run_id:
        parser.error("--attempt는 --run-id와 함께 사용해야 합니다.")

    submission = _load_submission(
        SubmissionStore(args.db),
        submission_id=args.submission_id or "",
        run_id=args.run_id or "",
        attempt=args.attempt,
    )
    if submission is None:
        print("조건에 맞는 작업자 제출물이 없습니다.", file=sys.stderr)
        return 1

    trace = build_worker_trace(submission)
    if args.json:
        print(json.dumps(trace, ensure_ascii=False, indent=2))
    else:
        print(render_worker_trace(trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

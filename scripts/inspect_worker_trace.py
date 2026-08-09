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
from shared.schema.feedback_schema import StoredWorkerSubmission


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="worker_submissions의 캡처-행동-전환 경로 조회"
    )
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "jobs.db")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--submission-id", help="정확한 제출물 ID")
    selector.add_argument("--run-id", help="작업자 실행 ID")
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
) -> StoredWorkerSubmission | None:
    return store.find_submission(
        submission_id=submission_id,
        run_id=run_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    submission = _load_submission(
        SubmissionStore(args.db),
        submission_id=args.submission_id or "",
        run_id=args.run_id or "",
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

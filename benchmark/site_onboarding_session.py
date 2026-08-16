"""Codex 신규 사이트 적용 세션의 시각과 개입 내역을 기록한다."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from benchmark.site_onboarding_contract import AcceptanceRun, SiteAdaptationRecord


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read(path: Path) -> SiteAdaptationRecord:
    return SiteAdaptationRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _write(path: Path, record: SiteAdaptationRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def _start(args: argparse.Namespace) -> SiteAdaptationRecord:
    prompt_sha256 = hashlib.sha256(args.prompt.read_bytes()).hexdigest()
    return SiteAdaptationRecord(
        site=args.site,
        homepage=args.homepage,
        approach=args.approach,
        baseline_sha=args.baseline_sha,
        task_id=args.task_id,
        codex_model=args.model,
        prompt_sha256=prompt_sha256,
        started_at=_now(),
    )


def _mark(args: argparse.Namespace) -> SiteAdaptationRecord:
    record = _read(args.session)
    if args.event == "first-success":
        return (
            record
            if record.first_success_at is not None
            else record.model_copy(update={"first_success_at": _now()})
        )
    field = "fix_iterations" if args.event == "fix" else "human_interventions"
    values = [*getattr(record, field), args.note]
    return record.model_copy(update={field: values})


def _acceptance(args: argparse.Namespace) -> SiteAdaptationRecord:
    record = _read(args.session)
    run = AcceptanceRun(
        query=args.query,
        contract_query=args.contract_query,
        substitution_reason=args.substitution_reason,
        summary_path=args.summary_path,
        review_path=args.review_path,
        passed=args.passed,
        runtime_sec=args.runtime_sec,
        total_tokens=args.total_tokens,
        estimated_cost_usd=args.estimated_cost_usd,
    )
    runs = list(record.acceptance_runs)
    index = next(
        (index for index, item in enumerate(runs) if item.query == run.query),
        None,
    )
    if index is None:
        runs.append(run)
    else:
        runs[index] = run
    return record.model_copy(update={"acceptance_runs": runs})


def _finish(args: argparse.Namespace) -> SiteAdaptationRecord:
    record = _read(args.session)
    return record.model_copy(
        update={
            "result_sha": args.result_sha,
            "finished_at": _now(),
            "status": args.status,
            "site_specific_changed_loc": args.site_specific_changed_loc,
            "common_runtime_changed_loc": args.common_runtime_changed_loc,
            "modified_product_files": args.modified_product_file,
            "locator_count": args.locator_count,
            "profile_line_count": args.profile_line_count,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="신규 사이트 적용 세션을 기록합니다.")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start")
    start.add_argument("session", type=Path)
    start.add_argument("--site", required=True)
    start.add_argument("--homepage", required=True)
    start.add_argument("--approach", choices=("classic", "vision"), required=True)
    start.add_argument("--baseline-sha", required=True)
    start.add_argument("--prompt", type=Path, required=True)
    start.add_argument("--model", required=True)
    start.add_argument("--task-id", default="")
    start.set_defaults(handler=_start)

    mark = sub.add_parser("mark")
    mark.add_argument("session", type=Path)
    mark.add_argument(
        "--event",
        choices=("first-success", "fix", "intervention"),
        required=True,
    )
    mark.add_argument("--note", default="")
    mark.set_defaults(handler=_mark)

    acceptance = sub.add_parser("acceptance")
    acceptance.add_argument("session", type=Path)
    acceptance.add_argument("--query", required=True)
    acceptance.add_argument("--contract-query", default="")
    acceptance.add_argument("--substitution-reason", default="")
    acceptance.add_argument("--summary-path", required=True)
    acceptance.add_argument("--review-path", default="")
    acceptance.add_argument("--runtime-sec", type=float, required=True)
    acceptance.add_argument("--passed", action="store_true")
    acceptance.add_argument("--total-tokens", type=int, default=0)
    acceptance.add_argument("--estimated-cost-usd", type=float, default=0.0)
    acceptance.set_defaults(handler=_acceptance)

    finish = sub.add_parser("finish")
    finish.add_argument("session", type=Path)
    finish.add_argument("--result-sha", required=True)
    finish.add_argument(
        "--status",
        choices=("completed", "failed", "invalid"),
        required=True,
    )
    finish.add_argument("--site-specific-changed-loc", type=int, default=0)
    finish.add_argument("--common-runtime-changed-loc", type=int, default=0)
    finish.add_argument("--modified-product-file", action="append", default=[])
    finish.add_argument("--locator-count", type=int, default=0)
    finish.add_argument("--profile-line-count", type=int, default=0)
    finish.set_defaults(handler=_finish)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    record = args.handler(args)
    _write(args.session, record)
    print(record.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

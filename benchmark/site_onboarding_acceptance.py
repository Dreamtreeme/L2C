"""신규 사이트 수집 결과를 공통 자동·사람 품질 기준으로 판정한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from benchmark.manual_evaluation import RunManualJudgement, evaluate_manual_run
from benchmark.quality_eval import (
    evaluate_collection_summary,
    evaluate_job_records,
    extract_job_records,
)
from shared.schema.agent_contract import DEFAULT_JOB_COLLECTION_FIELDS


def _domain_quality(
    jobs: list[dict[str, Any]],
    allowed_domains: list[str],
) -> dict[str, Any]:
    normalized_domains = {
        domain.casefold().removeprefix("www.")
        for domain in allowed_domains
        if domain.strip()
    }
    invalid_urls = []
    for job in jobs:
        url = str(job.get("url") or "").strip()
        host = (urlsplit(url).hostname or "").casefold().removeprefix("www.")
        if not host or not any(
            host == domain or host.endswith(f".{domain}")
            for domain in normalized_domains
        ):
            invalid_urls.append(url)
    return {
        "allowed_domains": sorted(normalized_domains),
        "invalid_urls": invalid_urls,
        "passed": bool(normalized_domains and not invalid_urls),
    }


def _hardcoding_quality(
    jobs: list[dict[str, Any]],
    patch_text: str,
) -> dict[str, Any]:
    if not patch_text:
        return {"checked": False, "matched_literals": [], "passed": True}
    literals = {
        str(job.get(field) or "").strip()
        for job in jobs
        for field in ("url", "company_name", "position")
        if len(str(job.get(field) or "").strip()) >= 4
    }
    matched = sorted(literal for literal in literals if literal in patch_text)
    return {"checked": True, "matched_literals": matched, "passed": not matched}


def evaluate_site_onboarding_acceptance(
    summary: dict[str, Any],
    *,
    homepage: str,
    required_fields: list[str],
    jobs: list[dict[str, Any]] | None = None,
    allowed_domains: list[str] | None = None,
    patch_text: str = "",
    judgement: RunManualJudgement | None = None,
    require_manual: bool = False,
) -> dict[str, Any]:
    """수집 수량·스키마·출처·하드코딩과 블라인드 판정을 결합한다."""

    result = (
        summary.get("result") if isinstance(summary.get("result"), dict) else summary
    )
    job_records = jobs if jobs is not None else extract_job_records(summary)
    collection_quality = evaluate_collection_summary(result)
    job_quality = evaluate_job_records(
        job_records,
        required_fields=required_fields,
    )
    homepage_host = urlsplit(homepage).hostname or ""
    domain_quality = _domain_quality(
        job_records,
        allowed_domains or [homepage_host],
    )
    hardcoding_quality = _hardcoding_quality(job_records, patch_text)
    automatic_passed = bool(
        collection_quality["passed"]
        and job_quality["record_count"] == collection_quality["target_count"]
        and job_quality["schema_valid_rate"] == 1.0
        and job_quality["required_field_coverage"] == 1.0
        and job_quality["unique_url_rate"] == 1.0
        and domain_quality["passed"]
        and hardcoding_quality["passed"]
    )
    manual = evaluate_manual_run(summary, judgement) if judgement else None
    manual_passed = bool(manual and manual["manual_contract_passed"])
    return {
        "passed": bool(
            automatic_passed and (manual_passed if require_manual else True)
        ),
        "automatic_passed": automatic_passed,
        "manual_required": require_manual,
        "manual_passed": manual_passed if judgement else None,
        "collection_quality": collection_quality,
        "job_quality": job_quality,
        "domain_quality": domain_quality,
        "hardcoding_quality": hardcoding_quality,
        "manual_quality": manual,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="신규 사이트 수집 결과를 공통 품질 기준으로 판정합니다."
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--homepage", required=True)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--allowed-domain", action="append")
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--manual-judgement", type=Path)
    parser.add_argument("--require-manual", action="store_true")
    parser.add_argument("--required-field", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    judgement = (
        RunManualJudgement.model_validate_json(
            args.manual_judgement.read_text(encoding="utf-8")
        )
        if args.manual_judgement
        else None
    )
    required_fields = args.required_field or [
        field.value for field in DEFAULT_JOB_COLLECTION_FIELDS
    ]
    jobs = None
    if args.jobs:
        jobs = extract_job_records(json.loads(args.jobs.read_text(encoding="utf-8")))
    elif args.db_path:
        from shared.db.database import Database

        result = (
            summary.get("result")
            if isinstance(summary.get("result"), dict)
            else summary
        )
        jobs = [
            job.model_dump(mode="json")
            for job in Database(args.db_path).load_jobs(
                result.get("document_ids") or []
            )
        ]
    report = evaluate_site_onboarding_acceptance(
        summary,
        homepage=args.homepage,
        required_fields=required_fields,
        jobs=jobs,
        allowed_domains=args.allowed_domain,
        patch_text=(args.patch.read_text(encoding="utf-8") if args.patch else ""),
        judgement=judgement,
        require_manual=args.require_manual,
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_site_onboarding_acceptance"]

"""신규 사이트의 Classic·Vision 증분 적용 공수를 비교한다."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from benchmark.site_onboarding_contract import (
    SiteAdaptationManifest,
    SiteAdaptationRecord,
)


def _record_summary(record: SiteAdaptationRecord) -> dict[str, Any]:
    runtimes = [run.runtime_sec for run in record.acceptance_runs]
    return {
        **record.model_dump(mode="json"),
        "human_intervention_count": len(record.human_interventions),
        "fix_iteration_count": len(record.fix_iterations),
        "modified_file_count": len(record.modified_product_files),
        "acceptance_success_count": sum(run.passed for run in record.acceptance_runs),
        "acceptance_attempt_count": len(record.acceptance_runs),
        "prompt_to_first_success_sec": record.prompt_to_first_success_sec,
        "prompt_to_acceptance_sec": record.prompt_to_acceptance_sec,
        "runtime_median_sec": round(median(runtimes), 3) if runtimes else None,
        "llm_tokens": sum(run.total_tokens for run in record.acceptance_runs),
        "llm_cost_usd": round(
            sum(run.estimated_cost_usd for run in record.acceptance_runs),
            9,
        ),
    }


def _contract_passed(
    record: SiteAdaptationRecord,
    manifest: SiteAdaptationManifest,
) -> bool:
    expected_queries = manifest.task_contract.acceptance_queries
    actual_queries = [
        run.contract_query or run.query for run in record.acceptance_runs
    ]
    return bool(
        record.status == "completed"
        and record.baseline_sha == manifest.baseline_sha
        and record.prompt_sha256 == manifest.prompt_sha256
        and record.finished_at is not None
        and actual_queries == expected_queries
        and all(run.passed for run in record.acceptance_runs)
    )


def _decision(sites: list[dict[str, Any]]) -> str:
    if not sites or any(not site["classic_contract_passed"] for site in sites):
        return "비교 불가"
    if any(not site["vision_contract_passed"] for site in sites):
        return "기각"
    vision_wins = [
        site["prompt_to_acceptance_sec_saved"] > 0
        and site["site_specific_changed_loc_saved"] > 0
        for site in sites
    ]
    if all(vision_wins):
        return "지지"
    classic_wins = [
        site["prompt_to_acceptance_sec_saved"] < 0
        and site["site_specific_changed_loc_saved"] < 0
        for site in sites
    ]
    return "기각" if all(classic_wins) else "혼합"


def evaluate_site_adaptation(
    manifest: SiteAdaptationManifest,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, SiteAdaptationRecord]] = defaultdict(dict)
    for record in manifest.records:
        grouped[record.site][record.approach] = record

    sites: list[dict[str, Any]] = []
    for site, approaches in sorted(grouped.items()):
        classic = approaches.get("classic")
        vision = approaches.get("vision")
        if classic is None or vision is None:
            raise ValueError(f"{site}에는 classic과 vision 기록이 모두 필요합니다.")
        classic_summary = _record_summary(classic)
        vision_summary = _record_summary(vision)
        classic_contract_passed = _contract_passed(classic, manifest)
        vision_contract_passed = _contract_passed(vision, manifest)
        classic_time = classic.prompt_to_acceptance_sec
        vision_time = vision.prompt_to_acceptance_sec
        sites.append(
            {
                "site": site,
                "classic": classic_summary,
                "vision": vision_summary,
                "classic_contract_passed": classic_contract_passed,
                "vision_contract_passed": vision_contract_passed,
                "comparison_valid": (
                    classic_contract_passed and vision_contract_passed
                ),
                "prompt_to_acceptance_sec_saved": round(
                    (classic_time or 0.0) - (vision_time or 0.0),
                    3,
                ),
                "site_specific_changed_loc_saved": (
                    classic.site_specific_changed_loc - vision.site_specific_changed_loc
                ),
            }
        )

    return {
        "schema_version": 2,
        "baseline_sha": manifest.baseline_sha,
        "prompt_sha256": manifest.prompt_sha256,
        "task_contract": manifest.task_contract.model_dump(mode="json"),
        "foundation": {
            **manifest.foundation.model_dump(mode="json"),
            "duration_sec": manifest.foundation.duration_sec,
        },
        "site_count": len(sites),
        "decision": _decision(sites),
        "sites": sites,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classic·Vision 신규 사이트 적용 공수를 비교합니다.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = SiteAdaptationManifest.model_validate_json(
        args.manifest.read_text(encoding="utf-8"),
    )
    output = json.dumps(
        evaluate_site_adaptation(manifest),
        ensure_ascii=False,
        indent=2,
    )
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SiteAdaptationManifest",
    "SiteAdaptationRecord",
    "evaluate_site_adaptation",
]

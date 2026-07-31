"""신규 사이트의 Classic·Vision 적용 공수를 같은 단위로 비교한다."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Literal

from pydantic import BaseModel, Field


class SiteAdaptationRecord(BaseModel):
    site: str
    approach: Literal["classic", "vision"]
    implementation_minutes: float = Field(ge=0)
    site_specific_code_lines: int = Field(ge=0)
    modified_file_count: int = Field(ge=0)
    common_runtime_code_lines: int = Field(ge=0)
    fix_iteration_count: int = Field(ge=0)
    successful_runs: int = Field(ge=0)
    attempted_runs: int = Field(default=3, ge=1)
    runtime_sec: list[float] = Field(default_factory=list)
    notes: str = ""


class SiteAdaptationManifest(BaseModel):
    schema_version: int = 1
    commit_sha: str
    task_contract: dict[str, Any]
    records: list[SiteAdaptationRecord]


def evaluate_site_adaptation(
    manifest: SiteAdaptationManifest,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, SiteAdaptationRecord]] = defaultdict(dict)
    for record in manifest.records:
        grouped[record.site][record.approach] = record

    sites = []
    for site, approaches in sorted(grouped.items()):
        classic = approaches.get("classic")
        vision = approaches.get("vision")
        if classic is None or vision is None:
            raise ValueError(
                f"{site}에는 classic과 vision 기록이 모두 필요합니다."
            )
        classic_contract_passed = (
            classic.attempted_runs >= 3
            and classic.successful_runs == classic.attempted_runs
            and len(classic.runtime_sec) == classic.attempted_runs
        )
        vision_contract_passed = (
            vision.attempted_runs >= 3
            and vision.successful_runs == vision.attempted_runs
            and len(vision.runtime_sec) == vision.attempted_runs
        )
        comparison_valid = (
            classic_contract_passed and vision_contract_passed
        )
        sites.append(
            {
                "site": site,
                "classic": classic.model_dump(),
                "vision": vision.model_dump(),
                "classic_contract_passed": classic_contract_passed,
                "vision_contract_passed": vision_contract_passed,
                "comparison_valid": comparison_valid,
                "implementation_minutes_saved": round(
                    classic.implementation_minutes
                    - vision.implementation_minutes,
                    3,
                ),
                "site_specific_code_lines_saved": (
                    classic.site_specific_code_lines
                    - vision.site_specific_code_lines
                ),
                "vision_profile_only": (
                    comparison_valid
                    and vision.common_runtime_code_lines == 0
                    and vision.site_specific_code_lines == 0
                ),
                "classic_runtime_median_sec": (
                    round(median(classic.runtime_sec), 3)
                    if classic.runtime_sec
                    else None
                ),
                "vision_runtime_median_sec": (
                    round(median(vision.runtime_sec), 3)
                    if vision.runtime_sec
                    else None
                ),
            }
        )
    return {
        "schema_version": 1,
        "commit_sha": manifest.commit_sha,
        "task_contract": manifest.task_contract,
        "site_count": len(sites),
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

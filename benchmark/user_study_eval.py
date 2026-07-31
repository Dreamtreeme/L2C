"""수동 조사와 L2C 사용 과제의 사람 활동시간·품질을 비교한다."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Literal

from pydantic import BaseModel, Field


class UserStudyRecord(BaseModel):
    participant_id: str
    task_id: str
    mode: Literal["manual", "l2c"]
    order: int = Field(ge=1)
    total_completion_sec: float = Field(ge=0)
    human_active_sec: float = Field(ge=0)
    result_review_sec: float = Field(ge=0)
    suitable_job_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    missing_field_count: int = Field(ge=0)
    factual_error_count: int = Field(ge=0)
    citation_link_rate: float = Field(ge=0, le=1)
    usefulness_score: int = Field(ge=1, le=5)
    trust_score: int = Field(ge=1, le=5)
    correction_count: int = Field(ge=0)
    notes: str = ""


class UserStudyManifest(BaseModel):
    schema_version: int = 1
    study_contract: dict[str, Any]
    records: list[UserStudyRecord]


def evaluate_user_study(
    manifest: UserStudyManifest,
) -> dict[str, Any]:
    by_mode: dict[str, list[UserStudyRecord]] = defaultdict(list)
    participants = set()
    tasks = set()
    for record in manifest.records:
        by_mode[record.mode].append(record)
        participants.add(record.participant_id)
        tasks.add(record.task_id)

    def summarize(records: list[UserStudyRecord]) -> dict[str, Any]:
        if not records:
            return {"record_count": 0}
        return {
            "record_count": len(records),
            "median_total_completion_sec": round(
                median(item.total_completion_sec for item in records),
                3,
            ),
            "median_human_active_sec": round(
                median(item.human_active_sec for item in records),
                3,
            ),
            "median_review_sec": round(
                median(item.result_review_sec for item in records),
                3,
            ),
            "total_suitable_jobs": sum(
                item.suitable_job_count for item in records
            ),
            "total_duplicates": sum(
                item.duplicate_count for item in records
            ),
            "total_missing_fields": sum(
                item.missing_field_count for item in records
            ),
            "total_factual_errors": sum(
                item.factual_error_count for item in records
            ),
            "mean_citation_link_rate": round(
                sum(item.citation_link_rate for item in records)
                / len(records),
                3,
            ),
            "mean_usefulness_score": round(
                sum(item.usefulness_score for item in records)
                / len(records),
                3,
            ),
            "mean_trust_score": round(
                sum(item.trust_score for item in records)
                / len(records),
                3,
            ),
            "total_corrections": sum(
                item.correction_count for item in records
            ),
        }

    manual = summarize(by_mode["manual"])
    l2c = summarize(by_mode["l2c"])
    modes_by_participant: dict[str, set[str]] = defaultdict(set)
    modes_by_task: dict[str, set[str]] = defaultdict(set)
    for record in manifest.records:
        modes_by_participant[record.participant_id].add(record.mode)
        modes_by_task[record.task_id].add(record.mode)
    required_participants = int(
        manifest.study_contract.get("participants") or len(participants)
    )
    configured_tasks = manifest.study_contract.get("tasks")
    required_tasks = (
        len(configured_tasks)
        if isinstance(configured_tasks, list) and configured_tasks
        else len(tasks)
    )
    design_complete = bool(
        participants
        and tasks
        and len(participants) >= required_participants
        and len(tasks) >= required_tasks
        and all(
            modes == {"manual", "l2c"}
            for modes in modes_by_participant.values()
        )
        and all(
            modes == {"manual", "l2c"}
            for modes in modes_by_task.values()
        )
    )
    manual_active = manual.get("median_human_active_sec")
    l2c_active = l2c.get("median_human_active_sec")
    activity_reduction = (
        round((manual_active - l2c_active) / manual_active, 6)
        if design_complete
        and manual_active
        and l2c_active is not None
        else None
    )
    return {
        "schema_version": 1,
        "study_contract": manifest.study_contract,
        "participant_count": len(participants),
        "task_count": len(tasks),
        "design_complete": design_complete,
        "manual": manual,
        "l2c": l2c,
        "human_activity_reduction_rate": activity_reduction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="사람 수동 조사와 L2C 예비 실험을 비교합니다.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = UserStudyManifest.model_validate_json(
        args.manifest.read_text(encoding="utf-8"),
    )
    output = json.dumps(
        evaluate_user_study(manifest),
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

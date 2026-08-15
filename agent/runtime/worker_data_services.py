"""작업자 그래프가 애플리케이션 데이터 기능을 호출하는 포트 계약."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from shared.schema.experience_rule_schema import ExperienceRule
from shared.schema.jd_schema import JobCapture


ExistingJobCardMarker = Callable[
    [list[dict[str, Any]], str],
    tuple[list[dict[str, Any]], list[dict[str, Any]]],
]
ExistingJobUrlLookup = Callable[[str, Sequence[JobCapture]], dict[str, Any]]
ExperienceRuleLoader = Callable[..., list[tuple[str, ExperienceRule]]]
RecipeReplayRecorder = Callable[[str, bool], bool]


@dataclass(frozen=True, slots=True)
class WorkerDataServices:
    """그래프에서 필요한 데이터 조회와 정제 함수를 보관한다."""

    mark_existing_job_cards: ExistingJobCardMarker
    find_existing_job_url: ExistingJobUrlLookup
    load_experience_rules: ExperienceRuleLoader
    record_recipe_replay: RecipeReplayRecorder


__all__ = ["ExperienceRuleLoader", "WorkerDataServices"]

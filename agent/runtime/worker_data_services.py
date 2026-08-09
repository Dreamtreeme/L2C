"""작업자 그래프가 애플리케이션 데이터 기능을 호출하는 포트 계약."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Callable, Protocol

from agent.runtime.worker_contracts import WorkerState
from shared.schema.recipe_schema import SiteRecipe
from shared.schema.jd_schema import CollectedJob, JobPosting


DetailJobExtractor = Callable[[WorkerState, str], JobPosting | None]
ExistingJobCardMarker = Callable[
    [list[dict[str, Any]], str],
    tuple[list[dict[str, Any]], list[dict[str, Any]]],
]
ExistingJobUrlLookup = Callable[[str, Sequence[CollectedJob]], dict[str, Any]]


class SiteRecipeLoader(Protocol):
    """사이트와 작업 분류에 맞는 활성 레시피를 조회한다."""

    def __call__(
        self,
        site: str,
        *,
        task_category: str | None = None,
    ) -> list[tuple[str, SiteRecipe]]: ...


@dataclass(frozen=True, slots=True)
class WorkerDataServices:
    """그래프에서 필요한 데이터 조회와 정제 함수를 보관한다."""

    extract_job_detail: DetailJobExtractor
    mark_existing_job_cards: ExistingJobCardMarker
    find_existing_job_url: ExistingJobUrlLookup
    load_site_recipes: SiteRecipeLoader


__all__ = ["WorkerDataServices"]

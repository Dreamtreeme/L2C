"""비전 작업자 한 번의 실행과 재귀 한도 결과를 관리한다."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agent.application.collection_request_builder import (
    build_site_goal,
)
from agent.application.worker_execution_service import (
    execute_worker_graph,
)
from agent.config import get_settings
from agent.graph.state_factory import create_worker_state
from agent.recipe.task_category import (
    DEFAULT_SEARCH_TASK_CATEGORY,
    normalize_task_category,
)
from agent.runtime.job_field_contract import (
    build_job_collection_contract,
)
from agent.sites import load_site_profile
from agent.utils.model_dump import dump_model
from shared.schema.collection_intent import (
    CollectionCountMode,
    CollectionIntent,
)
from shared.schema.feedback_schema import WorkerSubmission

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerRunResult:
    """작업자 실행에서 제출·저장 계층에 필요한 결과."""

    submission: WorkerSubmission
    extracted_jd: dict[str, Any]
    site_slug: str
    site_name: str


def run_worker_once(
    collection_intent: CollectionIntent,
    task_category: str | None = None,
    run_id: str | None = None,
    worker_runtime: Any = None,
) -> WorkerRunResult:
    """비전 작업자 한 번을 실행하고 검토 전 제출물을 반환한다."""

    from agent.recipe.reviewer import build_worker_submission, new_worker_run_id

    site_profile = load_site_profile(collection_intent.site)
    site_slug = site_profile.slug or collection_intent.site or "unknown"
    site_name = site_profile.display_name or site_slug
    run_id = run_id or new_worker_run_id()
    task_category = normalize_task_category(
        task_category
        or collection_intent.task_category
        or DEFAULT_SEARCH_TASK_CATEGORY
    )
    resolved_intent = collection_intent.model_copy(
        update={"site": site_slug, "task_category": task_category},
    )
    intent_payload = dump_model(resolved_intent)
    search_keyword = resolved_intent.search_keyword
    target_count = resolved_intent.target_count
    job_collection_contract = build_job_collection_contract(
        intent_payload,
        profile_fields=site_profile.collection_policy.required_fields,
    )
    intent_payload["required_fields"] = list(
        job_collection_contract["required_fields"]
    )
    goal = build_site_goal(
        resolved_intent,
        site_profile,
        job_collection_contract=job_collection_contract,
    )
    initial_state = create_worker_state(
        goal,
        worker_run_id=run_id,
    )
    initial_state["job_collection_contract"] = job_collection_contract
    initial_state["recipe_params"] = {
        "query": search_keyword,
        "keyword": search_keyword,
        "search_keyword": search_keyword,
        "target_count": target_count,
        "site": site_slug,
        "task_category": task_category,
        "count_mode": intent_payload.get(
            "count_mode",
            CollectionCountMode.UNSPECIFIED.value,
        ),
        "collection_intent": intent_payload,
    }
    from agent.runtime.action_permissions import (
        build_public_collection_permission_contract,
    )

    initial_state["action_permission_contract"] = (
        build_public_collection_permission_contract(
            site_profile,
            initial_state["recipe_params"],
        )
    )

    logger.info(
        "비전 작업자 그래프 시작: site=%s",
        site_slug,
    )
    recursion_limit = get_settings().vision.recursion_limit
    final_state, hit_recursion_limit = execute_worker_graph(
        initial_state,
        site_profile,
        recursion_limit,
        worker_runtime=worker_runtime,
    )
    extracted = final_state.get("extracted_jd", {}) or {}
    is_finished = bool(final_state.get("is_finished", False))
    run_status = "stopped"
    if is_finished:
        run_status = "finished"
    elif hit_recursion_limit:
        run_status = "recursion_limit"
    submission = build_worker_submission(
        final_state,
        run_status=run_status,
        hit_recursion_limit=hit_recursion_limit,
        persisted_count=0,
        run_id=run_id,
    )

    return WorkerRunResult(
        submission=submission,
        extracted_jd=extracted,
        site_slug=site_slug,
        site_name=site_name,
    )


__all__ = [
    "WorkerRunResult",
    "run_worker_once",
]

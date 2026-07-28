"""비전 작업자 한 번의 실행과 재귀 한도 결과를 관리한다."""

from __future__ import annotations

import logging
from typing import Any

from agent.application.collection_request_builder import (
    append_review_feedback,
    build_direct_search_url,
    build_site_goal,
    extract_search_intent,
    load_collection_profile,
    normalize_target_count,
)
from agent.application.worker_execution_service import (
    execute_worker_graph,
    prepare_worker_start_screen,
    run_graph_with_last_state,
)
from agent.config import get_settings
from agent.recipe.task_category import (
    DEFAULT_SEARCH_TASK_CATEGORY,
    normalize_task_category,
)
from agent.utils.model_dump import dump_model
from shared.schema.collection_intent import (
    CollectionCountMode,
    normalize_collection_intent,
)

logger = logging.getLogger(__name__)


def _commit_feedback_episodes(
    final_state: dict,
    hit_recursion_limit: bool,
    is_finished: bool,
    run_id: str = "",
    review_attempt: int = 0,
) -> int:
    """나중에 Critic이 검토할 피드백 episode를 저장한다."""

    episodes = list(final_state.get("feedback_episodes", []) or [])
    if not episodes:
        return 0
    run_status = (
        "finished"
        if is_finished
        else "recursion_limit"
        if hit_recursion_limit
        else "stopped"
    )
    try:
        from agent.recipe.feedback_store import FeedbackStore

        saved = FeedbackStore().commit_episodes(
            episodes,
            run_id=run_id or None,
            run_status=run_status,
            source="realtime_scraping",
            review_attempt=review_attempt,
        )
        logger.info(
            "작업자 피드백 episode 저장: episodes=%s, saved=%s, status=%s",
            len(episodes),
            saved,
            run_status,
        )
        return saved
    except Exception as exc:
        logger.debug("작업자 피드백 episode 저장 생략: %s", exc)
        return 0


def _worker_run_status(
    hit_recursion_limit: bool,
    is_finished: bool,
) -> str:
    if is_finished:
        return "finished"
    if hit_recursion_limit:
        return "recursion_limit"
    return "stopped"


def worker_review_retries() -> int:
    return get_settings().recipe.worker_review_retries


def _suggested_recursion_limit(current_limit: int) -> int:
    increment = get_settings().vision.recursion_limit_increment
    return current_limit + increment


def needs_human_limit_approval(
    *,
    hit_recursion_limit: bool,
    is_finished: bool,
    persisted_count: int,
    target_count: int = 0,
) -> bool:
    if not get_settings().vision.hitl_on_recursion_limit:
        return False
    persisted = int(persisted_count or 0)
    target = int(target_count or 0)
    if target > 0 and persisted >= target:
        return False
    return bool(
        hit_recursion_limit
        and not is_finished
        and persisted > 0
    )


def _job_report_items(submission: dict) -> list[dict[str, str]]:
    summary = (
        submission.get("extracted_summary")
        if isinstance(submission, dict)
        else {}
    )
    jobs = summary.get("jobs") if isinstance(summary, dict) else []
    report_items: list[dict[str, str]] = []
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        report_items.append(
            {
                "company": str(job.get("company") or ""),
                "position": str(job.get("position") or ""),
                "url": str(job.get("url") or ""),
                "field_count": str(job.get("field_count") or ""),
            }
        )
    return report_items


def build_limit_intermediate_report(
    worker_result: dict,
    submission: dict,
    *,
    persisted_count: int,
    current_limit: int,
    target_count: int = 0,
) -> dict[str, Any]:
    final_state = (
        worker_result.get("final_state")
        if isinstance(worker_result, dict)
        else {}
    )
    if not isinstance(final_state, dict):
        final_state = {}
    summary = (
        submission.get("extracted_summary")
        if isinstance(submission, dict)
        else {}
    )
    if not isinstance(summary, dict):
        summary = {}
    suggested_limit = _suggested_recursion_limit(current_limit)
    target = int(
        target_count
        or submission.get("target_count")
        or worker_result.get("target_count")
        or 0
    )
    persisted = int(persisted_count or 0)
    return {
        "status": "needs_human_limit_approval",
        "reason": "recursion_limit_reached_with_partial_data",
        "current_recursion_limit": current_limit,
        "suggested_recursion_limit": suggested_limit,
        "collected_count": int(
            submission.get("collected_count")
            or summary.get("job_count")
            or 0
        ),
        "persisted_count": persisted,
        "target_count": target,
        "remaining_collection_count": (
            max(0, target - persisted)
            if target > 0
            else 0
        ),
        "collection_complete": bool(
            target > 0
            and persisted >= target
        ),
        "current_url": str(
            summary.get("current_url")
            or final_state.get("current_url")
            or ""
        ),
        "jobs": _job_report_items(submission),
        "question": (
            f"현재 recursion limit {current_limit}에 도달했습니다. "
            f"limit을 {suggested_limit}로 늘려 계속 진행할까요?"
        ),
    }


def limit_report_requires_more_collection(report: dict) -> bool:
    """명시적인 수집 개수 기준으로 추가 작업이 필요한지 판정한다."""

    if not isinstance(report, dict):
        return False
    target = int(report.get("target_count") or 0)
    persisted = int(report.get("persisted_count") or 0)
    if target > 0:
        return persisted < target
    return persisted > 0


def _initial_worker_state(
    goal: str,
    *,
    run_id: str = "",
    attempt_index: int = 0,
) -> dict:
    from agent.graph.state_factory import create_worker_state

    return create_worker_state(
        goal,
        worker_run_id=run_id,
        worker_attempt_index=attempt_index,
    )


def run_worker_once(
    search_keyword: str,
    site: str | None = None,
    target_count: int = 0,
    task_category: str = DEFAULT_SEARCH_TASK_CATEGORY,
    search_intent_resolved: bool = False,
    review_feedback: str | None = None,
    review_attempt: int = 0,
    run_id: str | None = None,
    task_context: dict[str, Any] | None = None,
    collection_intent: dict[str, Any] | None = None,
    worker_runtime: Any = None,
) -> dict:
    """비전 작업자 한 번을 실행하고 검토 전 제출물을 반환한다."""

    from agent.recipe.reviewer import build_worker_submission, new_worker_run_id

    site_profile = load_collection_profile(site)
    site_slug = site_profile.slug or site or "unknown"
    site_name = site_profile.display_name or site_slug
    run_id = run_id or new_worker_run_id()
    raw_search_keyword = search_keyword
    requested_target_count = normalize_target_count(target_count)
    if collection_intent:
        search_intent = dump_model(
            normalize_collection_intent(
                collection_intent,
                original_query=raw_search_keyword,
                site=site_slug,
                search_keyword=search_keyword,
                target_count=requested_target_count,
            )
        )
        search_intent["source"] = "structured_arguments"
        search_intent["error"] = ""
    elif search_intent_resolved:
        search_intent = dump_model(
            normalize_collection_intent(
                original_query=raw_search_keyword,
                site=site_slug,
                search_keyword=search_keyword,
                target_count=requested_target_count,
            )
        )
        search_intent["source"] = "structured_arguments"
        search_intent["error"] = ""
    else:
        search_intent = extract_search_intent(
            search_keyword,
            site_profile,
        )
    search_keyword = str(
        search_intent.get("search_keyword")
        or search_keyword
        or ""
    ).strip()
    inferred_target_count = normalize_target_count(
        search_intent.get("target_count") or 0
    )
    target_count = requested_target_count or inferred_target_count
    task_category = normalize_task_category(
        task_category
        or DEFAULT_SEARCH_TASK_CATEGORY
    )
    direct_search_url = build_direct_search_url(
        search_keyword,
        site_profile,
    )
    goal = append_review_feedback(
        build_site_goal(
            search_keyword,
            site_profile,
            direct_search_url,
            target_count=target_count,
            task_context=task_context,
            collection_intent=search_intent,
        ),
        review_feedback,
    )
    initial_state = _initial_worker_state(
        goal,
        run_id=run_id,
        attempt_index=review_attempt,
    )
    initial_state["recipe_params"] = {
        "query": search_keyword,
        "keyword": search_keyword,
        "target_count": target_count,
        "site": site_slug,
        "task_category": task_category,
        "count_mode": search_intent.get(
            "count_mode",
            CollectionCountMode.UNSPECIFIED.value,
        ),
        "collection_intent": search_intent,
    }

    logger.info(
        "비전 작업자 그래프 시작: site=%s attempt=%s",
        site_slug,
        review_attempt,
    )
    recursion_limit = get_settings().vision.recursion_limit
    final_state, hit_recursion_limit = execute_worker_graph(
        initial_state,
        site_profile,
        recursion_limit,
        worker_runtime=worker_runtime,
        prepare_screen=(
            prepare_worker_start_screen
            if worker_runtime is None
            else lambda state, profile: prepare_worker_start_screen(
                state,
                profile,
                worker_runtime=worker_runtime,
            )
        ),
        run_graph=run_graph_with_last_state,
    )
    extracted = final_state.get("extracted_jd", {}) or {}
    is_finished = bool(final_state.get("is_finished", False))
    run_status = _worker_run_status(
        hit_recursion_limit,
        is_finished,
    )
    feedback_saved = _commit_feedback_episodes(
        final_state,
        hit_recursion_limit,
        is_finished,
        run_id=run_id,
        review_attempt=review_attempt,
    )
    submission = build_worker_submission(
        final_state,
        site=site_slug,
        keyword=search_keyword,
        run_status=run_status,
        hit_recursion_limit=hit_recursion_limit,
        persisted_count=0,
        feedback_saved=feedback_saved,
        review_attempt=review_attempt,
        run_id=run_id,
        target_count=target_count,
        task_category=task_category,
    )
    observed_job_ids = list(
        submission.get("observed_job_ids")
        or []
    )

    return {
        "submission": submission,
        "extracted_jd": extracted,
        "final_state": final_state,
        "site_slug": site_slug,
        "site_name": site_name,
        "keyword": search_keyword,
        "raw_keyword": raw_search_keyword,
        "target_count": target_count,
        "task_category": task_category,
        "search_intent": search_intent,
        "collection_intent": search_intent,
        "task_context": task_context or {},
        "run_status": run_status,
        "hit_recursion_limit": hit_recursion_limit,
        "is_finished": is_finished,
        "feedback_saved": feedback_saved,
        "recursion_limit": recursion_limit,
        "observed_job_ids": observed_job_ids,
    }


__all__ = [
    "build_limit_intermediate_report",
    "limit_report_requires_more_collection",
    "needs_human_limit_approval",
    "run_worker_once",
    "worker_review_retries",
]

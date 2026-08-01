"""구조화된 수집 요청을 비전 작업자 서비스에 연결하는 도구 어댑터."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from agent.application.collection_request_builder import normalize_target_count
from agent.application.collection_service import (
    CollectionRequest,
    CollectionService,
    build_collection_operations,
)
from agent.recipe.task_category import (
    DEFAULT_SEARCH_TASK_CATEGORY,
)
from agent.utils.model_dump import dump_model
from shared.schema.collection_intent import (
    CollectionCountMode,
    normalize_collection_intent,
)

logger = logging.getLogger(__name__)


def _search_keyword(
    *,
    query: str | None,
    company: str | None,
    tech_stack: str | None,
) -> str:
    if query:
        return query
    if company and tech_stack:
        return f"{company} {tech_stack}"
    return company or tech_stack or ""


def _run_realtime_scraping(
    company: str | None = None,
    tech_stack: str | None = None,
    site: str | None = None,
    query: str | None = None,
    target_count: int = 0,
    task_category: str = DEFAULT_SEARCH_TASK_CATEGORY,
    search_intent_resolved: bool = False,
    count_mode: str = CollectionCountMode.UNSPECIFIED.value,
    posted_date_expression: str = "",
    posted_from: str = "",
    posted_to: str = "",
    experience: str = "",
    location: str = "",
    employment_type: str = "",
    freshness_required: bool = False,
    purpose: str = "collect",
    analysis_goal: str = "",
    original_query: str = "",
    required_fields: list[str] | None = None,
    worker_runtime: Any = None,
) -> str:
    """수집 요청을 정규화해 애플리케이션 수집 서비스에 전달한다."""

    keyword = _search_keyword(
        query=query,
        company=company,
        tech_stack=tech_stack,
    )
    if not keyword:
        return json.dumps(
            {
                "message": "collection failed: missing search keyword",
                "review": {"decision": "reject"},
            },
            ensure_ascii=False,
        )

    logger.info("비전 작업자 수집 요청: keyword=%r", keyword)
    collection_intent = normalize_collection_intent(
        {
            "original_query": original_query or query or keyword,
            "site": site or "",
            "search_keyword": keyword,
            "count_mode": count_mode,
            "target_count": target_count,
            "filters": {
                "posted_date_expression": posted_date_expression,
                "posted_from": posted_from,
                "posted_to": posted_to,
                "experience": experience,
                "location": location,
                "employment_type": employment_type,
            },
            "freshness_required": freshness_required,
            "purpose": purpose,
            "analysis_goal": analysis_goal,
            "required_fields": required_fields or [],
        }
    )
    result = CollectionService(
        build_collection_operations(worker_runtime)
    ).collect(
        CollectionRequest(
            search_keyword=keyword,
            site=site,
            target_count=target_count,
            task_category=(
                task_category
                or DEFAULT_SEARCH_TASK_CATEGORY
            ),
            search_intent_resolved=search_intent_resolved,
            collection_intent=dump_model(collection_intent),
        )
    )
    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )


def _invoke_realtime_scraping(
    worker_runtime: Any,
    arguments: dict[str, Any],
) -> str:
    """주어진 비전 런타임에서 수집 도구를 한 번 실행한다."""

    from agent.application.run_context import emit_run_event, run_context
    from agent.application.run_contracts import RunPhase, RunStatus
    from agent.graph.workflow import build_graph
    from agent.runtime.vision_worker_runtime import VisionWorkerRuntime

    owned_runtime = worker_runtime is None
    runtime = worker_runtime or VisionWorkerRuntime(graph_factory=build_graph)
    context_query = arguments.get("query") or " ".join(
        part
        for part in (
            arguments.get("company"),
            arguments.get("tech_stack"),
        )
        if part
    )
    try:
        with run_context(
            query=context_query,
            prefix="collection",
        ) as (context, created):
            with runtime.execution_session():
                result_text = _run_realtime_scraping(
                    **arguments,
                    search_intent_resolved=(
                        normalize_target_count(
                            arguments.get("target_count")
                        )
                        > 0
                    ),
                    worker_runtime=runtime,
                )
            if not created:
                return result_text

            try:
                payload = json.loads(result_text)
            except (TypeError, json.JSONDecodeError):
                payload = {"message": str(result_text)}
            payload["run_id"] = context.run_id
            payload["metrics"] = context.snapshot()
            failed = str(
                payload.get("message")
                or ""
            ).startswith("collection error")
            emit_run_event(
                "run_failed" if failed else "run_completed",
                (
                    RunPhase.FAILED
                    if failed
                    else RunPhase.COMPLETED
                ),
                (
                    "수집 작업이 실패했습니다."
                    if failed
                    else "수집 작업을 완료했습니다."
                ),
                status=(
                    RunStatus.FAILED
                    if failed
                    else RunStatus.COMPLETED
                ),
            )
            return json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
    finally:
        if owned_runtime:
            runtime.close()


@tool
def realtime_scraping(
    company: str = None,
    tech_stack: str = None,
    site: str = None,
    query: str = None,
    target_count: int = 0,
    task_category: str = DEFAULT_SEARCH_TASK_CATEGORY,
    count_mode: str = CollectionCountMode.UNSPECIFIED.value,
    posted_date_expression: str = "",
    posted_from: str = "",
    posted_to: str = "",
    experience: str = "",
    location: str = "",
    employment_type: str = "",
    freshness_required: bool = False,
    purpose: str = "collect",
    analysis_goal: str = "",
    original_query: str = "",
    required_fields: list[str] | None = None,
) -> str:
    """구조화된 요청에 따라 비전 작업자로 공고를 수집하고 승인된 결과만 저장한다."""

    return _invoke_realtime_scraping(
        None,
        {
            "company": company,
            "tech_stack": tech_stack,
            "site": site,
            "query": query,
            "target_count": target_count,
            "task_category": task_category,
            "count_mode": count_mode,
            "posted_date_expression": posted_date_expression,
            "posted_from": posted_from,
            "posted_to": posted_to,
            "experience": experience,
            "location": location,
            "employment_type": employment_type,
            "freshness_required": freshness_required,
            "purpose": purpose,
            "analysis_goal": analysis_goal,
            "original_query": original_query,
            "required_fields": required_fields or [],
        },
    )


class RuntimeRealtimeScrapingTool:
    """ApplicationRuntime의 비전 자원을 재사용하는 수집 도구 어댑터."""

    name = realtime_scraping.name
    description = realtime_scraping.description
    args_schema = realtime_scraping.args_schema

    def __init__(self, worker_runtime: Any):
        self.worker_runtime = worker_runtime

    def invoke(self, arguments: dict[str, Any]) -> str:
        return _invoke_realtime_scraping(
            self.worker_runtime,
            dict(arguments),
        )


def build_runtime_realtime_scraping_tool(
    worker_runtime: Any,
) -> RuntimeRealtimeScrapingTool:
    return RuntimeRealtimeScrapingTool(worker_runtime)


__all__ = [
    "RuntimeRealtimeScrapingTool",
    "build_runtime_realtime_scraping_tool",
    "realtime_scraping",
]

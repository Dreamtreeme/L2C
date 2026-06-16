from __future__ import annotations

import os
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.graph.commander_state import CommanderState
from agent.tools.realtime_scraping import (
    _close_browser_after_run,
    commit_worker_review,
    persist_accepted_worker_result,
    run_worker_once,
)
from agent.tools.sqlite_query import sqlite_query
from agent.utils.logger import logger


def _max_review_retries() -> int:
    try:
        return max(0, int(os.getenv("COMMANDER_WORKER_REVIEW_RETRIES", "1")))
    except ValueError:
        return 1


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _select_sites_for_query(query: str, sites: list[dict[str, Any]]) -> list[str]:
    """Select explicit sites from the query; otherwise fan out to all enabled profiles."""
    query_key = _normalize(query)
    explicit: list[str] = []
    for entry in sites:
        slug = str(entry.get("slug") or "")
        names = [slug, str(entry.get("display_name") or "")]
        names.extend(str(domain or "") for domain in entry.get("domains", []) or [])
        if any(_normalize(name) and _normalize(name) in query_key for name in names):
            explicit.append(slug)
    selected = explicit or [str(entry.get("slug")) for entry in sites if entry.get("slug")]
    return list(dict.fromkeys(selected))


def commander_plan_node(state: CommanderState) -> dict:
    from agent.sites import list_supported_sites

    existing_queue = [str(site) for site in state.get("site_queue", []) if site]
    if existing_queue:
        site_queue = existing_queue
    else:
        site_queue = _select_sites_for_query(state.get("user_query", ""), list_supported_sites(enabled_only=True))
    logger.info("[commander_graph] planned sites: %s", site_queue)
    return {
        "site_queue": site_queue,
        "current_site_index": int(state.get("current_site_index", 0) or 0),
        "max_review_retries": int(state.get("max_review_retries", _max_review_retries()) or 0),
        "review_attempt": 0,
        "review_feedback": "",
        "done": False,
    }


def select_site_node(state: CommanderState) -> dict:
    site_queue = list(state.get("site_queue", []) or [])
    index = int(state.get("current_site_index", 0) or 0)
    if index >= len(site_queue):
        return {"done": True, "current_site": "", "current_run_id": ""}
    return {
        "current_site": site_queue[index],
        "review_attempt": 0,
        "review_feedback": "",
        "current_run_id": "",
        "current_worker_result": {},
        "current_submission": {},
        "current_review": {},
        "current_submission_id": "",
        "done": False,
    }


def route_after_select(state: CommanderState) -> str:
    return "query_db" if state.get("done") else "run_worker"


def run_worker_node(state: CommanderState) -> dict:
    try:
        worker_result = run_worker_once(
            state.get("user_query", ""),
            site=state.get("current_site") or None,
            review_feedback=state.get("review_feedback") or None,
            review_attempt=int(state.get("review_attempt", 0) or 0),
            run_id=state.get("current_run_id") or None,
        )
    finally:
        _close_browser_after_run()
    submission = worker_result.get("submission", {}) or {}
    return {
        "current_worker_result": worker_result,
        "current_submission": submission,
        "current_run_id": submission.get("run_id", state.get("current_run_id", "")),
    }


def review_submission_node(state: CommanderState) -> dict:
    submission = dict(state.get("current_submission") or {})
    review, submission_id = commit_worker_review(submission, source="commander_graph")
    review_record = {
        "site": state.get("current_site", ""),
        "attempt": int(state.get("review_attempt", 0) or 0),
        "submission_id": submission_id,
        "review": review,
    }
    return {
        "current_review": review,
        "current_submission_id": submission_id,
        "worker_submissions": [submission],
        "reviews": [review_record],
    }


def route_after_review(state: CommanderState) -> str:
    review = state.get("current_review", {}) or {}
    if review.get("decision") == "accept":
        return "persist"
    if review.get("decision") == "revise" and int(state.get("review_attempt", 0) or 0) < int(state.get("max_review_retries", 0) or 0):
        return "retry"
    return "fail"


def prepare_retry_node(state: CommanderState) -> dict:
    from agent.recipe.reviewer import render_review_feedback

    return {
        "review_attempt": int(state.get("review_attempt", 0) or 0) + 1,
        "review_feedback": render_review_feedback(state.get("current_review", {}) or {}),
    }


def persist_accepted_node(state: CommanderState) -> dict:
    worker_result = dict(state.get("current_worker_result") or {})
    review = dict(state.get("current_review") or {})
    persisted_count, submission, persisted_review, submission_id = persist_accepted_worker_result(worker_result, review, source="commander_graph")
    return {
        "current_worker_result": worker_result,
        "current_submission": submission,
        "current_review": persisted_review,
        "current_submission_id": submission_id or state.get("current_submission_id", ""),
        "accepted_sites": [state.get("current_site", "")],
        "current_site_index": int(state.get("current_site_index", 0) or 0) + 1,
    }


def mark_failed_node(state: CommanderState) -> dict:
    return {
        "failed_sites": [
            {
                "site": state.get("current_site", ""),
                "attempt": int(state.get("review_attempt", 0) or 0),
                "review": state.get("current_review", {}) or {},
            }
        ],
        "current_site_index": int(state.get("current_site_index", 0) or 0) + 1,
    }


def query_recent_jobs() -> str:
    sql = (
        "SELECT id, url, company_name, position, raw_ocr_text "
        "FROM jobs ORDER BY updated_at DESC LIMIT 20"
    )
    return sqlite_query.invoke({"sql_query": sql})


def query_db_node(state: CommanderState) -> dict:
    return {"db_results": query_recent_jobs()}


def summarize_node(state: CommanderState) -> dict:
    accepted = [site for site in state.get("accepted_sites", []) if site]
    failed = [item for item in state.get("failed_sites", []) if item]
    answer = (
        "Commander graph completed. "
        f"accepted_sites={len(accepted)} ({', '.join(accepted) or 'none'}), "
        f"failed_sites={len(failed)}.\n\n"
        f"DB evidence:\n{state.get('db_results', '')}"
    )
    return {"final_answer": answer, "done": True}


def build_commander_graph():
    workflow = StateGraph(CommanderState)
    workflow.add_node("plan_sites", commander_plan_node)
    workflow.add_node("select_site", select_site_node)
    workflow.add_node("run_worker", run_worker_node)
    workflow.add_node("review_submission", review_submission_node)
    workflow.add_node("prepare_retry", prepare_retry_node)
    workflow.add_node("persist_accepted", persist_accepted_node)
    workflow.add_node("mark_failed", mark_failed_node)
    workflow.add_node("query_db", query_db_node)
    workflow.add_node("summarize", summarize_node)

    workflow.add_edge(START, "plan_sites")
    workflow.add_edge("plan_sites", "select_site")
    workflow.add_conditional_edges(
        "select_site",
        route_after_select,
        {"run_worker": "run_worker", "query_db": "query_db"},
    )
    workflow.add_edge("run_worker", "review_submission")
    workflow.add_conditional_edges(
        "review_submission",
        route_after_review,
        {"retry": "prepare_retry", "persist": "persist_accepted", "fail": "mark_failed"},
    )
    workflow.add_edge("prepare_retry", "run_worker")
    workflow.add_edge("persist_accepted", "select_site")
    workflow.add_edge("mark_failed", "select_site")
    workflow.add_edge("query_db", "summarize")
    workflow.add_edge("summarize", END)
    return workflow.compile()


def run_commander_graph(query: str, site_queue: list[str] | None = None) -> dict:
    initial: CommanderState = {
        "user_query": query,
        "site_queue": list(site_queue or []),
        "current_site_index": 0,
        "worker_submissions": [],
        "reviews": [],
        "accepted_sites": [],
        "failed_sites": [],
    }
    return build_commander_graph().invoke(initial)
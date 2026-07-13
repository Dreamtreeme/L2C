from __future__ import annotations

import os
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.graph.commander_state import CommanderState
from agent.application.worker_execution_service import worker_execution_session
from agent.tools.realtime_scraping import (
    _close_browser_after_run,
    build_limit_intermediate_report,
    commit_worker_review,
    limit_report_requires_more_collection,
    needs_human_limit_approval,
    persist_accepted_worker_result,
    run_worker_once,
)
from agent.tools.sqlite_query import sqlite_query
from agent.tools.task_triage import research_public_web, triage_user_task
from agent.utils.logger import logger
from agent.utils.model_dump import dump_model


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


def task_triage_node(state: CommanderState) -> dict:
    triage = triage_user_task(state.get("user_query", ""))
    payload = dump_model(triage)
    logger.info("[commander_graph] task triage: %s", payload)
    return {
        "task_triage": payload,
        "task_context": {
            "triage": payload,
            "allowed_actions": ["read", "navigate", "search"],
            "blocked_actions": _blocked_actions_for_triage(payload),
        },
        "pending_human_approval": False,
        "human_approval_reason": "",
        "human_approval_request": {},
    }


def research_node(state: CommanderState) -> dict:
    triage = dict(state.get("task_triage") or {})
    report = research_public_web(state.get("user_query", ""), triage)
    report_payload = dump_model(report)
    context = dict(state.get("task_context") or {})
    context["research_report"] = report_payload
    logger.info("[commander_graph] public research status=%s", report_payload.get("status"))
    return {
        "research_report": report_payload,
        "task_context": context,
    }


def _blocked_actions_for_triage(triage: dict[str, Any]) -> list[str]:
    blocked = ["submit", "agree", "pay", "transfer", "apply", "enter_password", "enter_personal_data"]
    if triage.get("risk_level") == "sensitive" or triage.get("sensitive_steps"):
        blocked.extend(str(step) for step in triage.get("sensitive_steps") or [])
    return list(dict.fromkeys(blocked))


def _task_needs_human_gate(state: CommanderState) -> bool:
    triage = dict(state.get("task_triage") or {})
    report = dict(state.get("research_report") or {})
    if triage.get("goal_type") == "job_collection" and not triage.get("requires_research"):
        return False
    if triage.get("risk_level") == "sensitive":
        return True
    if report.get("needs_user_confirmation") or report.get("needs_user_choice"):
        return True
    if triage.get("requires_research") and report.get("status") != "completed":
        return True
    return triage.get("known_or_unknown") == "unknown" and triage.get("goal_type") != "job_collection"


def route_after_research(state: CommanderState) -> str:
    return "human_gate" if _task_needs_human_gate(state) else "plan_sites"


def human_gate_node(state: CommanderState) -> dict:
    triage = dict(state.get("task_triage") or {})
    report = dict(state.get("research_report") or {})
    options = report.get("official_paths") or report.get("possible_sites") or []
    request = {
        "status": "needs_task_approval",
        "reason": _human_gate_reason(triage, report),
        "triage": triage,
        "research_report": report,
        "options": options[:8] if isinstance(options, list) else [],
        "blocked_actions": _blocked_actions_for_triage(triage),
        "question": "Choose an official route and confirm before login, personal data, agreement, application, payment, or other sensitive steps.",
    }
    return {
        "pending_human_approval": True,
        "human_approval_reason": request["reason"],
        "human_approval_request": request,
        "intermediate_report": request,
        "done": True,
    }


def _human_gate_reason(triage: dict[str, Any], report: dict[str, Any]) -> str:
    if triage.get("risk_level") == "sensitive":
        return "sensitive_task_requires_human_confirmation"
    if report.get("needs_user_choice"):
        return "multiple_public_routes_require_user_choice"
    if report.get("status") == "failed":
        return "public_research_failed_before_execution"
    return "unknown_task_requires_user_confirmation"


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
            task_context=dict(state.get("task_context") or {}),
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
    recursion_limit = int(worker_result.get("recursion_limit") or 60)
    target_count = int(worker_result.get("target_count") or submission.get("target_count") or 0)
    base_needs_approval = needs_human_limit_approval(
        hit_recursion_limit=bool(worker_result.get("hit_recursion_limit", False)),
        is_finished=bool(worker_result.get("is_finished", False)),
        persisted_count=persisted_count,
        target_count=target_count,
    )
    intermediate_report = (
        build_limit_intermediate_report(
            worker_result,
            submission,
            persisted_count=persisted_count,
            current_limit=recursion_limit,
            target_count=target_count,
        )
        if base_needs_approval
        else {}
    )
    needs_approval = base_needs_approval and limit_report_requires_more_collection(intermediate_report)
    index_delta = 0 if needs_approval else 1
    return {
        "current_worker_result": worker_result,
        "current_submission": submission,
        "current_review": persisted_review,
        "current_submission_id": submission_id or state.get("current_submission_id", ""),
        "accepted_sites": [state.get("current_site", "")],
        "current_site_index": int(state.get("current_site_index", 0) or 0) + index_delta,
        "pending_human_approval": needs_approval,
        "intermediate_report": intermediate_report,
    }


def route_after_persist(state: CommanderState) -> str:
    return "query_db" if state.get("pending_human_approval") else "select_site"


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


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _submission_job_urls(submission: dict[str, Any]) -> list[str]:
    summary = submission.get("extracted_summary") if isinstance(submission, dict) else {}
    jobs = summary.get("jobs") if isinstance(summary, dict) else []
    urls: list[str] = []
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        url = str(job.get("url") or "").strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def query_recent_jobs(submission: dict[str, Any] | None = None) -> str:
    urls = _submission_job_urls(submission or {})
    if urls:
        limited = urls[:20]
        url_list = ", ".join(_sql_literal(url) for url in limited)
        order_cases = " ".join(f"WHEN {_sql_literal(url)} THEN {idx}" for idx, url in enumerate(limited))
        sql = (
            "SELECT id, url, company_name, position, raw_ocr_text "
            f"FROM jobs WHERE url IN ({url_list}) "
            f"ORDER BY CASE url {order_cases} ELSE {len(limited)} END"
        )
    else:
        sql = (
            "SELECT id, url, company_name, position, raw_ocr_text "
            "FROM jobs ORDER BY updated_at DESC LIMIT 20"
        )
    return sqlite_query.invoke({"sql_query": sql})


def query_db_node(state: CommanderState) -> dict:
    try:
        db_results = query_recent_jobs(dict(state.get("current_submission") or {}))
    except TypeError:
        db_results = query_recent_jobs()
    return {"db_results": db_results}


def summarize_node(state: CommanderState) -> dict:
    if state.get("pending_human_approval"):
        report = dict(state.get("intermediate_report") or {})
        if report.get("status") == "needs_task_approval":
            options = report.get("options") or []
            option_lines = []
            for idx, option in enumerate(options[:8], start=1):
                if not isinstance(option, dict):
                    continue
                title = option.get("title") or option.get("domain") or "(untitled)"
                url = option.get("url") or ""
                option_lines.append(f"{idx}. {title} {url}".strip())
            if not option_lines:
                option_lines.append("none")
            triage = report.get("triage") if isinstance(report.get("triage"), dict) else {}
            sensitive = triage.get("sensitive_steps") or report.get("blocked_actions") or []
            answer = (
                "Human confirmation required before autonomous execution.\n"
                f"Reason: {report.get('reason', '')}\n"
                f"Goal type: {triage.get('goal_type', '')}\n"
                f"Risk level: {triage.get('risk_level', '')}\n"
                f"Research status: {(report.get('research_report') or {}).get('status', '')}\n\n"
                "Candidate public routes:\n"
                + "\n".join(option_lines)
                + "\n\nSensitive or blocked steps:\n"
                + "\n".join(f"- {item}" for item in sensitive[:12])
                + "\n\nPlease choose the route/site and explicitly confirm before any login, personal data, agreement, application, payment, account, finance, or legal-effect step."
            )
            return {"final_answer": answer, "done": True}

        jobs = report.get("jobs") or []
        job_lines = []
        for idx, job in enumerate(jobs[:8], start=1):
            if not isinstance(job, dict):
                continue
            title = " - ".join(part for part in [job.get("company"), job.get("position")] if part)
            url = job.get("url") or ""
            job_lines.append(f"{idx}. {title or '(title missing)'} {url}".strip())
        if not job_lines:
            job_lines.append("none")

        remaining = report.get("remaining_plan_preview") or []
        remaining_lines = [f"- {item}" for item in remaining[:4]] or ["- none"]
        answer = (
            "중간보고: 작업자가 recursion limit에 도달해 일시 중단했습니다.\n"
            f"사이트: {state.get('current_site', '')}\n"
            f"수집/저장: {report.get('collected_count', 0)}건 / {report.get('persisted_count', 0)}건\n"
            f"현재 limit: {report.get('current_recursion_limit', '')}\n"
            f"제안 limit: {report.get('suggested_recursion_limit', '')}\n"
            f"현재 URL: {report.get('current_url', '') or 'unknown'}\n"
            f"계획 진행: {report.get('current_plan_step', 0)} / {report.get('total_plan_steps', 0)}\n\n"
            "현재 저장된 공고:\n"
            + "\n".join(job_lines)
            + "\n\n남은 계획 미리보기:\n"
            + "\n".join(remaining_lines)
            + "\n\nlimit을 늘려 계속 진행할까요?\n\n"
            f"DB evidence:\n{state.get('db_results', '')}"
        )
        return {"final_answer": answer, "done": True}

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
    workflow.add_node("task_triage", task_triage_node)
    workflow.add_node("research", research_node)
    workflow.add_node("human_gate", human_gate_node)
    workflow.add_node("plan_sites", commander_plan_node)
    workflow.add_node("select_site", select_site_node)
    workflow.add_node("run_worker", run_worker_node)
    workflow.add_node("review_submission", review_submission_node)
    workflow.add_node("prepare_retry", prepare_retry_node)
    workflow.add_node("persist_accepted", persist_accepted_node)
    workflow.add_node("mark_failed", mark_failed_node)
    workflow.add_node("query_db", query_db_node)
    workflow.add_node("summarize", summarize_node)

    workflow.add_edge(START, "task_triage")
    workflow.add_edge("task_triage", "research")
    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {"human_gate": "human_gate", "plan_sites": "plan_sites"},
    )
    workflow.add_edge("human_gate", "summarize")
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
    workflow.add_conditional_edges(
        "persist_accepted",
        route_after_persist,
        {"select_site": "select_site", "query_db": "query_db"},
    )
    workflow.add_edge("mark_failed", "select_site")
    workflow.add_edge("query_db", "summarize")
    workflow.add_edge("summarize", END)
    return workflow.compile()


def run_commander_graph(
    query: str,
    site_queue: list[str] | None = None,
    *,
    run_id: str | None = None,
    event_sink=None,
) -> dict:
    from agent.application.run_context import emit_run_event, run_context
    from agent.application.run_contracts import RunPhase, RunStatus

    initial: CommanderState = {
        "user_query": query,
        "site_queue": list(site_queue or []),
        "current_site_index": 0,
        "worker_submissions": [],
        "reviews": [],
        "accepted_sites": [],
        "failed_sites": [],
        "pending_human_approval": False,
        "human_approval_reason": "",
        "human_approval_request": {},
        "intermediate_report": {},
        "task_triage": {},
        "research_report": {},
        "task_context": {},
    }
    with run_context(
        run_id=run_id,
        query=query,
        event_sink=event_sink,
        prefix="commander",
    ) as (context, created):
        with worker_execution_session():
            result = build_commander_graph().invoke(initial)
        if created:
            result["run_id"] = context.run_id
            result["metrics"] = context.snapshot()
            status = (
                RunStatus.WAITING_APPROVAL
                if result.get("pending_human_approval")
                else RunStatus.COMPLETED
            )
            emit_run_event(
                "approval_required" if status == RunStatus.WAITING_APPROVAL else "run_completed",
                RunPhase.REVIEW if status == RunStatus.WAITING_APPROVAL else RunPhase.COMPLETED,
                "사용자 승인이 필요합니다."
                if status == RunStatus.WAITING_APPROVAL
                else "지휘자 작업을 완료했습니다.",
                status=status,
            )
        return result

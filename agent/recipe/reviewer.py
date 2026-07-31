"""Commander/critic review helpers for child vision worker submissions.

The script layer only validates shape and observable facts. Semantic acceptance can
be delegated to an LLM review pass, and active Reflex promotion is intentionally
kept out of this module.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.recipe.text_utils import site_of
from agent.runtime.job_collection import job_items as _job_items
from agent.recipe.task_category import normalize_task_category
from agent.utils.job_fields import JOB_FIELD_ALIASES, deterministic_report_item, first_present, summary_text
from agent.utils.model_dump import dump_model
from shared.schema.feedback_schema import CommanderReview, SubmissionIssue, WorkerSubmission


class ReportJobSummaryItem(BaseModel):
    company: str = ""
    position: str = ""
    url: str = ""
    field_count: int = 0


class ReportJobSummary(BaseModel):
    jobs: list[ReportJobSummaryItem] = Field(default_factory=list)


def _empty_report_summary(jobs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [deterministic_report_item(job) for job in jobs[:limit]]


def _semantic_job_evidence(jobs: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    fields = (
        "company_name",
        "position",
        "url",
        "location",
        "employment_type",
        "posted_at",
        "deadline",
        "requirements",
        "main_tasks",
    )
    return [
        {
            field: summary_text(first_present(job, JOB_FIELD_ALIASES.get(field, [field])))[:1200]
            for field in fields
        }
        for job in jobs[:limit]
    ]


def _report_summary_mode() -> str:
    mode = get_settings().recipe.worker_summary_mode.strip().lower()
    return mode if mode in {"deterministic", "llm", "off"} else "deterministic"


def _llm_job_summary(jobs: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from agent.prompts.trust_boundary import external_content_contract_en

    compact_jobs = [
        {
            "index": idx,
            "field_count": len(job.keys()),
            "raw_job": job,
        }
        for idx, job in enumerate(jobs[:limit])
    ]
    from agent.application.model_policy import lightweight_model_name

    model_name = lightweight_model_name("VISION_WORKER_SUMMARY_MODEL")
    from agent.application.model_clients import get_structured_google_model

    llm = get_structured_google_model(
        model_name,
        ReportJobSummary,
        temperature=0.0,
        execution_role="lightweight",
    )
    messages = [
        SystemMessage(
            content=(
                external_content_contract_en()
                + "\nYou normalize job postings that were already extracted by a vision worker. "
                "Read field names in any language, including Korean. "
                "Return one summary item per input job in the same order. "
                "Do not invent missing facts; use an empty string when a value is unknown."
            )
        ),
        HumanMessage(content=json.dumps({"jobs": compact_jobs}, ensure_ascii=False, indent=2)),
    ]
    from agent.application.run_context import invoke_with_metrics

    response = invoke_with_metrics(
        llm,
        messages,
        "worker_summary",
        stream=True,
    )
    summary = dump_model(response)
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(summary.get("jobs") or []):
        if not isinstance(item, dict):
            continue
        source_job = compact_jobs[idx] if idx < len(compact_jobs) else {}
        out.append(
            {
                "company": str(item.get("company") or ""),
                "position": str(item.get("position") or ""),
                "url": str(item.get("url") or ""),
                "field_count": int(item.get("field_count") or source_job.get("field_count") or 0),
            }
        )
    if len(out) < len(compact_jobs):
        out.extend(_empty_report_summary([job["raw_job"] for job in compact_jobs[len(out):]], limit))
    return out[:limit]


def _report_job_summary(jobs: list[dict[str, Any]], limit: int = 10) -> tuple[list[dict[str, Any]], str, str]:
    if not jobs:
        return [], "none", ""
    mode = _report_summary_mode()
    if mode in {"deterministic", "off"}:
        return _empty_report_summary(jobs, limit), "disabled", ""
    try:
        return _llm_job_summary(jobs, limit=limit), "llm", ""
    except Exception as exc:  # pragma: no cover - provider failures are best-effort
        return _empty_report_summary(jobs, limit), "llm_failed", str(exc)[:200]


def new_worker_run_id() -> str:
    return f"worker-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"


def build_worker_submission(
    final_state: dict[str, Any],
    *,
    site: str = "",
    keyword: str = "",
    run_status: str = "",
    hit_recursion_limit: bool = False,
    persisted_count: int = 0,
    feedback_saved: int = 0,
    review_attempt: int = 0,
    run_id: str | None = None,
    target_count: int = 0,
    task_category: str = "",
) -> dict[str, Any]:
    """작업자 그래프 실행 결과를 구조화된 제출물(WorkerSubmission)로 만든다."""
    extracted_jd = final_state.get("extracted_jd", {}) or {}
    jobs = _job_items(extracted_jd)
    current_url = final_state.get("current_url", "") or ""
    resolved_site = site or site_of(current_url) or "unknown"
    recipe_params = final_state.get("recipe_params", {}) if isinstance(final_state.get("recipe_params"), dict) else {}
    resolved_task_category = normalize_task_category(task_category or recipe_params.get("task_category") or "")
    run_id = run_id or new_worker_run_id()
    report_jobs, report_source, report_error = _report_job_summary(jobs)
    recorded_steps = list(final_state.get("recorded_steps", []) or [])
    feedback_episodes = list(final_state.get("feedback_episodes", []) or [])
    transition_records = list(
        final_state.get("transition_records", []) or []
    )
    observed_job_ids = sorted(
        {
            int(item["job_id"])
            for item in (final_state.get("job_card_queue", []) or [])
            if isinstance(item, dict)
            and item.get("status") == "skipped"
            and str(item.get("job_id") or "").isdigit()
            and int(item["job_id"]) > 0
        }
    )
    extracted_summary = {
        "has_data": bool(jobs),
        "job_count": len(jobs),
        "observed_job_count": len(observed_job_ids),
        "jobs": report_jobs,
        "summary_source": report_source,
        "summary_error": report_error,
        "current_url": current_url,
        "action_count": len(final_state.get("action_history", []) or []),
        "job_results_availability": dict(final_state.get("job_results_availability", {}) or {}),
    }
    from agent.recipe.skill_metadata import build_skill_metadata_evidence

    skill_metadata_evidence = build_skill_metadata_evidence(
        goal=final_state.get("goal", "") or "",
        site=resolved_site,
        task_category=resolved_task_category,
        keyword=keyword,
        target_count=int(target_count or 0),
        recorded_steps=recorded_steps,
        feedback_episodes=feedback_episodes,
        extracted_summary=extracted_summary,
    )
    submission = WorkerSubmission(
        run_id=run_id,
        goal=final_state.get("goal", "") or "",
        site=resolved_site,
        task_category=resolved_task_category,
        keyword=keyword,
        run_status=run_status,
        review_attempt=review_attempt,
        is_finished=bool(final_state.get("is_finished", False)),
        hit_recursion_limit=bool(hit_recursion_limit),
        collected_count=len(jobs),
        observed_job_ids=observed_job_ids,
        target_count=int(target_count or 0),
        persisted_count=int(persisted_count or 0),
        feedback_saved=int(feedback_saved or 0),
        recorded_steps=recorded_steps,
        feedback_episodes=feedback_episodes,
        transition_records=transition_records,
        skill_metadata_evidence=skill_metadata_evidence,
        collection_intent=dict(recipe_params.get("collection_intent") or {}),
        semantic_evidence=_semantic_job_evidence(jobs),
        extracted_summary=extracted_summary,
        worker_notes="submitted after autonomous/reflex worker run",
    )
    return dump_model(submission)


def validate_submission_shape(submission: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate structure only; do not make semantic promotion decisions here."""
    issues: list[SubmissionIssue] = []

    def add(field: str, reason: str, severity: str = "error") -> None:
        issues.append(SubmissionIssue(field=field, reason=reason, severity=severity))

    if not submission.get("goal"):
        add("goal", "missing worker goal")
    if not submission.get("site"):
        add("site", "missing site identifier", "warning")
    if (
        int(submission.get("collected_count") or 0) <= 0
        and not submission.get("observed_job_ids")
    ):
        add("extracted_summary", "worker did not submit any collected job data")

    recorded_steps = submission.get("recorded_steps") or []
    feedback_episodes = submission.get("feedback_episodes") or []
    if not recorded_steps:
        add("recorded_steps", "no UI steps were recorded for later recipe review", "warning")
    if not feedback_episodes:
        add("feedback_episodes", "no action feedback episodes were recorded", "warning")

    for idx, step in enumerate(recorded_steps):
        if not isinstance(step, dict):
            add(f"recorded_steps[{idx}]", "step is not an object")
            continue
        action = step.get("action")
        if not action:
            add(f"recorded_steps[{idx}].action", "missing action")
        if action in {"click_marker", "type_in_marker"} and not step.get("target"):
            add(f"recorded_steps[{idx}].target", "target action is missing target metadata", "warning")

    for idx, episode in enumerate(feedback_episodes):
        if not isinstance(episode, dict):
            add(f"feedback_episodes[{idx}]", "episode is not an object", "warning")
            continue
        proposal = episode.get("proposal") if isinstance(episode.get("proposal"), dict) else {}
        feedback = episode.get("feedback") if isinstance(episode.get("feedback"), dict) else {}
        if not proposal.get("action"):
            add(f"feedback_episodes[{idx}].proposal.action", "missing proposed action", "warning")
        if not feedback.get("label"):
            add(f"feedback_episodes[{idx}].feedback.label", "missing feedback label", "warning")

    return [dump_model(issue) for issue in issues]


def shape_review(submission: dict[str, Any], issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    issues = issues if issues is not None else validate_submission_shape(submission)
    errors = [issue for issue in issues if issue.get("severity") == "error"]
    if errors:
        reasons = [f"{issue['field']}: {issue['reason']}" for issue in errors]
        review = CommanderReview(
            decision="revise",
            reasons=reasons,
            feedback_to_worker=(
                "Revise the worker submission before it can be accepted. "
                "Return collected job data with required fields, keep the OCR state/action evidence, "
                "and avoid claiming success without extracted jobs. Issues: " + "; ".join(reasons)
            ),
            accept_collected_data=False,
            continue_collection=True,
            recipe_candidate=False,
            confidence=0.78,
        )
        return dump_model(review)

    warnings = [f"{issue['field']}: {issue['reason']}" for issue in issues]
    review = CommanderReview(
        decision="accept",
        reasons=warnings or ["submission shape is valid"],
        feedback_to_worker="",
        accept_collected_data=True,
        continue_collection=False,
        recipe_candidate=bool(submission.get("recorded_steps")) and int(submission.get("collected_count") or 0) > 0,
        confidence=0.62 if warnings else 0.72,
    )
    return dump_model(review)


def build_worker_review_payload(
    submission: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """실시간 승인에 필요한 결과만 남기고 학습용 실행 원본은 제외한다."""

    summary = submission.get("extracted_summary")
    summary = summary if isinstance(summary, dict) else {}
    intent = submission.get("collection_intent")
    intent = intent if isinstance(intent, dict) else {}
    feedback = [
        item
        for item in (submission.get("feedback_episodes") or [])
        if isinstance(item, dict)
    ]
    transitions = [
        item
        for item in (submission.get("transition_records") or [])
        if isinstance(item, dict)
    ]
    steps = [
        item
        for item in (submission.get("recorded_steps") or [])
        if isinstance(item, dict)
    ]
    feedback_counts = Counter(
        str((item.get("feedback") or {}).get("label") or "unknown")
        for item in feedback
        if isinstance(item.get("feedback"), dict)
    )
    action_counts = Counter(str(item.get("action") or "unknown") for item in steps)
    failure_feedback = []
    for item in feedback:
        result = item.get("feedback") if isinstance(item.get("feedback"), dict) else {}
        if result.get("label") not in {"wrong_target", "no_effect", "loop_risk", "error"}:
            continue
        proposal = item.get("proposal") if isinstance(item.get("proposal"), dict) else {}
        failure_feedback.append(
            {
                "seq": item.get("seq"),
                "action": proposal.get("action") or "",
                "label": result.get("label") or "",
                "reason": str(result.get("reason") or "")[:500],
            }
        )
    transition_failures = []
    for item in transitions:
        status = str(item.get("status") or "").lower()
        if status in {"", "success", "passed", "complete", "completed"}:
            continue
        transition_failures.append(
            {
                "action_seq": item.get("action_seq"),
                "action": item.get("action") or "",
                "status": status,
                "reason": str(item.get("reason") or "")[:500],
            }
        )
    user_request = str(intent.get("original_query") or "").strip()
    return {
        "request": {
            "user_request": user_request or str(submission.get("goal") or "")[:1200],
            "site": submission.get("site") or "",
            "keyword": submission.get("keyword") or "",
            "task_category": submission.get("task_category") or "",
            "target_count": int(submission.get("target_count") or 0),
            "collection_intent": intent,
        },
        "execution": {
            "run_status": submission.get("run_status") or "",
            "is_finished": bool(submission.get("is_finished", False)),
            "hit_recursion_limit": bool(submission.get("hit_recursion_limit", False)),
            "collected_count": int(submission.get("collected_count") or 0),
            "observed_job_ids": list(submission.get("observed_job_ids") or [])[:20],
            "persisted_count": int(submission.get("persisted_count") or 0),
            "job_results_availability": dict(summary.get("job_results_availability") or {}),
            "action_count": int(summary.get("action_count") or len(steps)),
            "action_counts": dict(action_counts),
            "feedback_counts": dict(feedback_counts),
            "failure_feedback": failure_feedback[:12],
            "transition_failures": transition_failures[:12],
        },
        "jobs": list(submission.get("semantic_evidence") or [])[:20],
        "job_summary": list(summary.get("jobs") or [])[:20],
        "shape_issues": issues,
        "review_rules": [
            "수집된 공고의 관련성과 저장 가능 여부를 실행 완료 여부와 별도로 판단한다.",
            "관련 있는 유효 공고가 있으면 목표 개수 미달이어도 accept_collected_data=true로 둔다.",
            "같은 검색 범위에서 추가 행동이 꼭 필요할 때만 continue_collection=true로 둔다.",
            "화면에 확인된 전체 결과를 모두 처리했다면 목표 개수 미달만으로 재실행하지 않는다.",
            "레시피 승격은 후처리 Critic이 판단하므로 여기서는 후보 여부만 표시한다.",
        ],
    }


def _llm_review(
    submission: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from agent.application.model_clients import get_structured_google_model
    from agent.prompts.trust_boundary import external_content_contract_en

    from agent.application.model_policy import commander_model_name

    model_name = commander_model_name("VISION_WORKER_REVIEW_MODEL")
    llm = get_structured_google_model(
        model_name,
        CommanderReview,
        temperature=0.0,
        execution_role="commander",
    )
    compact = build_worker_review_payload(submission, issues)
    messages = [
        SystemMessage(
            content=(
                external_content_contract_en()
                + "\nYou are the commander reviewing a child vision worker submission. "
                "Return only the structured CommanderReview schema. Decide data acceptance separately from "
                "whether another collection attempt is needed. Set accept_collected_data=true for relevant, "
                "persistable jobs even when the requested count was not reached. Set continue_collection=true "
                "only when another attempt in the same search scope is necessary."
            )
        ),
        HumanMessage(content=json.dumps(compact, ensure_ascii=False, indent=2)),
    ]
    from agent.application.run_context import invoke_with_metrics

    response = invoke_with_metrics(llm, messages, "worker_review")
    review = dump_model(response)
    return dump_model(CommanderReview(**review))


def review_worker_submission(submission: dict[str, Any]) -> dict[str, Any]:
    """Review a submission. Shape is always checked; semantic LLM review is opt-in."""
    issues = validate_submission_shape(submission)
    fallback = shape_review(submission, issues)
    if fallback.get("decision") != "accept":
        return fallback
    mode = get_settings().recipe.worker_review_mode.strip().lower()
    if mode != "llm":
        return fallback
    try:
        return _llm_review(submission, issues)
    except Exception as exc:  # pragma: no cover - keep worker completion resilient
        fallback = dict(fallback)
        fallback.setdefault("reasons", []).append(f"llm_review_unavailable: {str(exc)[:200]}")
        return fallback


def render_review_feedback(review: dict[str, Any]) -> str:
    if review.get("decision") == "accept":
        return ""
    feedback = review.get("feedback_to_worker") or "; ".join(review.get("reasons") or [])
    return (
        "Commander review rejected the previous worker submission. "
        "Use this feedback on the next attempt: " + feedback
    )


__all__ = [
    "build_worker_review_payload",
    "build_worker_submission",
    "new_worker_run_id",
    "render_review_feedback",
    "review_worker_submission",
    "shape_review",
    "validate_submission_shape",
]

"""Commander/critic review helpers for child vision worker submissions.

The script layer only validates shape and observable facts. Semantic acceptance can
be delegated to an LLM review pass, and active Reflex promotion is intentionally
kept out of this module.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent.recipe.state_key import site_of
from shared.schema.feedback_schema import CommanderReview, SubmissionIssue, WorkerSubmission


class ReportJobSummaryItem(BaseModel):
    company: str = ""
    position: str = ""
    url: str = ""
    field_count: int = 0


class ReportJobSummary(BaseModel):
    jobs: list[ReportJobSummaryItem] = Field(default_factory=list)


def _dump_model(model) -> dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _looks_like_job_list(value: Any) -> bool:
    return isinstance(value, list) and any(isinstance(item, dict) and item for item in value)


def _job_items(extracted_jd: Any) -> list[dict[str, Any]]:
    if not isinstance(extracted_jd, dict) or not extracted_jd:
        return []
    for value in extracted_jd.values():
        if _looks_like_job_list(value):
            return [item for item in value if isinstance(item, dict) and item]
    return [extracted_jd] if extracted_jd else []


def _empty_report_summary(jobs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "company": "",
            "position": "",
            "url": "",
            "field_count": len(job.keys()),
        }
        for job in jobs[:limit]
    ]


def _report_summary_mode() -> str:
    mode = os.getenv("VISION_WORKER_SUMMARY_MODE", "llm").strip().lower()
    return mode if mode in {"llm", "off"} else "llm"


def _llm_job_summary(jobs: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    compact_jobs = [
        {
            "index": idx,
            "field_count": len(job.keys()),
            "raw_job": job,
        }
        for idx, job in enumerate(jobs[:limit])
    ]
    model_name = os.getenv("VISION_WORKER_SUMMARY_MODEL", os.getenv("VISION_WORKER_REVIEW_MODEL", "gemini-3.5-flash"))
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0).with_structured_output(ReportJobSummary)
    messages = [
        SystemMessage(
            content=(
                "You normalize job postings that were already extracted by a vision worker. "
                "Read field names in any language, including Korean. "
                "Return one summary item per input job in the same order. "
                "Do not invent missing facts; use an empty string when a value is unknown."
            )
        ),
        HumanMessage(content=json.dumps({"jobs": compact_jobs}, ensure_ascii=False, indent=2)),
    ]
    response = llm.invoke(messages)
    summary = _dump_model(response)
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
    if _report_summary_mode() == "off":
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
) -> dict[str, Any]:
    """작업자 그래프 실행 결과를 구조화된 제출물(WorkerSubmission)로 만든다."""
    extracted_jd = final_state.get("extracted_jd", {}) or {}
    jobs = _job_items(extracted_jd)
    current_url = final_state.get("current_url", "") or ""
    resolved_site = site or site_of(current_url) or "unknown"
    run_id = run_id or new_worker_run_id()
    report_jobs, report_source, report_error = _report_job_summary(jobs)
    recorded_steps = list(final_state.get("recorded_steps", []) or [])
    feedback_episodes = list(final_state.get("feedback_episodes", []) or [])
    transition_observations = list(final_state.get("transition_observations", []) or [])
    extracted_summary = {
        "has_data": bool(jobs),
        "job_count": len(jobs),
        "jobs": report_jobs,
        "summary_source": report_source,
        "summary_error": report_error,
        "current_url": current_url,
        "action_count": len(final_state.get("action_history", []) or []),
    }
    from agent.recipe.skill_metadata import build_skill_metadata_evidence

    skill_metadata_evidence = build_skill_metadata_evidence(
        goal=final_state.get("goal", "") or "",
        site=resolved_site,
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
        keyword=keyword,
        run_status=run_status,
        review_attempt=review_attempt,
        is_finished=bool(final_state.get("is_finished", False)),
        hit_recursion_limit=bool(hit_recursion_limit),
        collected_count=len(jobs),
        target_count=int(target_count or 0),
        persisted_count=int(persisted_count or 0),
        feedback_saved=int(feedback_saved or 0),
        recorded_steps=recorded_steps,
        feedback_episodes=feedback_episodes,
        transition_observations=transition_observations,
        skill_metadata_evidence=skill_metadata_evidence,
        extracted_summary=extracted_summary,
        worker_notes="submitted after autonomous/reflex worker run",
    )
    return _dump_model(submission)


def validate_submission_shape(submission: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate structure only; do not make semantic promotion decisions here."""
    issues: list[SubmissionIssue] = []

    def add(field: str, reason: str, severity: str = "error") -> None:
        issues.append(SubmissionIssue(field=field, reason=reason, severity=severity))

    if not submission.get("goal"):
        add("goal", "missing worker goal")
    if not submission.get("site"):
        add("site", "missing site identifier", "warning")
    if int(submission.get("collected_count") or 0) <= 0:
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
        if not step.get("state_key"):
            add(f"recorded_steps[{idx}].state_key", "missing OCR state key")
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

    return [_dump_model(issue) for issue in issues]


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
            recipe_candidate=False,
            confidence=0.78,
        )
        return _dump_model(review)

    warnings = [f"{issue['field']}: {issue['reason']}" for issue in issues]
    review = CommanderReview(
        decision="accept",
        reasons=warnings or ["submission shape is valid"],
        feedback_to_worker="",
        recipe_candidate=bool(submission.get("recorded_steps")) and int(submission.get("collected_count") or 0) > 0,
        confidence=0.62 if warnings else 0.72,
    )
    return _dump_model(review)


def _llm_review(submission: dict[str, Any], issues: list[dict[str, Any]], fallback: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = os.getenv("VISION_WORKER_REVIEW_MODEL", "gemini-3.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0).with_structured_output(CommanderReview)
    compact = {
        "submission": submission,
        "shape_issues": issues,
        "review_rules": [
            "Accept only if the submitted jobs and UI evidence plausibly satisfy the user goal.",
            "Revise when the worker clicked/collected the wrong target, omitted required data, or needs another run.",
            "Reject only when the site/run cannot satisfy the goal after available feedback.",
            "Do not activate a Reflex recipe; only mark whether it is a candidate for later replay testing.",
        ],
    }
    messages = [
        SystemMessage(
            content=(
                "You are the commander reviewing a child vision worker submission. "
                "Return only the structured CommanderReview schema."
            )
        ),
        HumanMessage(content=json.dumps(compact, ensure_ascii=False, indent=2)),
    ]
    last_error = ""
    for _ in range(2):
        try:
            response = llm.invoke(messages)
            review = _dump_model(response)
            return _dump_model(CommanderReview(**review))
        except Exception as exc:  # pragma: no cover - provider/schema failures are best-effort
            last_error = str(exc)
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous review was not valid CommanderReview JSON/schema output. "
                        "Retry with fields: decision, reasons, feedback_to_worker, recipe_candidate, confidence."
                    )
                )
            )
    fallback = dict(fallback)
    fallback.setdefault("reasons", []).append(f"llm_review_failed: {last_error[:200]}")
    return fallback


def review_worker_submission(submission: dict[str, Any]) -> dict[str, Any]:
    """Review a submission. Shape is always checked; semantic LLM review is opt-in."""
    issues = validate_submission_shape(submission)
    fallback = shape_review(submission, issues)
    if fallback.get("decision") != "accept":
        return fallback
    mode = os.getenv("VISION_WORKER_REVIEW_MODE", "shape").strip().lower()
    if mode != "llm":
        return fallback
    try:
        return _llm_review(submission, issues, fallback)
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

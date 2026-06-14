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

from agent.recipe.state_key import site_of
from shared.schema.feedback_schema import CommanderReview, SubmissionIssue, WorkerSubmission


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
    if any(key in extracted_jd for key in ("url", "URL", "company_name", "position")):
        return [extracted_jd]
    return [extracted_jd] if extracted_jd else []


def _pick(job: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = job.get(key)
        if value:
            return value
    return ""


def _first_text(job: dict[str, Any], exclude: set[str]) -> str:
    for key, value in job.items():
        if key in exclude:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _job_summary(jobs: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    out = []
    for job in jobs[:limit]:
        url = _pick(job, "url", "URL")
        company = _pick(job, "company_name", "company", "companyName") or _first_text(job, {"url", "URL"})
        position = _pick(job, "position", "title", "job_title", "jobTitle")
        out.append(
            {
                "company": company,
                "position": position,
                "url": url,
                "field_count": len(job.keys()),
            }
        )
    return out


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
) -> dict[str, Any]:
    """Build the structured handoff from a worker graph run."""
    extracted_jd = final_state.get("extracted_jd", {}) or {}
    jobs = _job_items(extracted_jd)
    current_url = final_state.get("current_url", "") or ""
    resolved_site = site or site_of(current_url) or "unknown"
    run_id = run_id or f"worker-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
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
        persisted_count=int(persisted_count or 0),
        feedback_saved=int(feedback_saved or 0),
        recorded_steps=list(final_state.get("recorded_steps", []) or []),
        feedback_episodes=list(final_state.get("feedback_episodes", []) or []),
        extracted_summary={
            "has_data": bool(jobs),
            "job_count": len(jobs),
            "jobs": _job_summary(jobs),
            "current_url": current_url,
            "action_count": len(final_state.get("action_history", []) or []),
        },
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
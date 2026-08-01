"""비전 작업자 제출물을 만들고 관찰 가능한 실행 사실을 검증한다."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from agent.recipe.text_utils import site_of
from agent.runtime.job_collection import job_items as _job_items
from agent.recipe.task_category import normalize_task_category
from agent.utils.job_fields import JOB_FIELD_ALIASES, deterministic_report_item, first_present, summary_text
from agent.utils.model_dump import dump_model
from shared.schema.feedback_schema import CommanderReview, SubmissionIssue, WorkerSubmission


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
    report_jobs = _empty_report_summary(jobs, 10)
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


def review_worker_submission(submission: dict[str, Any]) -> dict[str, Any]:
    """작업자 제출물의 구조와 관찰 가능한 실행 사실을 검증한다."""
    return shape_review(submission, validate_submission_shape(submission))


__all__ = [
    "build_worker_submission",
    "new_worker_run_id",
    "review_worker_submission",
    "shape_review",
    "validate_submission_shape",
]

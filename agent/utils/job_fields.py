from __future__ import annotations

from typing import Any


JOB_FIELD_ALIASES: dict[str, list[str]] = {
    "company_name": ["company_name", "company", "companyName", "\ud68c\uc0ac\uba85", "\uae30\uc5c5\uba85", "\ud68c\uc0ac"],
    "position": ["position", "job_title", "jobTitle", "title", "role", "\uc9c1\ubb34\uba85", "\uacf5\uace0\uba85", "\ud3ec\uc9c0\uc158"],
    "url": ["url", "URL", "job_url", "jobUrl", "posting_url", "\uacf5\uace0url", "\uc0c1\uc138url"],
    "main_tasks": ["main_tasks", "mainTasks", "\uc8fc\uc694\uc5c5\ubb34", "\uc8fc\uc694 \uc5c5\ubb34", "\ub2f4\ub2f9\uc5c5\ubb34"],
    "requirements": ["requirements", "qualification", "qualifications", "\uc790\uaca9\uc694\uac74", "\uc790\uaca9 \uc694\uac74", "\uc790\uaca9\uc870\uac74"],
    "preferred": ["preferred", "preferences", "preferred_qualifications", "\uc6b0\ub300\uc0ac\ud56d", "\uc6b0\ub300 \uc0ac\ud56d"],
    "benefits": ["benefits", "welfare", "\ud61c\ud0dd", "\ubcf5\uc9c0", "\ud61c\ud0dd\ubc0f\ubcf5\uc9c0"],
    "tech_stack": ["tech_stack", "techStack", "skills", "\uae30\uc220\uc2a4\ud0dd", "\uae30\uc220 \uc2a4\ud0dd"],
    "location": ["location", "\uadfc\ubb34\uc9c0", "\uc9c0\uc5ed"],
    "employment_type": ["employment_type", "employmentType", "\uace0\uc6a9\ud615\ud0dc"],
    "deadline": ["deadline", "\ub9c8\uac10\uc77c", "\uc811\uc218\ub9c8\uac10"],
    "salary": ["salary", "\uc5f0\ubd09", "\uae09\uc5ec"],
}


def first_present(job: dict[str, Any], aliases: list[str]) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in job.items()}
    for alias in aliases:
        if alias in job and job.get(alias) not in (None, "", [], {}):
            return job.get(alias)
        value = lowered.get(alias.lower())
        if value not in (None, "", [], {}):
            return value
    return None


def summary_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return ""
    return str(value).strip()


def deterministic_report_item(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "company": summary_text(first_present(job, JOB_FIELD_ALIASES["company_name"])),
        "position": summary_text(first_present(job, JOB_FIELD_ALIASES["position"])),
        "url": summary_text(first_present(job, JOB_FIELD_ALIASES["url"])),
        "field_count": len(job.keys()),
    }


def deterministic_job_for_persistence(job: dict[str, Any]) -> dict[str, Any]:
    """이미 추출된 원본 JSON에서 표준 저장 필드만 얕게 보정합니다."""
    normalized = dict(job)
    for field, aliases in JOB_FIELD_ALIASES.items():
        if normalized.get(field) in (None, "", [], {}):
            value = first_present(job, aliases)
            if value not in (None, "", [], {}):
                normalized[field] = value
    normalized["_normalization_source"] = "deterministic"
    return normalized

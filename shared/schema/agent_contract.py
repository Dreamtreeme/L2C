"""LLM과 백엔드가 공유하는 실행 필드 계약."""

from __future__ import annotations

from shared.schema.jd_schema import JobField

JOB_COLLECTION_FIELD_LABELS: dict[str, str] = {
    JobField.COMPANY_NAME.value: "회사명",
    JobField.POSITION.value: "직무명",
    JobField.URL.value: "공고 URL",
    JobField.JOB_CATEGORY.value: "직군",
    JobField.EXPERIENCE_TEXT.value: "경력 조건 원문",
    JobField.EDUCATION.value: "학력",
    JobField.EMPLOYMENT_TYPE.value: "고용 형태",
    JobField.LOCATION.value: "근무지",
    JobField.POSTED_AT.value: "게시일",
    JobField.POSTED_AT_TEXT.value: "게시일 원문",
    JobField.DEADLINE.value: "마감일",
    JobField.TECH_STACK.value: "기술 스택",
    JobField.MAIN_TASKS.value: "주요 업무",
    JobField.REQUIREMENTS.value: "자격 요건",
    JobField.PREFERRED.value: "우대 사항",
    JobField.BENEFITS.value: "혜택 및 복지",
    JobField.SALARY.value: "급여",
    JobField.RAW_OCR_TEXT.value: "OCR 원문",
}

DEFAULT_JOB_COLLECTION_FIELDS: tuple[JobField, ...] = (
    JobField.COMPANY_NAME,
    JobField.POSITION,
    JobField.URL,
    JobField.MAIN_TASKS,
    JobField.REQUIREMENTS,
)

ANSWER_EVIDENCE_FIELDS: tuple[str, ...] = tuple(
    field.value
    for field in (
        JobField.COMPANY_NAME,
        JobField.POSITION,
        JobField.JOB_CATEGORY,
        JobField.EXPERIENCE_TEXT,
        JobField.EMPLOYMENT_TYPE,
        JobField.LOCATION,
        JobField.POSTED_AT,
        JobField.POSTED_AT_TEXT,
        JobField.TECH_STACK,
        JobField.MAIN_TASKS,
        JobField.REQUIREMENTS,
        JobField.PREFERRED,
        JobField.BENEFITS,
        JobField.RAW_OCR_TEXT,
    )
)

__all__ = [
    "ANSWER_EVIDENCE_FIELDS",
    "DEFAULT_JOB_COLLECTION_FIELDS",
    "JOB_COLLECTION_FIELD_LABELS",
]

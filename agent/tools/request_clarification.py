"""지휘자가 사용자에게 추가 입력을 요청하는 구조화 도구."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from shared.schema.investigation_schema import (
    ClarificationOption,
    ClarificationQuestion,
)


@tool
def request_clarification(
    question: str,
    missing_fields: list[str] | None = None,
    reason: str = "",
    question_id: str = "",
    field: str = "",
    options: list[dict] | None = None,
    allow_custom: bool = True,
) -> str:
    """수집 방향을 결정할 정보가 부족할 때 구조화된 확인 질문을 반환한다."""

    normalized_question = str(question or "").strip()
    if not normalized_question:
        normalized_question = "어떤 채용공고를 찾아야 하는지 조금 더 구체적으로 알려주세요."
    normalized_fields = [
        str(item).strip()
        for item in (missing_fields or [])
        if str(item).strip()
    ]
    normalized_field = str(field or (normalized_fields[0] if normalized_fields else "request_detail")).strip()
    normalized_question_id = str(question_id or f"clarify_{normalized_field}").strip()
    clarification = ClarificationQuestion(
        question_id=normalized_question_id,
        field=normalized_field,
        question=normalized_question,
        options=[ClarificationOption.model_validate(item) for item in (options or [])],
        allow_custom=bool(allow_custom),
        reason=str(reason or "").strip(),
    )
    return json.dumps(
        {
            "needs_clarification": True,
            **clarification.model_dump(mode="json"),
            "missing_fields": normalized_fields,
        },
        ensure_ascii=False,
    )


__all__ = ["request_clarification"]

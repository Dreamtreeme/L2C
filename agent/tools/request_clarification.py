"""지휘자가 사용자에게 추가 입력을 요청하는 구조화 도구."""

from __future__ import annotations

import json

from langchain_core.tools import tool


@tool
def request_clarification(
    question: str,
    missing_fields: list[str] | None = None,
    reason: str = "",
) -> str:
    """수집 방향을 결정할 정보가 부족할 때 한 번의 구체적인 확인 질문을 반환한다."""

    normalized_question = str(question or "").strip()
    if not normalized_question:
        normalized_question = "어떤 채용공고를 찾아야 하는지 조금 더 구체적으로 알려주세요."
    return json.dumps(
        {
            "needs_clarification": True,
            "question": normalized_question,
            "missing_fields": [
                str(field).strip()
                for field in (missing_fields or [])
                if str(field).strip()
            ],
            "reason": str(reason or "").strip(),
        },
        ensure_ascii=False,
    )


__all__ = ["request_clarification"]

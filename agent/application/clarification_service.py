"""사용자의 확인 답변을 확정된 조사 조건으로 반영한다."""

from __future__ import annotations

import calendar
from datetime import date, timedelta

from shared.schema.investigation_schema import (
    ClarificationAnswer,
    ClarificationQuestion,
    InvestigationPurpose,
    InvestigationRequest,
    InvestigationStatus,
)


def _subtract_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 - months
    year, month_zero_based = divmod(month_index, 12)
    month = month_zero_based + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _resolve_period(value: str, today: date) -> tuple[date, date]:
    normalized = str(value or "").strip().upper()
    if normalized == "P30D":
        return today - timedelta(days=30), today
    if normalized == "P3M":
        return _subtract_months(today, 3), today
    if normalized == "P6M":
        return _subtract_months(today, 6), today
    if "/" in normalized:
        start_text, end_text = normalized.split("/", 1)
        return date.fromisoformat(start_text), date.fromisoformat(end_text)
    raise ValueError(f"지원하지 않는 기간 값입니다: {value}")


def _selected_value(
    question: ClarificationQuestion,
    answer: ClarificationAnswer,
) -> str:
    if answer.custom_value.strip():
        if not question.allow_custom:
            raise ValueError("이 질문은 직접 입력을 허용하지 않습니다.")
        return answer.custom_value.strip()
    if answer.value.strip():
        return answer.value.strip()
    for option in question.options:
        if option.option_id == answer.selected_option_id:
            return option.value
    raise ValueError("질문의 선택지와 일치하는 답변을 찾을 수 없습니다.")


def apply_clarification_answer(
    investigation: InvestigationRequest,
    answer: ClarificationAnswer,
    *,
    today: date | None = None,
) -> InvestigationRequest:
    """질문 하나의 답변을 조사 상태에 누적한다."""

    question = next(
        (
            item
            for item in investigation.clarification_questions
            if item.question_id == answer.question_id
        ),
        None,
    )
    if question is None:
        raise ValueError("현재 조사에 없는 확인 질문입니다.")

    selected_value = _selected_value(question, answer)
    resolved_answer = answer.model_copy(update={"value": selected_value})
    answers = [
        item
        for item in investigation.clarification_answers
        if item.question_id != answer.question_id
    ]
    answers.append(resolved_answer)

    constraints = investigation.constraints.model_copy(deep=True)
    if question.field == "recent_period":
        period_start, period_end = _resolve_period(selected_value, today or date.today())
        constraints.posted_from = period_start.isoformat()
        constraints.posted_to = period_end.isoformat()
        if investigation.purpose == InvestigationPurpose.TREND:
            comparison_end = period_start - timedelta(days=1)
            if selected_value.upper() == "P3M":
                comparison_start = _subtract_months(period_start, 3)
            elif selected_value.upper() == "P6M":
                comparison_start = _subtract_months(period_start, 6)
            else:
                comparison_start = comparison_end - (period_end - period_start)
            constraints.comparison_posted_from = comparison_start.isoformat()
            constraints.comparison_posted_to = comparison_end.isoformat()
    elif question.field == "site_scope":
        constraints.sites = [] if selected_value == "all_enabled" else [selected_value]
    elif question.field == "target_count":
        if selected_value == "visible_all":
            constraints.count_mode = "visible_all"
            constraints.target_count = 0
        else:
            constraints.count_mode = "explicit"
            constraints.target_count = int(selected_value)
    elif hasattr(constraints, question.field):
        setattr(constraints, question.field, selected_value)

    unresolved_fields = [
        item for item in investigation.unresolved_fields if item != question.field
    ]
    status = (
        InvestigationStatus.AWAITING_CLARIFICATION
        if unresolved_fields
        else InvestigationStatus.CHECKING_EVIDENCE
    )
    return investigation.model_copy(
        update={
            "constraints": constraints,
            "clarification_answers": answers,
            "unresolved_fields": unresolved_fields,
            "status": status,
        }
    )


__all__ = ["apply_clarification_answer"]

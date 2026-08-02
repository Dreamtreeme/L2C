"""사용자의 확인 답변을 확정된 조사 조건으로 반영한다."""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from shared.schema.investigation_schema import (
    ClarificationAnswer,
    ClarificationQuestion,
    InvestigationConstraints,
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
    days_match = re.fullmatch(r"P(\d+)D", normalized)
    if days_match:
        return today - timedelta(days=int(days_match.group(1))), today
    months_match = re.fullmatch(r"P(\d+)M", normalized)
    if months_match:
        return _subtract_months(today, int(months_match.group(1))), today
    if "/" in normalized:
        start_text, end_text = normalized.split("/", 1)
        return date.fromisoformat(start_text), date.fromisoformat(end_text)
    raise ValueError(f"지원하지 않는 기간 값입니다: {value}")


def _is_period_value(value: str) -> bool:
    normalized = str(value or "").strip().upper()
    return bool(
        re.fullmatch(r"P\d+[DM]", normalized)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}", normalized)
    )


def _resolve_comparison_period(value: str) -> tuple[date, date, date, date]:
    normalized = str(value or "").strip()
    match = re.fullmatch(
        r"current=(\d{4}-\d{2}-\d{2})/(\d{4}-\d{2}-\d{2});"
        r"comparison=(\d{4}-\d{2}-\d{2})/(\d{4}-\d{2}-\d{2})",
        normalized,
    )
    if match is None:
        raise ValueError("비교 기간 선택값에 현재 기간과 비교 기간이 필요합니다.")
    return tuple(date.fromisoformat(item) for item in match.groups())


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


def _selected_option(
    question: ClarificationQuestion,
    answer: ClarificationAnswer,
):
    return next(
        (
            option
            for option in question.options
            if option.option_id == answer.selected_option_id
            or (answer.value.strip() and option.value == answer.value.strip())
        ),
        None,
    )


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
    if question.field == "recent_period" or (
        question.field in {"posted_from", "posted_to"}
        and _is_period_value(selected_value)
    ):
        period_start, period_end = _resolve_period(selected_value, today or date.today())
        constraints.posted_from = period_start.isoformat()
        constraints.posted_to = period_end.isoformat()
        if investigation.purpose == InvestigationPurpose.TREND:
            comparison_end = period_start - timedelta(days=1)
            months_match = re.fullmatch(r"P(\d+)M", selected_value.upper())
            if months_match:
                comparison_start = _subtract_months(
                    period_start,
                    int(months_match.group(1)),
                )
            else:
                comparison_start = comparison_end - (period_end - period_start)
            constraints.comparison_posted_from = comparison_start.isoformat()
            constraints.comparison_posted_to = comparison_end.isoformat()
    elif question.field == "comparison_period":
        current_start, current_end, comparison_start, comparison_end = (
            _resolve_comparison_period(selected_value)
        )
        constraints.posted_from = current_start.isoformat()
        constraints.posted_to = current_end.isoformat()
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
    elif question.field == "occupation_domain_concept_keys":
        option = _selected_option(question, answer)
        if answer.custom_value.strip():
            constraints.occupation_domain_query = ""
            constraints.occupation_domain_concept_keys = []
            constraints.occupation_query = selected_value
            constraints.collection_search_term = selected_value
            constraints.occupation_concept_keys = []
            constraints.occupation_scope_required = False
            constraints.occupation_scope_mode = "unspecified"
            constraints.occupation_resolution = "unresolved"
        else:
            constraints.occupation_domain_concept_keys = [selected_value]
            if option is not None and option.collection_search_term:
                constraints.occupation_domain_query = option.collection_search_term
    elif question.field == "occupation_concept_keys":
        option = _selected_option(question, answer)
        if answer.custom_value.strip():
            constraints.occupation_query = selected_value
            constraints.collection_search_term = selected_value
            constraints.occupation_concept_keys = []
            constraints.occupation_scope_mode = "unspecified"
            constraints.occupation_resolution = "unresolved"
        else:
            constraints.occupation_concept_keys = [selected_value]
            selected_all = answer.selected_option_id == "all-descendants"
            constraints.occupation_scope_mode = "all" if selected_all else "selected"
            constraints.occupation_resolution = (
                "reviewed_alias"
                if question.facet_type == "semantic_occupation"
                else "user_selected"
            )
            if (
                question.facet_type != "semantic_occupation"
                and not selected_all
                and option is not None
                and option.collection_search_term
            ):
                constraints.collection_search_term = option.collection_search_term
                constraints.occupation_query = option.collection_search_term
    elif question.field == "occupation_query":
        constraints.occupation_query = selected_value
        constraints.collection_search_term = selected_value
        constraints.occupation_concept_keys = []
        constraints.occupation_scope_required = False
        constraints.occupation_scope_mode = "unspecified"
        constraints.occupation_resolution = "unresolved"
    elif question.field == "skill_queries":
        constraints.skill_queries = [selected_value]
        constraints.skill_concept_keys = []
    elif question.field == "analysis_dimensions":
        constraints.analysis_dimensions = [selected_value]
    elif question.field == "sites":
        constraints.sites = [selected_value]
    elif question.field == "posted_from":
        constraints.posted_from = selected_value
    elif question.field == "posted_to":
        constraints.posted_to = selected_value
    elif question.field == "location":
        constraints.location = selected_value
    elif question.field == "experience":
        constraints.experience = selected_value
    elif question.field == "employment_type":
        constraints.employment_type = selected_value

    constraints = InvestigationConstraints.model_validate(
        constraints.model_dump(mode="python")
    )

    remaining_questions = [
        item
        for item in investigation.clarification_questions
        if item.question_id != question.question_id
    ]
    status = (
        InvestigationStatus.AWAITING_CLARIFICATION
        if remaining_questions
        else InvestigationStatus.CHECKING_EVIDENCE
    )
    return investigation.model_copy(
        update={
            "constraints": constraints,
            "clarification_answers": answers,
            "clarification_questions": remaining_questions,
            "status": status,
        }
    )


__all__ = ["apply_clarification_answer"]

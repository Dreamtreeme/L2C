"""비전 작업자에 전달할 수집 의도와 사이트 목표를 구성한다."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote_plus

from agent.config import get_settings
from agent.runtime.job_field_contract import (
    build_job_collection_contract,
    field_contract_items,
)
from agent.sites.profile import SiteProfile
from agent.utils.model_dump import dump_model
from shared.schema.collection_intent import (
    CollectionCountMode,
    CollectionIntent,
    normalize_collection_intent,
)

logger = logging.getLogger(__name__)


def normalize_target_count(value: Any) -> int:
    """사용자가 요청한 수집 개수를 안전한 정수 범위로 정규화한다."""

    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, count))


def _default_site_slug() -> str:
    from agent.sites import list_supported_sites

    sites = list_supported_sites()
    if not sites:
        raise ValueError("활성화된 사이트 프로필이 없습니다.")
    return sites[0].slug


def load_collection_profile(site: str | None) -> SiteProfile:
    from agent.sites import load_site_profile

    return load_site_profile(site or _default_site_slug())


def _profile_site_terms(profile: SiteProfile) -> list[str]:
    terms = [
        profile.slug,
        profile.display_name,
    ]
    terms.extend(profile.aliases)
    terms.extend(profile.domains)
    base_url = profile.base_url
    if base_url:
        terms.append(
            str(base_url)
            .replace("https://", "")
            .replace("http://", "")
            .strip("/")
        )
    cleaned = {str(term).strip() for term in terms if str(term or "").strip()}
    return sorted(cleaned, key=len, reverse=True)


def _search_intent_mode() -> str:
    mode = get_settings().vision.search_intent_mode.strip().lower()
    return mode if mode in {"llm", "off"} else "llm"


def extract_search_intent(
    raw_query: str,
    profile: SiteProfile,
) -> dict[str, Any]:
    """사용자 요청에서 작업자에게 전달할 전체 수집 조건을 추출한다."""

    original = str(raw_query or "").strip()
    if not original:
        return dump_model(CollectionIntent())

    mode = _search_intent_mode()
    if mode == "off":
        intent = dump_model(
            CollectionIntent(
                original_query=original,
                search_keyword=original,
            )
        )
        intent["source"] = "disabled"
        intent["error"] = ""
        return intent

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from agent.application.model_clients import get_structured_google_model
        from agent.application.model_policy import lightweight_model_name
        from agent.application.run_context import invoke_with_metrics

        model_name = lightweight_model_name("VISION_SEARCH_INTENT_MODEL")
        llm = get_structured_google_model(
            model_name,
            CollectionIntent,
            temperature=0.0,
            execution_role="lightweight",
        )
        messages = [
            SystemMessage(
                content=(
                    "Extract a job-search intent for an autonomous vision worker. "
                    "Return only the actual phrase that should be searched "
                    "on the target job site. "
                    "Remove site names, URLs, filler commands, and analysis/reporting words. "
                    "Preserve date, experience, location, employment type, "
                    "freshness, and analysis purpose. "
                    "Use count_mode=explicit only for an explicit number and set target_count. "
                    "Use count_mode=visible_all for all/every posting or "
                    "when no count is specified. "
                    "Use purpose=compare or trend only when requested. "
                    "Leave required_fields empty; the evidence plan and site profile "
                    "supply that contract. "
                    "Do not translate or broaden the keyword."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "user_request": original,
                        "site": profile.slug,
                        "site_terms": _profile_site_terms(profile),
                        "default_query_target": profile.default_query_target,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        ]
        data = dump_model(
            invoke_with_metrics(
                llm,
                messages,
                "search_intent",
                stream=True,
            )
        )
        intent = dump_model(
            normalize_collection_intent(
                data,
                original_query=original,
                site=profile.slug,
                search_keyword=original,
            )
        )
        intent["source"] = "llm"
        intent["error"] = ""
        return intent
    except Exception as exc:  # pragma: no cover - 공급자 장애 시 원문을 사용한다.
        logger.warning(
            "검색 의도 추출에 실패해 원문을 사용합니다: %s",
            exc,
        )
        intent = dump_model(
            CollectionIntent(
                original_query=original,
                search_keyword=original,
            )
        )
        intent["source"] = "llm_failed"
        intent["error"] = str(exc)[:200]
        return intent


def build_direct_search_url(
    search_keyword: str,
    profile: SiteProfile,
) -> str:
    navigation = profile.navigation_policy
    if not navigation.allow_direct_search_url:
        return ""
    template = navigation.search_url_template.strip()
    if not search_keyword or not template:
        return ""
    encoded = quote_plus(search_keyword)
    return template.format(
        query=encoded,
        keyword=encoded,
        raw_query=search_keyword,
    )


def _compact_research_for_goal(research: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(research, dict):
        return {}
    compact: dict[str, Any] = {
        "status": research.get("status", ""),
        "query": research.get("query", ""),
        "meaning": research.get("meaning", ""),
        "requirements": research.get("requirements", []),
        "sensitive_steps": research.get("sensitive_steps", []),
    }
    paths = research.get("official_paths") or research.get("possible_sites") or []
    if isinstance(paths, list):
        compact["routes"] = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "domain": item.get("domain", ""),
            }
            for item in paths[:5]
            if isinstance(item, dict)
        ]
    return compact


def _task_context_section(task_context: dict[str, Any] | None) -> str:
    if not isinstance(task_context, dict) or not task_context:
        return ""
    triage = (
        task_context.get("triage")
        if isinstance(task_context.get("triage"), dict)
        else {}
    )
    research = (
        task_context.get("research_report")
        if isinstance(task_context.get("research_report"), dict)
        else {}
    )
    allowed = task_context.get("allowed_actions") or []
    blocked = task_context.get("blocked_actions") or []
    return (
        "[Task triage and safety context]\n"
        f"triage={json.dumps(triage, ensure_ascii=False)}\n"
        "research_summary="
        f"{json.dumps(_compact_research_for_goal(research), ensure_ascii=False)}\n"
        f"allowed_actions={json.dumps(allowed, ensure_ascii=False)}\n"
        f"blocked_actions={json.dumps(blocked, ensure_ascii=False)}\n"
        "Proceed automatically only for read, navigation, search, "
        "and public information collection. "
        "Before login, personal data, authentication, agreement, "
        "application, payment, account, "
        "finance, or legal-effect steps, stop and request human confirmation.\n\n"
    )


def build_site_goal(
    search_keyword: str,
    profile: SiteProfile,
    direct_search_url: str = "",
    target_count: int = 0,
    task_context: dict[str, Any] | None = None,
    collection_intent: dict[str, Any] | None = None,
) -> str:
    """확정된 요청과 사이트 프로필로 작업자 목표문을 만든다."""

    site_name = profile.display_name or profile.slug
    base_url = profile.base_url
    site_skill = profile.guidance.strip()
    navigation = profile.navigation_policy
    start_url = str(navigation.start_url or base_url or "").strip()
    search_entry = navigation.search_entry.strip()
    navigation_section = ""
    direct_search_section = ""
    if direct_search_url:
        direct_search_section = (
            "[Code-generated search URL]\n"
            f"{direct_search_url}\n"
            "Use this exact URL with open_browser when direct search navigation is useful. "
            "Do not hand-encode Korean query text.\n\n"
        )
    elif start_url:
        navigation_section = (
            "[Navigation start]\n"
            f"Open only the site home page with open_browser: {start_url}\n"
            "Do not construct or open a search/query URL yourself. "
            "Find the visible search input, type the user query, and submit from the page."
            f"{' ' + search_entry if search_entry else ''}\n\n"
        )
    intent = normalize_collection_intent(
        collection_intent,
        site=profile.slug,
        search_keyword=search_keyword,
        target_count=target_count,
    )
    job_collection_contract = build_job_collection_contract(
        dump_model(intent),
        profile_fields=profile.collection_policy.required_fields,
    )
    required_field_items = field_contract_items(
        job_collection_contract["required_fields"]
    )
    list_fields = {
        "tech_stack",
        "main_tasks",
        "requirements",
        "preferred",
        "benefits",
    }
    required_record_shape = {
        item["field"]: (
            []
            if item["field"] in list_fields
            else ""
        )
        for item in required_field_items
    }
    target_section = ""
    if int(target_count or 0) > 0:
        target_section = (
            "[Collection target]\n"
            f"Collect up to {int(target_count)} distinct job postings for this request. "
            "When that many valid detail pages have been collected and submitted, "
            "finish instead of opening more cards.\n\n"
        )
    elif intent.count_mode == CollectionCountMode.VISIBLE_ALL:
        target_section = (
            "[Collection target]\n"
            "Collect every relevant job card visible on the first stable search-result screen. "
            "Do not invent a fixed item count and do not continue to additional result pages "
            "unless the user explicitly requested it.\n\n"
        )
    task_context_section = _task_context_section(task_context)
    filter_values = {
        "posted_date_expression": intent.filters.posted_date_expression,
        "posted_from": intent.filters.posted_from,
        "posted_to": intent.filters.posted_to,
        "experience": intent.filters.experience,
        "location": intent.filters.location,
        "employment_type": intent.filters.employment_type,
    }
    confirmed_filters = {
        key: value
        for key, value in filter_values.items()
        if str(value or "").strip()
    }
    confirmed_request_section = (
        "[Confirmed collection constraints]\n"
        f"filters={json.dumps(confirmed_filters, ensure_ascii=False)}\n"
        f"freshness_required={str(bool(intent.freshness_required)).lower()}\n"
        f"purpose={intent.purpose.value}\n"
        f"analysis_goal={intent.analysis_goal}\n"
        "Apply a visible site filter when the current UI supports it. "
        "Otherwise verify the condition from visible job evidence. "
        "Do not infer a date, location, experience, or employment condition "
        "that is not visible.\n\n"
    )

    return (
        f"{site_name}({base_url})에서 '{search_keyword}' 채용공고를 검색하고 수집하세요. "
        "검색 결과 화면에 도달하면 공고 수와 카드/행 목록을 확인하고, "
        "방문할 공고별 순회 계획을 세운 뒤 상세 페이지를 하나씩 방문하세요. "
        "상세 페이지에서 필요한 정보를 수집한 뒤 목록으로 돌아와 "
        "이미 클릭한 공고는 제외하고 다음 공고를 선택하세요.\n\n"
        f"{navigation_section}"
        f"{direct_search_section}"
        f"{target_section}"
        f"{confirmed_request_section}"
        f"{task_context_section}"
        "[필수 공고 필드 계약]\n"
        f"required_fields={json.dumps(required_field_items, ensure_ascii=False)}\n"
        f"required_record_shape={json.dumps(required_record_shape, ensure_ascii=False)}\n"
        "상세 화면의 scroll 또는 본문 펼치기 click_marker를 선택할 때마다 현재 화면에서 "
        "확인한 필드를 observed_fields에 함께 기록하십시오. 모든 필수 필드에 화면 근거가 "
        "모였을 때만 finish_detail_reading을 호출하십시오. 페이지 끝까지 확인했는데 공고가 "
        "제공하지 않는 필드는 page_exhausted=true와 unavailable_fields로 명시하십시오. "
        "값을 추측해서 빈 필드를 채우지 마십시오.\n\n"
        f"[선택된 사이트 스킬]\n{site_skill}"
    )


__all__ = [
    "build_direct_search_url",
    "build_site_goal",
    "extract_search_intent",
    "load_collection_profile",
    "normalize_target_count",
]

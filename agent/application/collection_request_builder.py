"""비전 작업자에 전달할 수집 의도와 사이트 목표를 구성한다."""

from __future__ import annotations

import json
from typing import Any

from agent.runtime.job_field_contract import (
    build_job_collection_contract,
    field_contract_items,
)
from agent.sites.profile import SiteProfile
from agent.utils.model_dump import dump_model
from shared.schema.collection_intent import (
    CollectionCountMode,
    CollectionIntent,
)


def build_site_goal(
    intent: CollectionIntent,
    profile: SiteProfile,
    *,
    job_collection_contract: dict[str, Any] | None = None,
) -> str:
    """확정된 요청과 사이트 프로필로 작업자 목표문을 만든다."""

    site_name = profile.display_name or profile.slug
    start_url = profile.base_url.strip()
    if start_url:
        navigation_section = (
            "[Navigation start]\n"
            f"Open only the site home page with open_browser: {start_url}\n"
            "Do not construct or open a search/query URL yourself. "
            "Find the visible search input, type the user query, and submit from the page.\n\n"
        )
    else:
        navigation_section = ""
    resolved_contract = (
        job_collection_contract
        or build_job_collection_contract(
            dump_model(intent),
            profile_fields=profile.collection_policy.required_fields,
        )
    )
    required_field_items = field_contract_items(
        resolved_contract["required_fields"]
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
    if intent.target_count > 0:
        target_section = (
            "[Collection target]\n"
            f"Collect up to {intent.target_count} distinct job postings for this request. "
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
    else:
        target_section = ""
    confirmed_request = dump_model(intent)
    confirmed_request["filters"] = {
        key: value
        for key, value in confirmed_request["filters"].items()
        if str(value or "").strip()
    }
    constraints_section = (
        "[Confirmed collection constraints]\n"
        f"{json.dumps(confirmed_request, ensure_ascii=False)}\n"
        "Apply a visible site filter when the current UI supports it. "
        "Otherwise verify the condition from visible job evidence. "
        "Do not infer a date, location, experience, or employment condition "
        "that is not visible.\n\n"
    )

    return (
        f"{site_name}({profile.base_url})에서 '{intent.search_keyword}' 채용공고를 검색하고 수집하세요. "
        "검색 결과 화면에 도달하면 공고 수와 카드/행 목록을 확인하고, "
        "방문할 공고별 순회 계획을 세운 뒤 상세 페이지를 하나씩 방문하세요. "
        "상세 페이지에서 필요한 정보를 수집한 뒤 목록으로 돌아와 "
        "이미 클릭한 공고는 제외하고 다음 공고를 선택하세요.\n\n"
        f"{navigation_section}"
        f"{target_section}"
        f"{constraints_section}"
        "[필수 공고 필드 계약]\n"
        f"required_fields={json.dumps(required_field_items, ensure_ascii=False)}\n"
        f"required_record_shape={json.dumps(required_record_shape, ensure_ascii=False)}\n"
        "상세 화면의 scroll 또는 본문 펼치기 click_marker를 선택할 때마다 현재 화면에서 "
        "확인한 필드를 observed_fields에 함께 기록하십시오. 모든 필수 필드에 화면 근거가 "
        "모였을 때만 finish_detail_reading을 호출하십시오. 페이지 끝까지 확인했는데 공고가 "
        "제공하지 않는 필드는 page_exhausted=true와 unavailable_fields로 명시하십시오. "
        "값을 추측해서 빈 필드를 채우지 마십시오.\n\n"
        f"[선택된 사이트 스킬]\n{profile.guidance.strip()}"
    )


__all__ = [
    "build_site_goal",
]

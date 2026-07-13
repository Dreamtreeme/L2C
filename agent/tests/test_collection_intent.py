from agent.utils.model_dump import dump_model
from shared.schema.collection_intent import (
    CollectionCountMode,
    CollectionPurpose,
    normalize_collection_intent,
)


def test_explicit_count_takes_precedence_over_count_mode():
    intent = normalize_collection_intent(
        {"count_mode": "visible_all", "target_count": 3},
        search_keyword="iOS 개발자",
    )

    assert intent.count_mode == CollectionCountMode.EXPLICIT
    assert intent.target_count == 3


def test_visible_all_does_not_create_fixed_count():
    intent = normalize_collection_intent(
        {"count_mode": "visible_all", "target_count": 0},
        search_keyword="백엔드 개발자",
    )

    assert intent.count_mode == CollectionCountMode.VISIBLE_ALL
    assert intent.target_count == 0


def test_collection_intent_preserves_date_filters_and_analysis_goal():
    intent = normalize_collection_intent(
        {
            "original_query": "지난달 서울 AI 공고를 비교해줘",
            "site": "wanted",
            "search_keyword": "AI 개발자",
            "count_mode": "visible_all",
            "filters": {
                "posted_date_expression": "지난달",
                "location": "서울",
            },
            "freshness_required": True,
            "purpose": "compare",
            "analysis_goal": "회사별 요구 기술 비교",
        }
    )
    payload = dump_model(intent)

    assert payload["filters"]["posted_date_expression"] == "지난달"
    assert payload["filters"]["location"] == "서울"
    assert intent.freshness_required is True
    assert intent.purpose == CollectionPurpose.COMPARE
    assert intent.analysis_goal == "회사별 요구 기술 비교"


def test_realtime_scraping_tool_exposes_structured_request_fields():
    from agent.tools.realtime_scraping import realtime_scraping

    properties = realtime_scraping.args_schema.model_json_schema()["properties"]

    assert {
        "original_query",
        "count_mode",
        "posted_date_expression",
        "posted_from",
        "posted_to",
        "experience",
        "location",
        "employment_type",
        "freshness_required",
        "purpose",
        "analysis_goal",
    } <= set(properties)

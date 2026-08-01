import pytest

from shared.schema.collection_intent import (
    CollectionCountMode,
    CollectionIntent,
    CollectionPurpose,
)


@pytest.mark.parametrize(
    ("count_mode", "target_count", "expected_mode"),
    [
        ("visible_all", 3, CollectionCountMode.EXPLICIT),
        ("visible_all", 0, CollectionCountMode.VISIBLE_ALL),
        ("explicit", 0, CollectionCountMode.UNSPECIFIED),
    ],
)
def test_collection_intent_aligns_count_mode(
    count_mode,
    target_count,
    expected_mode,
):
    intent = CollectionIntent(
        search_keyword="iOS 개발자",
        count_mode=count_mode,
        target_count=target_count,
    )

    assert intent.count_mode == expected_mode


def test_collection_intent_preserves_confirmed_filters_and_goal():
    intent = CollectionIntent(
        original_query="서울 AI 공고를 비교해줘",
        site="wanted",
        search_keyword="AI 개발자",
        filters={"posted_from": "2026-07-01", "location": "서울"},
        freshness_required=True,
        purpose="compare",
        analysis_goal="회사별 요구 기술 비교",
    )

    assert intent.filters.location == "서울"
    assert intent.filters.posted_from == "2026-07-01"
    assert intent.purpose == CollectionPurpose.COMPARE
    assert intent.analysis_goal == "회사별 요구 기술 비교"


def test_collection_intent_rejects_unknown_fields():
    with pytest.raises(ValueError):
        CollectionIntent(search_keyword="개발자", company="ABC")

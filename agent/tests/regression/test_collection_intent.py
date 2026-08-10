import pytest

from shared.schema.collection_intent import (
    CollectionCountMode,
    CollectionIntent,
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

from agent.vision.target_snapshot import (
    build_action_target_snapshot,
    build_marker_target_snapshot,
)


def test_marker_target_snapshot_uses_shared_pixel_and_ratio_coordinates():
    snapshot = build_marker_target_snapshot(
        [
            {
                "id": 7,
                "text": "검색",
                "type": "icon",
                "bbox": [100, 50, 300, 150],
            }
        ],
        "7",
        screen_signature={"size": [1000, 500]},
        target_label="검색 입력",
    )

    assert snapshot == {
        "marker_id": 7,
        "text": "검색",
        "marker_type": "icon",
        "bbox": [100, 50, 300, 150],
        "center": [200, 100],
        "bbox_ratio": [0.1, 0.1, 0.3, 0.3],
        "center_ratio": [0.2, 0.2],
        "target_label": "검색 입력",
    }


def test_action_target_snapshot_ignores_non_target_and_reports_missing_marker():
    state = {
        "current_markers": [],
        "screen_signature": {"size": [1000, 500]},
    }

    assert build_action_target_snapshot(
        state,
        "press_key",
        {"key": "ENTER"},
    ) is None
    assert build_action_target_snapshot(
        state,
        "click_marker",
        {"marker_id": 9},
    ) == {"marker_id": 9, "missing": True}

from agent.graph.worker_selection_policy import (
    SelectionPolicy,
    decide_selection_entry,
)
from agent.graph.worker_transition_policy import (
    decide_after_ocr,
    decide_before_ocr,
)


def test_selection_entry_prioritizes_pending_action_over_screen_policy():
    decision = decide_selection_entry(
        has_pending_action=True,
        low_information_screen=True,
        low_information_capture_count=10,
        low_information_max_cycles=3,
        has_active_reflex_recipe=True,
    )

    assert decision.policy == SelectionPolicy.KEEP_PENDING_ACTION


def test_reflex_no_change_is_blocked_before_ocr():
    decision = decide_before_ocr(
        source="reflex",
        visual_changed=False,
    )

    assert decision.reason == "reflex_no_screen_change"
    assert decision.block_reflex_recipe is True


def test_general_transition_requires_markers_to_verify_change():
    unverified = decide_after_ocr(
        source="autonomous",
        markers_present=False,
        url_changed=True,
        visual_changed=False,
    )
    verified = decide_after_ocr(
        source="autonomous",
        markers_present=True,
        url_changed=True,
        visual_changed=False,
    )

    assert unverified.status == "unknown"
    assert unverified.reason == "transition_change_unverified"
    assert verified.status == "ready"
    assert verified.reason == "screen_change_url_matched"

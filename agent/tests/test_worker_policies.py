from agent.graph.worker_selection_policy import (
    SelectionPolicy,
    decide_queue_return,
    decide_selection_entry,
)
from agent.graph.worker_transition_policy import (
    decide_after_ocr,
    decide_before_ocr,
    decide_transition_probe,
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


def test_queue_return_wait_requires_saved_target_phash():
    missing_target = decide_queue_return(
        replay_available=False,
        is_return_action=True,
        ocr_complete=False,
        replay_reason="phash_mismatch",
        transition_needs_ocr=False,
        target_phash_available=False,
    )
    saved_target = decide_queue_return(
        replay_available=False,
        is_return_action=True,
        ocr_complete=False,
        replay_reason="phash_mismatch",
        transition_needs_ocr=False,
        target_phash_available=True,
    )

    assert missing_target.policy == SelectionPolicy.CONTINUE
    assert saved_target.policy == SelectionPolicy.WAIT_FOR_RESULTS_SCREEN


def test_transition_probe_changes_from_wait_to_ocr_fallback():
    waiting = decide_transition_probe(elapsed_sec=0.5, timeout_sec=2.0)
    timed_out = decide_transition_probe(elapsed_sec=2.0, timeout_sec=2.0)

    assert waiting.status == "pending"
    assert waiting.needs_ocr is False
    assert timed_out.status == "unknown"
    assert timed_out.needs_ocr is True


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

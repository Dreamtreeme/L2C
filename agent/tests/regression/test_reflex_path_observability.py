from agent.observability.reflex_paths import (
    reflex_selection_observation,
    reflex_step_observation,
    summarize_reflex_paths,
)


def test_reflex_path_observation_tracks_lifecycle_events() -> None:
    selected = reflex_selection_observation(
        {
            "replay": {"reflex_trace": {
                "hit": True,
                "recipe_key": "experience9#abc",
                "recipe_step_index": 0,
                "recipe_step_count": 2,
            }}
        }
    )
    completed = reflex_step_observation(
        {
            "transition": {"transition_result": {
                "source": "reflex",
                "status": "ready",
                "recipe_key": "experience9#abc",
                "recipe_step_index": 1,
                "recipe_step_count": 2,
                "source_reasoning_call_count": 2,
            }}
        }
    )
    failed = reflex_selection_observation(
        {
            "replay": {"reflex_trace": {
                "hit": False,
                "reason": "roi_phash_distance",
                "path_failed": True,
                "recipe_key": "experience9#abc",
                "recipe_step_index": 1,
                "recipe_step_count": 3,
            }}
        }
    )

    assert selected["reflex_path_event"] == "started"
    assert completed["reflex_path_event"] == "completed"
    assert completed["reflex_source_reasoning_replaced_count"] == 2
    assert completed["reflex_reasoning_call_reduction"] == 2
    assert failed["reflex_path_event"] == "failed"
    assert failed["reflex_fallback_required"] is True
    assert failed["reflex_path_failure_reason"] == "roi_phash_distance"


def test_reflex_path_summary_separates_step_hits_from_path_completion() -> None:
    steps = [
        {
            "component": "graph:reflex",
            "action_source": "reflex",
            "reflex_path_event": "started",
            "reflex_path_step_index": 0,
        },
        {
            "component": "graph:transition",
            "reflex_path_event": "step_completed",
            "reflex_path_step_index": 0,
            "reflex_source_reasoning_replaced_count": 1,
            "reflex_reasoning_call_reduction": 1,
        },
        {
            "component": "graph:reflex",
            "action_source": "reflex",
            "reflex_path_event": "step_selected",
            "reflex_path_step_index": 1,
        },
        {
            "component": "graph:transition",
            "reflex_path_event": "completed",
            "reflex_path_step_index": 1,
            "reflex_source_reasoning_replaced_count": 2,
            "reflex_reasoning_call_reduction": 2,
        },
    ]

    summary = summarize_reflex_paths(steps)

    assert summary["reflex_step_hit_count"] == 2
    assert summary["reflex_step_completed_count"] == 2
    assert summary["reflex_source_reasoning_replaced_count"] == 3
    assert summary["reflex_reasoning_call_reduction"] == 3
    assert summary["reflex_path_started_count"] == 1
    assert summary["reflex_path_completed_count"] == 1
    assert summary["reflex_path_completion_rate"] == 1.0


def test_reflex_pending_transition_does_not_count_reduction_early() -> None:
    observation = reflex_step_observation(
        {
            "transition": {
                "transition_result": {
                    "source": "reflex",
                    "status": "needs_ocr",
                    "recipe_key": "experience9#abc",
                    "recipe_step_index": 0,
                    "recipe_step_count": 1,
                    "source_reasoning_call_count": 1,
                }
            }
        }
    )

    assert "reflex_reasoning_call_reduction" not in observation

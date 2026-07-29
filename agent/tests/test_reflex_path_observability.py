from agent.observability.reflex_paths import (
    reflex_selection_observation,
    reflex_transition_observation,
    summarize_reflex_paths,
)


def test_reflex_path_observation_tracks_start_and_completion() -> None:
    selected = reflex_selection_observation(
        {
            "reflex_trace": {
                "hit": True,
                "recipe_key": "path4#abc",
                "recipe_step_index": 0,
                "recipe_step_count": 2,
            }
        }
    )
    completed = reflex_transition_observation(
        {
            "transition_result": {
                "source": "reflex",
                "status": "ready",
                "recipe_key": "path4#abc",
                "recipe_step_index": 1,
                "recipe_step_count": 2,
            }
        }
    )

    assert selected["reflex_path_event"] == "started"
    assert completed["reflex_path_event"] == "completed"


def test_reflex_path_observation_tracks_mid_path_fallback() -> None:
    failed = reflex_selection_observation(
        {
            "reflex_trace": {
                "hit": False,
                "reason": "roi_phash_distance",
                "path_failed": True,
                "recipe_key": "path4#abc",
                "recipe_step_index": 1,
                "recipe_step_count": 3,
            }
        }
    )

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
        },
    ]

    summary = summarize_reflex_paths(steps)

    assert summary["reflex_step_hit_count"] == 2
    assert summary["reflex_path_started_count"] == 1
    assert summary["reflex_path_completed_count"] == 1
    assert summary["reflex_path_completion_rate"] == 1.0

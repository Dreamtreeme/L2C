import pytest

from agent.application.recipe_candidate_review_service import (
    review_and_apply_candidate,
)
from agent.recipe.candidate_store import RecipeCandidateStore
from agent.recipe.promotion_policy import evaluate_candidate_step_evidence
from agent.recipe.store import RecipeStore
from agent.recipe.submission_store import SubmissionStore


def _state(capture_id, role, phash, url="wanted.co.kr/search"):
    return {
        "capture_id": capture_id,
        "url_template": url,
        "page_role": role,
        "screen_context_signature": {"phash": phash, "size": [1920, 1080]},
    }


def _feedback(seq, action, label="partial", *, marker_texts=None):
    return {
        "seq": seq,
        "proposal": {"action": action, "args": {}},
        "feedback": {"label": label},
        "observation": {
            "before": {
                "url": "https://www.wanted.co.kr/search",
                "marker_texts": marker_texts or [],
            },
            "result": {"status": "success"},
        },
    }


def _transition(seq, capture_id, role, phash, *, status="ready"):
    return {
        "action_seq": seq,
        "source": "autonomous",
        "status": status,
        "reason": "screen_change_pixels_matched",
        "after_state": _state(capture_id, role, phash),
    }


def _candidate_submission():
    return {
        "run_id": "worker-contract",
        "goal": "채용공고 수집",
        "collection_intent": {
            "site": "wanted",
            "search_keyword": "AI 엔지니어",
            "task_category": "검색",
        },
        "recorded_steps": [
            {
                "seq": 0,
                "decision_capture_id": "capture:0001",
                "url_template": "wanted.co.kr/",
                "page_role": "home",
                "before_state": _state(
                    "capture:0001", "home", "1" * 16, "wanted.co.kr/"
                ),
                "action": "click_marker",
                "replay_mode": "fixed",
                "component": "search_button",
                "target": {
                    "text": "검색",
                    "bbox_ratio": [0.75, 0.1, 0.85, 0.2],
                    "center_ratio": [0.8, 0.15],
                },
                "roi_signature": {
                    "algorithm": "roi-phash-dct64-v2",
                    "phash": "0" * 16,
                    "crop_rect_ratio": [0.7, 0.0, 0.9, 0.3],
                    "target_center_ratio": [0.8, 0.15],
                },
            }
        ],
        "transition_records": [
            _transition(0, "capture:0002", "search_overlay", "2" * 16)
        ],
        "feedback_episodes": [
            _feedback(0, "click_marker", "success", marker_texts=["채용", "검색"])
        ],
    }


def _store_candidate(db_path, submission):
    submission_id = SubmissionStore(db_path).commit_submission(
        submission,
        source="test",
    )
    return RecipeCandidateStore(db_path).commit_candidate(
        submission,
        submission_id=submission_id,
    )


def _promote(db_path, submission, verdicts):
    candidate_id = _store_candidate(db_path, submission)
    return review_and_apply_candidate(
        candidate_id,
        db_path=db_path,
        mode="promote",
        critic=lambda _payload: {
            "decision": "accept",
            "reasons": ["실행 증거 확인"],
            "feedback_to_worker": "",
            "step_verdicts": verdicts,
            "confidence": 0.9,
        },
    )


def test_candidate_promotion_keeps_only_safe_roi_target(tmp_path):
    submission = _candidate_submission()
    submission["recorded_steps"].extend(
        [
            {
                "seq": 1,
                "page_role": "search",
                "action": "click_marker",
                "component": "job_card_title",
                "target": {"text": "실행마다 달라지는 공고"},
                "roi_signature": {
                    "phash": "f" * 16,
                    "crop_rect_ratio": [0.1, 0.2, 0.6, 0.4],
                },
            },
            {
                "seq": 2,
                "action": "press_key",
                "replay_mode": "fixed",
                "param": {"key": "enter"},
            },
        ]
    )
    db_path = tmp_path / "critic.db"
    result = _promote(
        db_path,
        submission,
        [
            {"seq": 0, "keep": True},
            {"seq": 2, "keep": False},
        ],
    )

    recipes = RecipeStore(db_path).get_by_site("wanted")
    assert result["promotion"]["promoted_action_count"] == 1
    assert recipes[0]["transitions"][0]["actions"][0]["action"] == "click_marker"


def test_candidate_without_recorded_after_state_is_not_promoted(tmp_path):
    submission = _candidate_submission()
    submission["transition_records"][0].pop("after_state")

    result = _promote(
        tmp_path / "missing-after-state.db",
        submission,
        [{"seq": 0, "keep": True}],
    )

    assert result["promotion"]["promoted_action_count"] == 0


def test_critic_cannot_rewrite_autonomous_recipe_fields(tmp_path):
    db_path = tmp_path / "critic-authority.db"
    candidate_id = _store_candidate(db_path, _candidate_submission())

    def critic(payload):
        assert payload["required_step_verdicts"] == [
            {"seq": 0, "action": "click_marker"}
        ]
        return {
            "decision": "accept",
            "reasons": ["원본 단계 유지"],
            "step_verdicts": [{"seq": 0, "keep": True}],
            "skill_metadata": {"task_category": "결제"},
            "transition_contracts": [{"seq": 0, "contract": {"values": ["임의"]}}],
            "confidence": 0.9,
        }

    result = review_and_apply_candidate(
        candidate_id,
        db_path=db_path,
        mode="promote",
        critic=critic,
    )
    recipe = RecipeStore(db_path).get_by_site("wanted")[0]
    action = recipe["transitions"][0]["actions"][0]

    assert action["action"] == "click_marker"
    assert action["component"] == "search_button"
    assert "transition_contract" not in action
    assert recipe["skill_metadata"]["task_category"] == "검색"
    assert "skill_metadata" not in result


def test_contextual_actions_are_promoted_as_one_verified_path(tmp_path):
    submission = _candidate_submission()
    submission["recorded_steps"].extend(
        [
            {
                "seq": 1,
                "before_state": _state("capture:0002", "search_overlay", "2" * 16),
                "page_role": "search_overlay",
                "action": "type_in_marker",
                "replay_mode": "parameterized",
                "component": "search_input",
                "target": {"text": "검색어"},
                "param": {"text": "AI 엔지니어", "slot_name": "query"},
                "slot_refs": ["query"],
                "roi_signature": {
                    "phash": "1" * 16,
                    "crop_rect_ratio": [0.1, 0.1, 0.7, 0.2],
                },
            },
            {
                "seq": 2,
                "before_state": _state("capture:0003", "search_overlay", "3" * 16),
                "page_role": "search_overlay",
                "action": "press_key",
                "replay_mode": "fixed",
                "param": {"key": "enter"},
                "screen_context_signature": {
                    "phash": "2" * 16,
                    "size": [1920, 1080],
                },
            },
        ]
    )
    submission["feedback_episodes"].extend(
        [_feedback(1, "type_in_marker"), _feedback(2, "press_key")]
    )
    submission["transition_records"].extend(
        [
            _transition(1, "capture:0003", "search_overlay", "3" * 16),
            _transition(2, "capture:0004", "search_results", "4" * 16),
        ]
    )
    db_path = tmp_path / "grouped-path.db"
    result = _promote(
        db_path,
        submission,
        [{"seq": seq, "keep": True} for seq in (0, 1, 2)],
    )
    recipe = RecipeStore(db_path).get_by_site("wanted")[0]

    assert result["promotion"]["promoted_transition_count"] == 2
    assert [
        action["action"]
        for transition in recipe["transitions"]
        for action in transition["actions"]
    ] == ["click_marker", "type_in_marker", "press_key"]
    assert [action["action"] for action in recipe["transitions"][1]["actions"]] == [
        "type_in_marker",
        "press_key",
    ]


@pytest.mark.parametrize(
    ("enter_capture", "eligible"),
    [("capture:0003", True), ("capture:0099", False)],
)
def test_type_and_enter_require_state_continuity(enter_capture, eligible):
    candidate = {
        "steps": [
            {
                "seq": 2,
                "action": "type_in_marker",
                "replay_mode": "parameterized",
                "param": {"slot_name": "query", "text": "iOS 개발자"},
                "before_state": {"capture_id": "capture:0002"},
            },
            {
                "seq": 4,
                "action": "press_key",
                "replay_mode": "fixed",
                "param": {"key": "enter"},
                "before_state": {"capture_id": enter_capture},
            },
        ],
        "payload": {
            "feedback_episodes": [
                _feedback(2, "type_in_marker"),
                _feedback(4, "press_key"),
            ],
            "transition_records": [
                {
                    "action_seq": 2,
                    "source": "autonomous",
                    "status": "unknown",
                    "reason": "no_screen_change",
                    "after_state": {"capture_id": "capture:0003"},
                },
                {
                    "action_seq": 4,
                    "source": "autonomous",
                    "status": "ready",
                    "reason": "screen_change_pixels_matched",
                    "after_state": {"capture_id": "capture:0004"},
                },
            ],
        },
    }

    verdict = evaluate_candidate_step_evidence(candidate)[2]

    assert verdict["eligible"] is eligible
    if eligible:
        assert verdict["execution_group_seqs"] == [2, 4]
        assert verdict["effect_verified_by_seq"] == 4
    else:
        assert verdict["blocking_reasons"] == ["no_screen_change"]


def test_candidate_promotion_blocks_no_effect_step(tmp_path):
    submission = _candidate_submission()
    submission["feedback_episodes"][0]["feedback"] = {
        "label": "no_effect",
        "reason": "screen_unchanged",
    }
    db_path = tmp_path / "no-effect.db"
    result = _promote(db_path, submission, [{"seq": 0, "keep": True}])

    assert result["promotion"]["promoted"] is False
    assert result["promotion"]["skipped_steps"][0]["reason"] == "feedback_no_effect"
    assert RecipeStore(db_path).get_by_site("wanted") == []

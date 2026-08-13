import json
import sqlite3

from PIL import Image

from agent.application.recipe_candidate_review_service import (
    build_candidate_review_payload,
    review_and_apply_candidate,
)
from agent.recipe.candidate_promotion import reviewable_candidate_transitions
from agent.recipe.candidate_store import RecipeCandidateStore
from agent.recipe.store import RecipeStore
from agent.recipe.submission_store import SubmissionStore
from shared.schema.feedback_schema import RecipeCandidate, WorkerSubmission


def _state(observation_id, role, phash, url="wanted.co.kr/search"):
    return {
        "observation_id": observation_id,
        "url_template": url,
        "page_role": role,
        "screen_context_signature": {
            "phash": phash,
            "size": [1920, 1080],
        },
    }


def _target_action(seq, action="click_marker", **values):
    payload = {
        "source_seq": seq,
        "action": action,
        "replay_mode": "fixed",
        "target": {
            "text": values.pop("text", "검색"),
            "marker_type": "text",
            "center_ratio": [0.8, 0.15],
        },
        "roi_signature": {
            "phash": str(seq % 10) * 16,
            "crop_rect_ratio": [0.7, 0.0, 0.9, 0.3],
        },
    }
    payload.update(values)
    return payload


def _event(
    seq,
    before,
    actions,
    after,
    *,
    transition_seq=None,
    source="autonomous",
    result_status="success",
    status="ready",
    reason="screen_change_pixels_matched",
):
    final_action = actions[-1]
    return {
        "seq": seq,
        "result": {
            "action": final_action["action"],
            "status": result_status,
        },
        "candidate_action": final_action,
        "before_checkpoint": before,
        "transition": {
            "seq": actions[0]["source_seq"] if transition_seq is None else transition_seq,
            "before": before,
            "actions": actions,
            "after": after,
            "evidence": {
                "source": source,
                "result_status": result_status,
                "status": status,
                "reason": reason,
                "visual_change_ratio": 0.42,
                "after_marker_texts": ["채용", "검색"],
            },
        },
    }


def _candidate_submission():
    before = _state("observation:0001", "home", "1" * 16, "wanted.co.kr/")
    after = _state("observation:0002", "search_overlay", "2" * 16)
    return {
        "run_id": "worker-contract",
        "goal": "채용공고 수집",
        "collection_intent": {
            "site": "wanted",
            "search_keyword": "AI 엔지니어",
            "task_category": "검색",
        },
        "action_events": [_event(0, before, [_target_action(0)], after)],
    }


def _store_candidate(db_path, submission):
    parsed = WorkerSubmission.model_validate(submission)
    run_id = SubmissionStore(db_path).commit_submission(parsed, source="test")
    return RecipeCandidateStore(db_path).commit_candidate(parsed, run_id=run_id)


def _promote(db_path, submission, verdicts):
    run_id = _store_candidate(db_path, submission)
    return review_and_apply_candidate(
        run_id,
        db_path=db_path,
        mode="promote",
        critic=lambda _payload: {
            "decision": "accept",
            "reasons": ["성공 경로 확인"],
            "transition_verdicts": verdicts,
        },
    )


def test_candidate_promotion_stores_recorded_transition_without_rebuilding(tmp_path):
    db_path = tmp_path / "direct-transition.db"

    result = _promote(db_path, _candidate_submission(), [{"seq": 0, "keep": True}])

    recipe = RecipeStore(db_path).get_by_site("wanted")[0]
    assert result["promotion"]["promoted_transition_count"] == 1
    assert recipe["transitions"][0]["actions"][0]["action"] == "click_marker"


def test_candidate_without_recorded_transition_is_not_stored(tmp_path):
    submission = _candidate_submission()
    submission["action_events"][0].pop("transition")

    assert _store_candidate(tmp_path / "missing.db", submission) == ""


def test_candidate_store_ignores_legacy_contract_without_deleting_it(tmp_path):
    db_path = tmp_path / "legacy-candidate.db"
    submission = WorkerSubmission.model_validate(_candidate_submission())
    run_id = SubmissionStore(db_path).commit_submission(submission, source="test")
    store = RecipeCandidateStore(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO recipe_candidates "
            "(run_id, contract_version, status, validation_json, review_attempts, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (
                run_id,
                2,
                "pending_review",
                "",
                0,
                "2026-08-13T00:00:00",
                "2026-08-13T00:00:00",
            ),
        )

    assert store.get_candidate(run_id) is None
    assert store.claim_review(run_id) is None
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM recipe_candidates "
            "WHERE run_id=? AND contract_version=2",
            (run_id,),
        ).fetchone()[0] == 1


def test_critic_has_pruning_authority_only(tmp_path):
    db_path = tmp_path / "critic-authority.db"
    run_id = _store_candidate(db_path, _candidate_submission())

    result = review_and_apply_candidate(
        run_id,
        db_path=db_path,
        mode="promote",
        critic=lambda _payload: {
            "decision": "accept",
            "transition_verdicts": [{"seq": 0, "keep": True}],
            "skill_metadata": {"task_category": "결제"},
        },
    )

    assert result["decision"] == "reject"
    assert RecipeStore(db_path).get_by_site("wanted") == []


def test_critic_payload_uses_transition_groups_without_raw_screen_blobs():
    submission = _candidate_submission()
    candidate = RecipeCandidate.from_submission(
        WorkerSubmission.model_validate(submission),
        run_id="worker-compact",
        status="recorded",
    )

    payload = build_candidate_review_payload(candidate)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["required_transition_verdicts"] == [
        {"seq": 0, "actions": ["click_marker"]}
    ]
    assert payload["candidate_transitions"][0]["actions"][0]["roi_available"]
    assert "1" * 16 not in serialized


def test_critic_payload_lists_ordered_screen_evidence(tmp_path):
    before_image = tmp_path / "before.png"
    after_image = tmp_path / "after.png"
    Image.new("RGB", (100, 100), "white").save(before_image)
    Image.new("RGB", (100, 100), "black").save(after_image)
    submission = _candidate_submission()
    event = submission["action_events"][0]
    event["result"]["before_marked_image"] = str(before_image)
    event["transition"]["evidence"]["marked_image"] = str(after_image)
    candidate = RecipeCandidate.from_submission(
        WorkerSubmission.model_validate(submission),
        run_id="screen-evidence",
        status="recorded",
    )

    payload = build_candidate_review_payload(candidate)

    assert payload["screen_evidence_order"] == [
        "transition 0 before",
        "transition 0 after",
    ]


def test_input_and_commit_remain_one_recorded_transition(tmp_path):
    submission = _candidate_submission()
    before = _state("observation:0002", "search_overlay", "2" * 16)
    type_action = _target_action(
        1,
        "type_in_marker",
        text="검색어",
        replay_mode="parameterized",
        param={"text": "AI 엔지니어", "slot_name": "search_keyword"},
        slot_refs=["search_keyword"],
        component="search_input",
    )
    commit_action = {
        "source_seq": 2,
        "action": "press_key",
        "replay_mode": "fixed",
        "param": {"key": "enter"},
    }
    submission["action_events"].extend(
        [
            {
                "seq": 1,
                "result": {"action": "type_in_marker", "status": "success"},
                "candidate_action": type_action,
                "before_checkpoint": before,
            },
            _event(
                2,
                before,
                [type_action, commit_action],
                _state("observation:0003", "search_results", "3" * 16),
            ),
        ]
    )
    db_path = tmp_path / "group.db"

    result = _promote(
        db_path,
        submission,
        [{"seq": seq, "keep": True} for seq in (0, 1)],
    )

    recipe = RecipeStore(db_path).get_by_site("wanted")[0]
    assert result["promotion"]["promoted_transition_count"] == 2
    assert [
        action["action"] for action in recipe["transitions"][1]["actions"]
    ] == ["type_in_marker", "press_key"]


def test_standalone_input_is_kept_as_preparation_for_following_action(tmp_path):
    submission = _candidate_submission()
    input_before = _state("observation:0002", "search_overlay", "2" * 16)
    input_after = _state("observation:0003", "search_overlay", "2" * 16)
    type_action = _target_action(
        1,
        "type_in_marker",
        text="검색어",
        replay_mode="parameterized",
        param={"text": "AI 엔지니어", "slot_name": "search_keyword"},
        slot_refs=["search_keyword"],
        component="search_input",
    )
    submission["action_events"].extend(
        [
            _event(
                1,
                input_before,
                [type_action],
                input_after,
                status="unknown",
                reason="no_screen_change",
            ),
            _event(
                2,
                input_after,
                [_target_action(2, text="검색 실행")],
                _state("observation:0004", "search_results", "3" * 16),
            ),
        ]
    )
    candidate = RecipeCandidate.from_submission(
        WorkerSubmission.model_validate(submission),
        run_id="standalone-input",
        status="recorded",
    )

    transitions, pruned = reviewable_candidate_transitions(candidate)
    payload = build_candidate_review_payload(candidate)

    assert [transition.seq for transition in transitions] == [0, 1, 2]
    assert pruned == []
    assert payload["candidate_transitions"][1]["prepares_transition_seq"] == 2

    db_path = tmp_path / "standalone-input.db"
    result = _promote(
        db_path,
        submission,
        [{"seq": seq, "keep": True} for seq in (0, 1, 2)],
    )

    recipe = RecipeStore(db_path).get_by_site("wanted")[0]
    assert result["promotion"]["promoted_transition_count"] == 3
    assert [
        transition["actions"][0]["action"]
        for transition in recipe["transitions"]
    ] == ["click_marker", "type_in_marker", "click_marker"]


def test_standalone_input_is_not_kept_when_following_enter_cannot_replay():
    submission = _candidate_submission()
    input_before = _state("observation:0002", "search_overlay", "2" * 16)
    input_after = _state("observation:0003", "search_overlay", "2" * 16)
    type_action = _target_action(
        1,
        "type_in_marker",
        replay_mode="parameterized",
        param={"text": "AI 엔지니어", "slot_name": "search_keyword"},
        slot_refs=["search_keyword"],
    )
    enter_action = {
        "source_seq": 2,
        "action": "press_key",
        "replay_mode": "fixed",
        "param": {"key": "enter"},
    }
    submission["action_events"].extend(
        [
            _event(
                1,
                input_before,
                [type_action],
                input_after,
                status="unknown",
                reason="no_screen_change",
            ),
            _event(
                2,
                input_after,
                [enter_action],
                _state("observation:0004", "search_results", "3" * 16),
            ),
        ]
    )
    candidate = RecipeCandidate.from_submission(
        WorkerSubmission.model_validate(submission),
        run_id="standalone-enter",
        status="recorded",
    )

    transitions, pruned = reviewable_candidate_transitions(candidate)

    assert [transition.seq for transition in transitions] == [0]
    assert {(item.seq, item.reason) for item in pruned} == {
        (1, "transition_not_ready"),
        (2, "unsupported_action_group"),
    }


def test_pruned_wrong_branch_reconnects_at_same_recorded_screen(tmp_path):
    submission = _candidate_submission()
    branch_screen = _state("observation:0010", "search_overlay", "2" * 16)
    wrong_after = _state("observation:0011", "unrelated", "9" * 16)
    submission["action_events"].extend(
        [
            _event(1, branch_screen, [_target_action(1, text="잘못된 메뉴")], wrong_after),
            _event(
                2,
                wrong_after,
                [{"source_seq": 2, "action": "go_back"}],
                _state("observation:0012", "search_overlay", "2" * 16),
            ),
            _event(
                3,
                _state("observation:0012", "search_overlay", "2" * 16),
                [_target_action(3, text="올바른 검색")],
                _state("observation:0013", "search_results", "3" * 16),
            ),
        ]
    )
    db_path = tmp_path / "pruned-branch.db"

    result = _promote(
        db_path,
        submission,
        [
            {"seq": 0, "keep": True},
            {"seq": 1, "keep": False, "reason": "abandoned branch"},
            {"seq": 3, "keep": True},
        ],
    )

    recipe = RecipeStore(db_path).get_by_site("wanted")[0]
    assert result["promotion"]["promoted_path_count"] == 1
    assert [transition["seq"] for transition in recipe["transitions"]] == [0, 1]
    assert [
        transition["actions"][0]["target"]["text"]
        for transition in recipe["transitions"]
    ] == ["검색", "올바른 검색"]


def test_pruned_gap_with_nearby_phash_rejects_disconnected_path(tmp_path):
    submission = _candidate_submission()
    branch_screen = _state("observation:0010", "search_overlay", "2" * 16)
    wrong_after = _state("observation:0011", "unrelated", "9" * 16)
    nearby_screen = _state("observation:0012", "search_overlay", "6" + "2" * 15)
    submission["action_events"].extend(
        [
            _event(1, branch_screen, [_target_action(1, text="잘못된 메뉴")], wrong_after),
            _event(
                2,
                wrong_after,
                [{"source_seq": 2, "action": "go_back"}],
                nearby_screen,
            ),
            _event(
                3,
                nearby_screen,
                [_target_action(3, text="다음 검색")],
                _state("observation:0013", "search_results", "3" * 16),
            ),
        ]
    )
    db_path = tmp_path / "nearby-screen.db"

    result = _promote(
        db_path,
        submission,
        [
            {"seq": 0, "keep": True},
            {"seq": 1, "keep": False, "reason": "abandoned branch"},
            {"seq": 3, "keep": True},
        ],
    )

    assert result["promotion"]["promoted"] is False
    assert result["promotion"]["promoted_path_count"] == 0
    assert result["decision"] == "reject"
    assert RecipeCandidateStore(db_path).get_candidate(
        submission["run_id"]
    ).status == "rejected"
    assert RecipeStore(db_path).get_by_site("wanted") == []
    assert any(
        item["reason"] == "path_disconnected_after_pruning"
        for item in result["promotion"]["pruned_transitions"]
    )


def test_non_autonomous_or_failed_transition_is_filtered_before_critic():
    submission = _candidate_submission()
    submission["action_events"].extend(
        [
            _event(
                1,
                _state("observation:0002", "search_overlay", "2" * 16),
                [_target_action(1)],
                _state("observation:0003", "job_detail", "3" * 16),
                source="job_card_queue",
            ),
            _event(
                2,
                _state("observation:0003", "job_detail", "3" * 16),
                [_target_action(2)],
                _state("observation:0004", "job_detail", "4" * 16),
                result_status="skipped",
            ),
        ]
    )
    candidate = RecipeCandidate.from_submission(
        WorkerSubmission.model_validate(submission),
        run_id="filtered",
        status="recorded",
    )

    transitions, pruned = reviewable_candidate_transitions(candidate)

    assert [transition.seq for transition in transitions] == [0]
    assert {(item.seq, item.reason) for item in pruned} == {
        (1, "not_autonomous"),
        (2, "action_skipped"),
    }


def test_invalid_critic_contract_is_not_called_again(tmp_path):
    db_path = tmp_path / "one-call.db"
    run_id = _store_candidate(db_path, _candidate_submission())
    calls = 0

    def critic(_payload):
        nonlocal calls
        calls += 1
        return {"decision": "accept", "transition_verdicts": []}

    result = review_and_apply_candidate(
        run_id,
        db_path=db_path,
        mode="promote",
        critic=critic,
    )

    assert calls == 1
    assert result["decision"] == "reject"
    assert RecipeStore(db_path).get_by_site("wanted") == []

import json

from agent.application.recipe_candidate_review_service import (
    build_candidate_review_payload,
    review_and_apply_candidate,
)
from agent.recipe.candidate_store import RecipeCandidateStore
from agent.recipe.store import ExperienceRuleStore
from agent.recipe.submission_store import SubmissionStore
from shared.schema.feedback_schema import RecipeCandidate, WorkerSubmission


def _screen(observation_id, role, phash, url):
    return {
        "observation_id": observation_id,
        "url_template": url,
        "page_role": role,
        "screen_context_signature": {"phash": phash, "size": [1000, 800]},
    }


def _target_action(seq, action="click_marker", *, text="검색", slot_name=None):
    param = {"text": text}
    if slot_name:
        param["slot_name"] = slot_name
    return {
        "source_seq": seq,
        "action": action,
        "target": {
            "text": text,
            "marker_type": "text",
            "bbox_ratio": [0.7, 0.1, 0.9, 0.2],
            "center_ratio": [0.8, 0.15],
        },
        "roi_signature": {
            "phash": str(seq + 1) * 16,
            "crop_rect_ratio": [0.65, 0.05, 0.95, 0.25],
        },
        "param": param if action == "type_in_marker" else {},
        "slot_refs": [slot_name] if slot_name else [],
        "intent": "검색을 수행한다",
        "target_role": "search_control",
        "component": "search_panel",
    }


def _press_key_action(seq, key="Enter"):
    return {
        "source_seq": seq,
        "action": "press_key",
        "param": {"key": key},
        "intent": "입력한 검색어를 제출한다",
    }


def _event(seq, before, actions, after, *, status="ready", reason="screen_change"):
    return {
        "seq": seq,
        "result": {"action": actions[-1]["action"], "status": "success"},
        "candidate_action": actions[-1],
        "before_checkpoint": before,
        "transition": {
            "seq": seq,
            "before": before,
            "actions": actions,
            "after": after,
            "intent": "검색 화면으로 이동한다",
            "evidence": {
                "source": "autonomous",
                "result_status": "success",
                "status": status,
                "reason": reason,
                "visual_change_ratio": 0.2,
            },
        },
    }


def _submission(events):
    return WorkerSubmission.model_validate(
        {
            "run_id": "worker-rule-test",
            "goal": "AI 엔지니어 채용공고 검색",
            "run_status": "success",
            "collected_count": 2,
            "persisted_count": 2,
            "collection_intent": {
                "site": "wanted",
                "search_keyword": "AI 엔지니어",
                "task_category": "검색",
            },
            "action_events": events,
        }
    )


def _store_candidate(db_path, submission):
    run_id = SubmissionStore(db_path).commit_submission(submission, source="test")
    RecipeCandidateStore(db_path).commit_candidate(submission, run_id=run_id)
    return run_id


def _critic(seqs):
    return lambda _payload: {
        "decision": "accept",
        "reasons": ["성공 경로"],
        "transition_verdicts": [
            {"seq": seq, "keep": True, "reason": "성공 경로"} for seq in seqs
        ],
    }


def _compiler(source_transition_seqs, action_seqs, *, grouped=False):
    def compile_rule(_payload):
        return {
            "steps": [
                {
                    "source_transition_seqs": source_transition_seqs,
                    "intent": "검색 요청을 제출한다",
                    "applicable_when": "검색 입력 영역이 보인다",
                    "decline_when": "검색 입력 영역이 둘 이상이다",
                    "actions": [
                        {
                            "source_action_seq": seq,
                            "target_description": (
                                "검색어 입력 영역" if seq == 0 and grouped else "검색 버튼"
                            ),
                            "target_role": "search_control",
                            "target_component": "search_panel",
                            "spatial_relation": "검색 패널 내부",
                            "input_slot": "search_keyword" if seq == 0 and grouped else "",
                        }
                        for seq in action_seqs
                    ],
                    "expected_effect": {
                        "kind": "page_change",
                        "description": "검색 결과 화면이 나타난다",
                    },
                }
            ]
        }

    return compile_rule


def test_critic_and_compiler_store_separate_experience_rule(tmp_path):
    before = _screen("o1", "home", "1" * 16, "wanted.co.kr/")
    after = _screen("o2", "search", "2" * 16, "wanted.co.kr/search")
    submission = _submission([_event(0, before, [_target_action(0)], after)])
    run_id = _store_candidate(tmp_path / "rules.db", submission)

    result = review_and_apply_candidate(
        run_id,
        db_path=tmp_path / "rules.db",
        mode="promote",
        critic=_critic([0]),
        compiler=_compiler([0], [0]),
    )

    stored = ExperienceRuleStore(tmp_path / "rules.db").get_by_site("wanted")
    assert result["promotion"]["rule_step_count"] == 1
    assert stored[0]["steps"][0]["source_transition_seqs"] == [0]
    assert stored[0]["steps"][0]["actions"][0]["action"] == "click_marker"


def test_input_and_submit_actions_compile_into_one_rule_step(tmp_path):
    before = _screen("o1", "home", "1" * 16, "wanted.co.kr/")
    after = _screen("o2", "search", "2" * 16, "wanted.co.kr/search")
    events = [
        _event(
            0,
            before,
            [
                _target_action(
                    0,
                    "type_in_marker",
                    text="AI 엔지니어",
                    slot_name="search_keyword",
                ),
                _press_key_action(1),
            ],
            after,
        )
    ]
    db_path = tmp_path / "grouped.db"
    run_id = _store_candidate(db_path, _submission(events))

    result = review_and_apply_candidate(
        run_id,
        db_path=db_path,
        mode="promote",
        critic=_critic([0]),
        compiler=_compiler([0], [0, 1], grouped=True),
    )

    rule = ExperienceRuleStore(db_path).get_by_site("wanted")[0]
    assert result["promotion"]["rule_action_count"] == 2
    assert len(rule["steps"]) == 1
    assert [action["action"] for action in rule["steps"][0]["actions"]] == [
        "type_in_marker",
        "press_key",
    ]
    assert "target" not in rule["steps"][0]["actions"][1]
    assert rule["skill_metadata"]["inputs"] == [{"name": "search_keyword"}]


def test_compiler_cannot_invent_source_action(tmp_path):
    before = _screen("o1", "home", "1" * 16, "wanted.co.kr/")
    after = _screen("o2", "search", "2" * 16, "wanted.co.kr/search")
    db_path = tmp_path / "invalid.db"
    run_id = _store_candidate(
        db_path,
        _submission([_event(0, before, [_target_action(0)], after)]),
    )

    result = review_and_apply_candidate(
        run_id,
        db_path=db_path,
        mode="promote",
        critic=_critic([0]),
        compiler=_compiler([0], [999]),
    )

    assert result["decision"] == "reject"
    assert ExperienceRuleStore(db_path).get_by_site("wanted") == []


def test_critic_payload_contains_observation_not_replay_policy():
    before = _screen("o1", "home", "1" * 16, "wanted.co.kr/")
    after = _screen("o2", "search", "2" * 16, "wanted.co.kr/search")
    submission = _submission([_event(0, before, [_target_action(0)], after)])
    candidate = RecipeCandidate.from_submission(
        submission,
        run_id=submission.run_id,
        status="recorded",
    )
    payload = build_candidate_review_payload(candidate)

    assert "replay_mode" not in json.dumps(payload, ensure_ascii=False)

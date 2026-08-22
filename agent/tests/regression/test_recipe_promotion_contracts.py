import json

from agent.application.recipe_candidate_review_service import (
    build_candidate_review_payload,
    review_and_apply_candidate,
)
from agent.application.recipe_execution_graph_service import (
    build_candidate_execution_graph,
    build_candidate_graph_payload,
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


def _critic(node_ids):
    return lambda _payload: {
        "decision": "accept",
        "reasons": ["성공 경로"],
        "node_verdicts": [
            {"node_id": node_id, "keep": True, "reason": "성공 경로"}
            for node_id in node_ids
        ],
    }


def _graph_builder(payload):
    seqs = [event["seq"] for event in payload["flat_log"]]
    return {
        "goal": payload["goal"],
        "nodes": [
            {
                "node_id": "node-1",
                "purpose": "검색 요청을 수행한다",
                "source_event_seqs": seqs,
                "grouping_reason": "동일한 검색 목적의 연속 행동",
            }
        ],
        "edges": [],
        "unassigned_event_seqs": [],
    }


def test_critic_and_deterministic_compiler_store_experience_rule(tmp_path):
    before = _screen("o1", "home", "1" * 16, "wanted.co.kr/")
    after = _screen("o2", "search", "2" * 16, "wanted.co.kr/search")
    submission = _submission([_event(0, before, [_target_action(0)], after)])
    run_id = _store_candidate(tmp_path / "rules.db", submission)

    result = review_and_apply_candidate(
        run_id,
        db_path=tmp_path / "rules.db",
        mode="promote",
        graph_builder=_graph_builder,
        critic=_critic(["node-1"]),
    )

    stored = ExperienceRuleStore(tmp_path / "rules.db").get_by_site("wanted")
    assert result["promotion"]["rule_step_count"] == 1
    assert stored[0]["nodes"][0]["node_id"] == "node-1"
    assert stored[0]["steps"][0]["source_node_id"] == "node-1"
    assert stored[0]["steps"][0]["source_transition_seqs"] == [0]
    assert stored[0]["steps"][0]["actions"][0]["action"] == "click_marker"
    assert "applicable_when" not in stored[0]["steps"][0]
    assert "decline_when" not in stored[0]["steps"][0]


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
        graph_builder=_graph_builder,
        critic=_critic(["node-1"]),
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


def test_graph_node_is_preserved_while_each_screen_transition_stays_separate(
    tmp_path,
):
    first = _screen("o1", "search", "1" * 16, "wanted.co.kr/search")
    second = _screen("o2", "search", "2" * 16, "wanted.co.kr/search")
    third = _screen("o3", "search", "3" * 16, "wanted.co.kr/search")
    fourth = _screen("o4", "search", "4" * 16, "wanted.co.kr/search")
    events = [
        _event(0, first, [_target_action(0, text="직군")], second),
        _event(1, second, [_target_action(1, text="SW 개발")], third),
        _event(2, third, [_target_action(2, text="적용")], fourth),
    ]
    db_path = tmp_path / "node-boundary.db"
    run_id = _store_candidate(db_path, _submission(events))

    result = review_and_apply_candidate(
        run_id,
        db_path=db_path,
        mode="promote",
        graph_builder=_graph_builder,
        critic=_critic(["node-1"]),
    )

    rule = ExperienceRuleStore(db_path).get_by_site("wanted")[0]
    assert result["promotion"]["rule_step_count"] == 3
    assert rule["nodes"] == [
        {
            "node_id": "node-1",
            "purpose": "검색 요청을 수행한다",
            "source_event_seqs": [0, 1, 2],
            "step_ids": ["step-1", "step-2", "step-3"],
        }
    ]
    assert [step["source_node_id"] for step in rule["steps"]] == [
        "node-1",
        "node-1",
        "node-1",
    ]
    assert [step["source_transition_seqs"] for step in rule["steps"]] == [
        [0],
        [1],
        [2],
    ]


def test_critic_prunes_a_whole_graph_node_before_rule_compilation(tmp_path):
    first = _screen("o1", "search", "1" * 16, "wanted.co.kr/search")
    second = _screen("o2", "detail", "2" * 16, "wanted.co.kr/jobs/wrong")
    third = _screen("o3", "search", "3" * 16, "wanted.co.kr/search")
    events = [
        _event(0, first, [_target_action(0, text="관련 없는 공고")], second),
        _event(1, second, [_target_action(1, text="목록 복귀")], third),
    ]
    db_path = tmp_path / "pruned-node.db"
    run_id = _store_candidate(db_path, _submission(events))

    def graph_builder(payload):
        return {
            "goal": payload["goal"],
            "nodes": [
                {
                    "node_id": "wrong-branch",
                    "purpose": "관련 없는 상세 진입",
                    "source_event_seqs": [0],
                },
                {
                    "node_id": "resume-search",
                    "purpose": "검색 목록 복귀",
                    "source_event_seqs": [1],
                },
            ],
            "edges": [
                {
                    "from_node": "wrong-branch",
                    "to_node": "resume-search",
                    "relation": "recovery",
                }
            ],
        }

    def critic(_payload):
        return {
            "decision": "accept",
            "node_verdicts": [
                {
                    "node_id": "wrong-branch",
                    "keep": False,
                    "reason": "실패한 탐색 분기",
                },
                {
                    "node_id": "resume-search",
                    "keep": True,
                    "reason": "재사용 가능한 복귀 단계",
                },
            ],
        }

    result = review_and_apply_candidate(
        run_id,
        db_path=db_path,
        mode="promote",
        graph_builder=graph_builder,
        critic=critic,
    )

    rule = ExperienceRuleStore(db_path).get_by_site("wanted")[0]
    assert [node["node_id"] for node in rule["nodes"]] == ["resume-search"]
    assert rule["steps"][0]["source_transition_seqs"] == [1]
    assert result["promotion"]["pruned_nodes"][0]["node_id"] == "wrong-branch"


def test_promotion_rejects_whole_candidate_when_critic_keeps_sensitive_action(
    tmp_path,
):
    before = _screen("o1", "detail", "1" * 16, "wanted.co.kr/detail")
    after = _screen("o2", "payment", "2" * 16, "wanted.co.kr/payment")
    action = _target_action(0, text="결제")
    action["risk_level"] = "sensitive"
    db_path = tmp_path / "sensitive.db"
    run_id = _store_candidate(
        db_path,
        _submission([_event(0, before, [action], after)]),
    )

    result = review_and_apply_candidate(
        run_id,
        db_path=db_path,
        mode="promote",
        graph_builder=_graph_builder,
        critic=_critic(["node-1"]),
    )

    assert result["decision"] == "reject"
    assert "sensitive action" in result["reasons"][-1]
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
    graph = build_candidate_execution_graph(candidate, builder=_graph_builder)
    payload = build_candidate_review_payload(candidate, graph)

    assert "replay_mode" not in json.dumps(payload, ensure_ascii=False)
    assert payload["actionable_nodes"][0]["transitions"][0][
        "replay_supported"
    ] is True


def test_critic_payload_marks_targetless_scroll_as_not_replayable():
    before = _screen("o1", "search", "1" * 16, "wanted.co.kr/search")
    after = _screen("o2", "search", "2" * 16, "wanted.co.kr/search")
    scroll = {
        "source_seq": 0,
        "action": "scroll",
        "param": {"direction": "down", "amount": "page"},
        "intent": "검색 결과를 더 본다",
    }
    submission = _submission([_event(0, before, [scroll], after)])
    candidate = RecipeCandidate.from_submission(
        submission,
        run_id=submission.run_id,
        status="recorded",
    )
    graph = build_candidate_execution_graph(candidate, builder=_graph_builder)

    payload = build_candidate_review_payload(candidate, graph)

    assert payload["actionable_nodes"][0]["transitions"][0][
        "replay_supported"
    ] is False


def test_graph_builder_receives_no_screen_change_action_without_code_pruning():
    screen = _screen("o1", "search", "1" * 16, "wanted.co.kr/search")
    after = _screen("o2", "search", "2" * 16, "wanted.co.kr/search")
    submission = _submission(
        [
            _event(
                0,
                screen,
                [_target_action(0, text="직군")],
                screen,
            ),
            _event(
                1,
                screen,
                [_target_action(1, text="SW 개발")],
                screen,
                status="unknown",
                reason="no_screen_change",
            ),
            _event(
                2,
                screen,
                [_target_action(2, text="적용")],
                after,
            ),
        ]
    )
    candidate = RecipeCandidate.from_submission(
        submission,
        run_id=submission.run_id,
        status="recorded",
    )

    payload = build_candidate_graph_payload(candidate)
    graph = build_candidate_execution_graph(candidate, builder=_graph_builder)

    assert [event["seq"] for event in payload["flat_log"]] == [0, 1, 2]
    assert payload["flat_log"][1]["transition"]["reason"] == "no_screen_change"
    assert graph.nodes[0].source_event_seqs == [0, 1, 2]

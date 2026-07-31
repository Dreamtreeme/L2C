import time

from agent.graph import (
    worker_execution_dispatch,
    worker_observation,
    worker_selection,
    worker_transition,
)
from agent.graph.worker_reflex import reflex_node
from agent.runtime.job_card_queue import replay_job_card_after_return


def _candidate_submission() -> dict:
    return {
        "run_id": "worker-contract",
        "goal": "채용공고 수집",
        "site": "wanted",
        "task_category": "검색",
        "keyword": "AI 엔지니어",
        "review_attempt": 0,
        "skill_metadata_evidence": {
            "site": "wanted",
            "task_category": "검색",
        },
        "recorded_steps": [
            {
                "seq": 0,
                "decision_capture_id": "capture:0001",
                "url_template": "wanted.co.kr/",
                "page_role": "home",
                "before_state": {
                    "capture_id": "capture:0001",
                    "url_template": "wanted.co.kr/",
                    "page_role": "home",
                    "screen_context_signature": {
                        "phash": "1" * 16,
                        "size": [1920, 1080],
                    },
                },
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
            {
                "action_seq": 0,
                "status": "ready",
                "marker_texts": ["검색어"],
                "after_state": {
                    "capture_id": "capture:0002",
                    "url_template": "wanted.co.kr/search",
                    "page_role": "search_overlay",
                    "screen_context_signature": {
                        "phash": "2" * 16,
                        "size": [1920, 1080],
                    },
                },
            }
        ],
        "feedback_episodes": [
            {
                "seq": 0,
                "proposal": {
                    "action": "click_marker",
                    "args": {"page_role": "home"},
                },
                "feedback": {"label": "success"},
                "observation": {
                    "before": {
                        "capture_id": "capture:0001",
                        "url": "https://www.wanted.co.kr/",
                        "screen_signature": {
                            "phash": "1" * 16,
                            "size": [1920, 1080],
                        },
                        "marker_texts": ["채용", "검색"],
                    },
                },
            }
        ],
    }


def test_candidate_promotion_keeps_only_safe_roi_target(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _candidate_submission()
    submission["recorded_steps"].append(
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
        }
    )
    submission["recorded_steps"].append(
        {
            "seq": 2,
            "action": "press_key",
            "replay_mode": "fixed",
            "param": {"key": "enter"},
        }
    )
    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        submission,
        review={
            "decision": "accept",
            "recipe_candidate": True,
            "confidence": 0.8,
        },
        source="test",
        submission_id="worker-contract:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        mode="promote",
        critic=lambda _payload: {
            "decision": "accept",
            "reasons": ["검색 버튼만 재사용 가능"],
            "feedback_to_worker": "",
            "step_verdicts": [
                {"seq": 0, "keep": True, "reason": "검색 버튼은 안정적"},
                {"seq": 2, "keep": False, "reason": "독립 키 입력 문맥 부족"},
            ],
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(tmp_path / "critic.db").get_by_site("wanted")
    assert review["promotion"]["promoted"] is True
    assert review["promotion"]["promoted_action_count"] == 1
    assert len(recipes) == 1
    assert (
        recipes[0]["transitions"][0]["actions"][0]["action"]
        == "click_marker"
    )


def test_critic_cannot_rewrite_autonomous_recipe_fields(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _candidate_submission()
    db_path = tmp_path / "critic-authority.db"
    candidate_id = RecipeCandidateStore(db_path).commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True},
        source="test",
        submission_id="worker-authority:0",
    )

    def critic(payload):
        assert payload["required_step_verdicts"] == [
            {"seq": 0, "action": "click_marker"}
        ]
        return {
            "decision": "accept",
            "reasons": ["원본 검색 버튼 단계 유지"],
            "step_verdicts": [{"seq": 0, "keep": True}],
            # 이전 계약의 실행 필드를 반환해도 새 스키마는 이를 폐기한다.
            "skill_metadata": {
                "task_category": "결제",
                "step_intents": [
                    {
                        "seq": 0,
                        "action": "press_key",
                        "component": "malicious_override",
                        "replay_mode": "fixed",
                    }
                ],
            },
            "transition_contracts": [
                {
                    "seq": 0,
                    "contract": {
                        "common_ready_cues": [
                            {"kind": "text_any", "values": ["임의 화면"]}
                        ]
                    },
                }
            ],
            "confidence": 0.9,
        }

    result = review_and_apply_candidate(
        candidate_id,
        db_path=db_path,
        mode="promote",
        critic=critic,
    )

    recipe = RecipeStore(db_path).get_by_site("wanted")[0]
    assert result["promotion"]["promoted"] is True
    action = recipe["transitions"][0]["actions"][0]
    assert action["action"] == "click_marker"
    assert action["component"] == "search_button"
    assert "transition_contract" not in action
    assert recipe["skill_metadata"]["task_category"] == "검색"
    assert "skill_metadata" not in result


def test_candidate_promotion_saves_consecutive_steps_as_one_path(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _candidate_submission()
    submission["recorded_steps"].append(
        {
            "seq": 1,
            "page_role": "search_results",
            "action": "click_marker",
            "component": "filter_button",
            "replay_mode": "fixed",
            "target": {"text": "직무 필터"},
            "roi_signature": {
                "phash": "1" * 16,
                "crop_rect_ratio": [0.1, 0.1, 0.4, 0.3],
            },
        }
    )
    submission["transition_records"].append(
        {
            "action_seq": 1,
            "status": "ready",
            "marker_texts": ["직무 선택"],
            "after_state": {
                "capture_id": "capture:0003",
                "url_template": "wanted.co.kr/search",
                "page_role": "search_results",
                "screen_context_signature": {
                    "phash": "3" * 16,
                    "size": [1920, 1080],
                },
            },
        }
    )
    submission["feedback_episodes"].append(
        {
            "seq": 1,
            "proposal": {
                "action": "click_marker",
                "args": {"page_role": "search_results"},
                "component_candidate": "filter_button",
            },
            "feedback": {"label": "success"},
            "observation": {
                "before": {"marker_texts": ["직무 필터"]},
            },
        }
    )
    db_path = tmp_path / "path-promotion.db"
    candidate_id = RecipeCandidateStore(db_path).commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True},
        source="test",
        submission_id="worker-path:0",
    )

    result = review_and_apply_candidate(
        candidate_id,
        db_path=db_path,
        mode="promote",
        critic=lambda _payload: {
            "decision": "accept",
            "reasons": ["두 단계가 모두 검증됨"],
            "feedback_to_worker": "",
            "step_verdicts": [
                {"seq": 0, "keep": True},
                {"seq": 1, "keep": True},
            ],
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(db_path).get_by_site("wanted")
    assert result["promotion"]["promoted_path_count"] == 1
    assert len(recipes) == 1
    assert [
        transition["seq"]
        for transition in recipes[0]["transitions"]
    ] == [0, 1]


def test_contextual_action_is_promoted_inside_recipe_path(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _candidate_submission()
    submission["skill_metadata_evidence"]["inputs"] = [
        {
            "name": "query",
            "description": "검색어",
            "required": True,
            "source": "recorded_step",
        }
    ]
    submission["recorded_steps"].extend(
        [
            {
                "seq": 1,
                "url_template": "wanted.co.kr/search",
                "page_role": "search_overlay",
                "declared_page_role": "search_overlay",
                "action": "type_in_marker",
                "replay_mode": "parameterized",
                "component": "search_input",
                "target": {"text": "검색어"},
                "value": "AI 엔지니어",
                "param": {
                    "text": "AI 엔지니어",
                    "slot_name": "query",
                },
                "slot_refs": ["query"],
                "roi_signature": {
                    "phash": "1" * 16,
                    "crop_rect_ratio": [0.1, 0.1, 0.7, 0.2],
                },
            },
            {
                "seq": 2,
                "url_template": "wanted.co.kr/search",
                "page_role": "search_overlay",
                "declared_page_role": "search_overlay",
                "action": "press_key",
                "replay_mode": "fixed",
                "param": {"key": "enter"},
                "screen_context_signature": {
                    "phash": "2" * 16,
                    "size": [1920, 1080],
                },
                "expected_after": "검색 결과가 표시된다.",
            },
        ]
    )
    submission["feedback_episodes"].extend(
        [
            {
                "seq": 1,
                "proposal": {
                    "action": "type_in_marker",
                    "args": {
                        "page_role": "search_overlay",
                        "target_component": "search_input",
                    },
                    "component_candidate": "search_input",
                },
                "feedback": {"label": "partial"},
                "observation": {
                    "before": {
                        "url": "https://www.wanted.co.kr/search",
                        "marker_texts": ["검색어"],
                    },
                    "result": {"status": "success"},
                },
            },
            {
                "seq": 2,
                "proposal": {
                    "action": "press_key",
                    "args": {
                        "key": "enter",
                        "page_role": "search_overlay",
                    },
                },
                "feedback": {"label": "partial"},
                "observation": {
                    "before": {
                        "url": "https://www.wanted.co.kr/search",
                        "marker_texts": ["AI 엔지니어"],
                    },
                    "result": {"status": "success"},
                },
            },
        ]
    )
    submission["transition_records"].extend(
        [
                {
                    "action_seq": 1,
                "action": "type_in_marker",
                "status": "ready",
                "reason": "screen_change_pixels_matched",
                "marker_count": 3,
                "visual_change_ratio": 0.08,
                    "marker_texts": ["AI 엔지니어"],
                    "after_state": {
                        "capture_id": "capture:0003",
                        "url_template": "wanted.co.kr/search",
                        "page_role": "search_overlay",
                        "screen_context_signature": {
                            "phash": "2" * 16,
                            "size": [1920, 1080],
                        },
                    },
                },
            {
                "action_seq": 2,
                "action": "press_key",
                "status": "ready",
                    "reason": "screen_change_pixels_matched",
                "marker_count": 5,
                "visual_change_ratio": 0.12,
                    "marker_texts": ["검색 결과", "AI 엔지니어"],
                    "after_state": {
                        "capture_id": "capture:0004",
                        "url_template": "wanted.co.kr/search",
                        "page_role": "search_results",
                        "screen_context_signature": {
                            "phash": "3" * 16,
                            "size": [1920, 1080],
                        },
                    },
                },
        ]
    )
    db_path = tmp_path / "followup.db"
    store = RecipeCandidateStore(db_path)
    candidate_id = store.commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True},
        source="test",
        submission_id="worker-followup:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=db_path,
        mode="promote",
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["검색 입력 직후 Enter 전환이 검증됨"],
            "feedback_to_worker": "",
            "step_verdicts": [
                {
                    "seq": item["seq"],
                    "keep": True,
                }
                for item in payload["required_step_verdicts"]
            ],
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(db_path).get_by_site("wanted")
    assert review["promotion"]["promoted_action_count"] == 3
    assert review["promotion"]["promoted_transition_count"] == 2
    assert len(recipes) == 1
    assert [
        action["action"]
        for transition in recipes[0]["transitions"]
        for action in transition["actions"]
    ] == ["click_marker", "type_in_marker", "press_key"]
    assert [
        action["action"]
        for action in recipes[0]["transitions"][1]["actions"]
    ] == ["type_in_marker", "press_key"]


def test_promotion_evidence_uses_enter_to_verify_preceding_type_action():
    from agent.recipe.promotion_policy import evaluate_candidate_step_evidence

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
                "before_state": {"capture_id": "capture:0003"},
            },
        ],
        "payload": {
            "feedback_episodes": [
                {
                    "seq": 2,
                    "feedback": {"label": "partial"},
                    "observation": {"result": {"status": "success"}},
                },
                {
                    "seq": 4,
                    "feedback": {"label": "partial"},
                    "observation": {"result": {"status": "success"}},
                },
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

    verdicts = evaluate_candidate_step_evidence(candidate)

    assert verdicts[2]["eligible"] is True
    assert verdicts[2]["execution_group_seqs"] == [2, 4]
    assert verdicts[2]["effect_verified_by_seq"] == 4
    assert verdicts[2]["evidence_mode"] == "deferred_group_effect"
    assert verdicts[4]["eligible"] is True
    assert verdicts[4]["execution_group_seqs"] == [2, 4]


def test_promotion_evidence_rejects_type_group_without_state_continuity():
    from agent.recipe.promotion_policy import evaluate_candidate_step_evidence

    candidate = {
        "steps": [
            {
                "seq": 2,
                "action": "type_in_marker",
                "param": {"slot_name": "query", "text": "iOS 개발자"},
                "before_state": {"capture_id": "capture:0002"},
            },
            {
                "seq": 4,
                "action": "press_key",
                "param": {"key": "enter"},
                "before_state": {"capture_id": "capture:0099"},
            },
        ],
        "payload": {
            "feedback_episodes": [
                {
                    "seq": 2,
                    "feedback": {"label": "partial"},
                    "observation": {"result": {"status": "success"}},
                },
                {
                    "seq": 4,
                    "feedback": {"label": "partial"},
                    "observation": {"result": {"status": "success"}},
                },
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

    verdicts = evaluate_candidate_step_evidence(candidate)

    assert verdicts[2]["eligible"] is False
    assert verdicts[2]["blocking_reasons"] == ["no_screen_change"]
    assert "execution_group_seqs" not in verdicts[2]


def test_contextual_action_after_reasoning_gap_is_not_standalone_recipe(
    tmp_path,
):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _candidate_submission()
    submission["recorded_steps"].append(
        {
            "seq": 2,
            "url_template": "wanted.co.kr/wd/{id}",
            "page_role": "job_detail",
            "declared_page_role": "job_detail",
            "action": "go_back",
            "replay_mode": "fixed",
            "param": {},
            "screen_context_signature": {
                "phash": "3" * 16,
                "size": [1920, 1080],
            },
            "expected_after": "검색 결과 목록이 표시된다.",
        }
    )
    submission["feedback_episodes"].extend(
        [
            {
                "seq": 1,
                "proposal": {
                    "action": "finish_detail_reading",
                    "args": {
                        "page_role": "job_detail",
                    },
                },
                "feedback": {"label": "success"},
                "observation": {
                    "before": {
                        "url": "https://www.wanted.co.kr/wd/123",
                        "marker_texts": ["주요업무", "자격요건"],
                    },
                    "result": {"status": "success"},
                },
            },
            {
                "seq": 2,
                "proposal": {
                    "action": "go_back",
                    "args": {"page_role": "job_detail"},
                },
                "feedback": {"label": "partial"},
                "observation": {
                    "before": {
                        "url": "https://www.wanted.co.kr/wd/123",
                        "marker_texts": ["주요업무", "자격요건"],
                    },
                    "result": {"status": "success"},
                },
            },
        ]
    )
    submission["transition_records"].append(
        {
            "action_seq": 2,
            "action": "go_back",
            "source": "autonomous",
            "status": "ready",
            "reason": "queue_return_phash_match",
            "marker_count": 2,
            "marker_texts": ["검색 결과", "iOS 개발자"],
            "ocr_skipped": True,
        }
    )
    db_path = tmp_path / "go-back.db"
    candidate_id = RecipeCandidateStore(db_path).commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True},
        source="test",
        submission_id="worker-go-back:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=db_path,
        mode="promote",
        critic=lambda payload: {
            "decision": "accept",
            "reasons": ["상세 완료 뒤 목록 복귀가 검증됨"],
            "feedback_to_worker": "",
            "step_verdicts": [
                {
                    "seq": item["seq"],
                    "keep": True,
                }
                for item in payload["required_step_verdicts"]
            ],
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(db_path).get_by_site("wanted")
    assert review["promotion"]["promoted_action_count"] == 1
    assert len(recipes) == 1
    assert [
        transition["actions"][0]["source_seq"]
        for transition in recipes[0]["transitions"]
    ] == [0]


def test_candidate_promotion_blocks_no_effect_step(tmp_path):
    from agent.recipe.candidate_reviewer import review_and_apply_candidate
    from agent.recipe.candidate_store import RecipeCandidateStore
    from agent.recipe.store import RecipeStore

    submission = _candidate_submission()
    submission["feedback_episodes"][0]["feedback"] = {
        "label": "no_effect",
        "reason": "screen_unchanged",
    }
    store = RecipeCandidateStore(tmp_path / "critic.db")
    candidate_id = store.commit_candidate(
        submission,
        review={"decision": "accept", "recipe_candidate": True},
        source="test",
        submission_id="worker-no-effect:0",
    )

    review = review_and_apply_candidate(
        candidate_id,
        db_path=tmp_path / "critic.db",
        mode="promote",
        critic=lambda _payload: {
            "decision": "accept",
            "reasons": ["모델은 승인했지만 실행 증거가 실패임"],
            "feedback_to_worker": "",
            "step_verdicts": [
                {"seq": 0, "keep": True},
            ],
            "confidence": 0.9,
        },
    )

    assert review["promotion"]["promoted"] is False
    assert review["promotion"]["skipped_steps"][0]["reason"] == "feedback_no_effect"
    assert RecipeStore(tmp_path / "critic.db").get_by_site("wanted") == []


def test_promotion_worker_stops_after_bounded_critic_failures(tmp_path, monkeypatch):
    from agent.application import recipe_promotion_service
    from agent.application.recipe_promotion_worker import RecipePromotionWorker
    from agent.recipe import candidate_reviewer
    from agent.recipe.candidate_store import RecipeCandidateStore

    db_path = tmp_path / "promotion.db"
    store = RecipeCandidateStore(db_path)
    candidate_id = store.commit_candidate(
        _candidate_submission(),
        review={"decision": "accept", "recipe_candidate": True},
        source="test",
        submission_id="worker-retry:0",
    )
    monkeypatch.setenv("VISION_RECIPE_AUTO_PROMOTE", "1")
    assert recipe_promotion_service.schedule_recipe_candidate_promotion(
        candidate_id,
        db_path=db_path,
    )
    monkeypatch.setattr(
        candidate_reviewer,
        "review_and_apply_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            TimeoutError("critic timeout")
        ),
    )
    worker = RecipePromotionWorker(
        db_path,
        retry_delay_sec=0,
        max_attempts=2,
    )

    assert worker.process_one()["status"] == "pending_review"
    assert worker.process_one()["status"] == "review_failed"
    failed = store.get_candidate(candidate_id)
    assert failed["review_attempts"] == 2
    assert "critic timeout" in failed["review_error"]

import time

from agent.graph import worker_execution, worker_observation, worker_transition
from agent.runtime.reflex_runtime import reflex_node
from agent.runtime.result_card_queue import queue_replay_after_return


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
                "page_role": "home",
                "action": "click_marker",
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
        "transition_observations": [
            {
                "action_seq": 0,
                "status": "ready",
                "marker_texts": ["검색어"],
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
                    "before": {"marker_texts": ["채용", "검색"]},
                },
            }
        ],
    }


def test_roi_record_and_replay_uses_target_crop(tmp_path):
    from PIL import Image, ImageDraw

    from agent.recipe.phash_replay import match_step_by_screen_signature
    from agent.recipe.record import record_ui_step

    saved = tmp_path / "saved.png"
    current = tmp_path / "current.png"
    for path in (saved, current):
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([150, 20, 170, 40], fill="black")
        if path == current:
            draw.rectangle([0, 120, 200, 200], fill="black")
        image.save(path)

    steps: list[dict] = []
    record_ui_step(
        steps,
        {
            "goal": "검색",
            "current_url": "https://www.wanted.co.kr",
            "current_page_role": "home",
            "screen_signature": {"phash": "f" * 16, "size": [200, 200]},
            "recent_images": [saved],
            "current_markers": [
                {"id": 1, "bbox": [150, 20, 170, 40], "text": "검색"},
            ],
        },
        "click_marker",
        {
            "marker_id": 1,
            "target_role": "button",
            "target_component": "search_button",
        },
        0,
    )

    marker_id, trace = match_step_by_screen_signature(
        steps[0],
        {"phash": "0" * 16, "size": [200, 200]},
        [{"id": 7, "bbox": [150, 20, 170, 40], "text": "검색"}],
        current_image_path=str(current),
    )

    assert steps[0]["page_role"] == "home"
    assert steps[0]["roi_signature"]["algorithm"] == "roi-phash-dct64-v2"
    assert marker_id == 7
    assert trace["matched"] is True
    assert trace["mode"] == "roi_phash"


def test_roi_replay_rejects_step_without_roi_signature():
    from agent.recipe.phash_replay import match_step_by_screen_signature

    marker_id, trace = match_step_by_screen_signature(
        {
            "screen_signature": {"phash": "0" * 16, "size": [1000, 1000]},
            "target": {
                "text": "검색",
                "bbox_ratio": [0.79, 0.08, 0.83, 0.12],
                "center_ratio": [0.81, 0.1],
            },
        },
        {"phash": "0" * 16, "size": [1000, 1000]},
        [{"id": 3, "bbox": [790, 80, 830, 120], "text": "검색"}],
    )

    assert marker_id is None
    assert trace["reason"] == "roi_signature_missing"


def test_no_effect_reuses_ocr_only_for_matching_capture(monkeypatch, tmp_path):
    from PIL import Image

    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (800, 600), "white").save(screenshot)

    class FakePerception:
        def capture_screen(self):
            return screenshot

        def analyze_ui(self, _path):
            raise AssertionError("같은 화면에서는 OCR을 다시 실행하면 안 됩니다.")

    monkeypatch.setattr(
        worker_observation,
        "_perception_engine",
        lambda: FakePerception(),
    )
    monkeypatch.setattr(
        worker_observation,
        "raw_screen_phash_signature",
        lambda _path: {"phash": "0" * 16, "size": [800, 600]},
    )

    result = worker_observation.observe_screen_cycle(
        {
            "worker_run_id": "worker-no-effect",
            "worker_attempt_index": 0,
            "current_capture_id": "worker-no-effect:attempt:00:capture:0004",
            "capture_sequence": 4,
            "current_screenshot": str(screenshot),
            "current_url": "https://example.com/jobs",
            "current_url_stale": False,
            "current_markers": [
                {"id": 1, "bbox": [10, 20, 200, 60], "text": "검색"},
            ],
            "ui_context": "검색",
            "marked_image": str(screenshot),
            "screen_signature": {"phash": "0" * 16, "size": [800, 600]},
            "current_page_role": "search",
            "analysis_mode": "full",
            "ocr_complete": True,
            "reflex_blocked_recipe_keys": [],
            "pending_transition": {
                "action": "click_marker",
                "action_seq": 3,
                "from_capture_id": "worker-no-effect:attempt:00:capture:0004",
                "source": "reflex",
                "recipe_key": "roi#search",
                "before_url": "https://example.com/jobs",
                "before_phash": "0" * 16,
                "before_screenshot": str(screenshot),
                "started_at": time.time(),
                "contract": {},
            },
        }
    )

    assert result["transition_reason"] == "reflex_no_screen_change"
    assert result["ocr_complete"] is True
    assert result["current_markers"][0]["id"] == 1
    assert (
        result["previous_screen_observation"]["capture_id"]
        == "worker-no-effect:attempt:00:capture:0005"
    )

    monkeypatch.setattr(
        worker_transition,
        "transition_has_visual_change",
        lambda _pending, _path: (False, 0.0),
    )
    monkeypatch.setattr(
        worker_transition,
        "transition_no_effect_by_phash",
        lambda _pending, _url, _signature: (True, 0),
    )
    stale = worker_transition.evaluate_transition_node(
        {
            "current_capture_id": "worker-test:capture:0003",
            "current_screenshot": str(screenshot),
            "current_url": "https://example.com/jobs",
            "raw_screen_signature": {"phash": "0" * 16, "size": [800, 600]},
            "ocr_complete": False,
            "current_markers": [],
            "previous_screen_observation": {
                "capture_id": "worker-test:capture:0001",
                "screenshot": str(screenshot),
                "markers": [{"id": 4, "bbox": [10, 20, 30, 40]}],
            },
            "pending_transition": {
                "action": "click_marker",
                "from_capture_id": "worker-test:capture:0002",
                "source": "autonomous",
                "before_url": "https://example.com/jobs",
                "before_phash": "0" * 16,
                "before_screenshot": str(screenshot),
                "started_at": time.time(),
            },
        }
    )

    assert stale.get("ocr_complete") is None
    assert stale["transition_observations"][0]["marker_count"] == 0


def test_result_queue_replays_cached_card_after_return():
    state = {
        "result_card_queue": [
            {
                "queue_id": "card-2",
                "status": "pending",
                "title": "두 번째 iOS 개발자",
                "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                "center_ratio": [0.4, 0.425],
                "target": {
                    "text": "두 번째 iOS 개발자",
                    "bbox_ratio": [0.3, 0.4, 0.5, 0.45],
                    "center_ratio": [0.4, 0.425],
                },
            }
        ],
        "result_page_memory": {
            "screen_signature": {
                "phash": "0" * 16,
                "size": [1000, 1000],
                "anchors": ["두 번째 iOS 개발자"],
            },
        },
    }

    request, markers, trace = queue_replay_after_return(
        state,
        {"action": "go_back"},
        "https://www.wanted.co.kr/search?query=ios",
        [],
        {
            "phash": "0" * 16,
            "size": [1000, 1000],
            "anchors": ["두 번째 iOS 개발자"],
        },
    )

    assert request is not None
    assert trace["hit"] is True
    assert request.tool_calls[0].name == "click_marker"
    assert markers[0]["bbox"] == [300, 400, 500, 450]


def test_recipe_store_scopes_by_site_and_task_category(tmp_path):
    from agent.recipe.store import RecipeStore

    store = RecipeStore(tmp_path / "recipes.db")
    store.commit_recipe(
        "wanted",
        "검색",
        [
            {
                "seq": 0,
                "page_role": "home",
                "action": "click_marker",
                "target": {"text": "검색", "center_ratio": [0.8, 0.1]},
                "roi_signature": {
                    "phash": "0" * 16,
                    "target_center_ratio": [0.8, 0.1],
                },
            }
        ],
        metadata={"task_category": "검색"},
    )

    assert len(store.get_site_recipes("wanted", task_category="검색")) == 1
    assert store.get_site_recipes("wanted", task_category="로그인") == []
    assert store.get_site_recipes("saramin", task_category="검색") == []


def test_reflex_replays_one_parameterized_roi_step(monkeypatch, tmp_path):
    from PIL import Image, ImageDraw

    from agent.vision.screen_signature import compute_target_roi_signature
    from shared.schema.recipe_schema import RecipeStep, SiteRecipe

    screenshot = tmp_path / "screen.png"
    image = Image.new("RGB", (200, 120), "white")
    ImageDraw.Draw(image).rectangle([10, 10, 70, 40], fill="black")
    image.save(screenshot)
    roi_signature = compute_target_roi_signature(
        screenshot,
        [10, 10, 70, 40],
        [200, 120],
    )

    class FakeStore:
        def get_site_recipes(self, site, task_category=None):
            assert site == "wanted"
            return [
                (
                    "recipe-search",
                    SiteRecipe(
                        site="wanted",
                        goal="검색",
                        steps=[
                            RecipeStep(
                                seq=0,
                                action="type_in_marker",
                                page_role="home",
                                replay_mode="parameterized",
                                roi_signature=roi_signature,
                                target={
                                    "text": "검색",
                                    "bbox_ratio": [0.05, 0.0833, 0.35, 0.3333],
                                    "center_ratio": [0.2, 0.2083],
                                },
                                param={"slot_name": "query"},
                                slot_refs=["query"],
                            )
                        ],
                    ),
                )
            ]

    monkeypatch.setattr("agent.recipe.store.RecipeStore", lambda: FakeStore())
    result = reflex_node(
        {
            "goal": "AI 엔지니어 공고",
            "current_url": "https://www.wanted.co.kr",
            "current_page_role": "home",
            "screen_signature": {"size": [200, 120]},
            "recent_images": [screenshot],
            "current_markers": [
                {"id": 7, "bbox": [10, 10, 70, 40], "text": "검색"},
            ],
            "recipe_params": {
                "site": "wanted",
                "task_category": "검색",
                "query": "AI 엔지니어",
            },
        }
    )

    call = result["pending_action"].tool_calls[0]
    assert result["reflex_trace"]["hit"] is True
    assert call.name == "type_in_marker"
    assert call.args["marker_id"] == 7
    assert call.args["text"] == "AI 엔지니어"
    assert len(result["pending_action"].tool_calls) == 1


def test_detail_finish_extracts_once_and_clears_buffer(monkeypatch):
    monkeypatch.setattr(
        worker_execution,
        "_extract_job_from_detail_ocr_buffer",
        lambda _state, current_url: {
            "company_name": "보이저엑스",
            "position": "iOS 개발자",
            "url": current_url,
            "requirements": ["Swift"],
        },
    )

    result, extracted = worker_execution._dispatch_state(
        "finish_detail_reading",
        {"page_role": "job_detail", "detail_complete": True},
        {},
        current_url="https://www.wanted.co.kr/wd/1",
        state={
            "detail_ocr_buffer": {
                "url": "https://www.wanted.co.kr/wd/1",
                "lines": [{"text": "자격요건 Swift"}],
            }
        },
    )

    assert result["status"] == "success"
    assert result["_detail_ocr_buffer"] == {}
    assert extracted["공고목록"][0]["position"] == "iOS 개발자"


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
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "task_category": "검색",
                "step_intents": [
                    {"seq": 0, "action": "click_marker", "replay_mode": "fixed"},
                    {"seq": 1, "action": "click_marker", "replay_mode": "reasoning"},
                    {"seq": 2, "action": "press_key", "replay_mode": "fixed"},
                ],
            },
            "confidence": 0.9,
        },
    )

    recipes = RecipeStore(tmp_path / "critic.db").get_by_site("wanted")
    assert review["promotion"]["promoted"] is True
    assert review["promotion"]["promoted_step_count"] == 1
    assert len(recipes) == 1
    assert recipes[0]["steps"][0]["action"] == "click_marker"


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
            "promote_to_active_recipe": True,
            "skill_metadata": {
                "site": "wanted",
                "task_category": "검색",
                "step_intents": [
                    {"seq": 0, "action": "click_marker", "replay_mode": "fixed"},
                ],
            },
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

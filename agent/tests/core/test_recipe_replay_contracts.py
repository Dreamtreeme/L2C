from PIL import Image, ImageDraw

from agent.recipe.replay import ReflexReplayContext, ReplayInputs, select_reflex_replay
from agent.recipe.replay_runtime import (
    replay_session_after_transition,
    verify_replay_after_state,
)
from agent.tests.worker_test_support import worker_state
from shared.schema.experience_rule_schema import (
    ExpectedEffect,
    ExperienceRule,
    ExperienceRuleStep,
    ReplaySession,
    RuleAction,
    RuleScreen,
    RuleTarget,
)


def _rule():
    return ExperienceRule(
        site="wanted",
        goal="채용공고 검색",
        skill_metadata={"task_category": "검색"},
        steps=[
            ExperienceRuleStep(
                step_id="step-1",
                source_transition_seqs=[3],
                before=RuleScreen(
                    url_template="example.com/search",
                    page_role="search",
                    reference_signature={"phash": "1" * 16, "size": [1000, 800]},
                ),
                actions=[
                    RuleAction(
                        source_seq=3,
                        action="click_marker",
                        target=RuleTarget(
                            reference={
                                "text": "검색",
                                "marker_type": "text",
                                "bbox_ratio": [0.7, 0.1, 0.9, 0.2],
                                "center_ratio": [0.8, 0.15],
                            },
                            reference_roi_signature={
                                "phash": "a" * 16,
                                "crop_rect_ratio": [0.65, 0.05, 0.95, 0.25],
                            },
                        ),
                    )
                ],
                intent="검색을 실행한다",
                expected_effect=ExpectedEffect(
                    kind="screen_change",
                    description="검색 결과가 보인다",
                    expected_url_template="example.com/search",
                    expected_page_role="search",
                ),
            )
        ],
    )


def _marker(marker_id=7, text="검색"):
    return {
        "id": marker_id,
        "bbox": [700, 80, 900, 160],
        "text": text,
        "type": "text",
    }


def _context(
    marker=None,
    *,
    url="https://example.com/search",
    page_role="search",
):
    return ReflexReplayContext(
        markers=[marker] if marker is not None else [],
        inputs=ReplayInputs(),
        task_category="검색",
        site="wanted",
        current_image_path="screen.png",
        current_page_role=page_role,
        current_url=url,
        current_url_template="example.com/search",
        observation_id="observation:1",
        screen_size=[1000, 800],
        blocked_rule_keys=set(),
        used_rule_keys=set(),
        replay_session=None,
        rule_candidates=[("experience-rule10#test", _rule())],
    )


def _state():
    return worker_state(
        observation={
            "observation_id": "observation:1",
            "current_url": "https://example.com/search",
            "current_page_role": "search",
            "current_screenshot": "screen.png",
            "screen_signature": {"phash": "1" * 16, "size": [1000, 800]},
        }
    )


def test_roi_match_replays_saved_coordinate_without_ocr_markers(monkeypatch):
    monkeypatch.setattr(
        "agent.recipe.replay.roi_signature_match",
        lambda *_args, **_kwargs: {"matched": True, "mode": "roi_phash"},
    )

    selection, _log = select_reflex_replay(
        _state(),
        _context(page_role=""),
    )

    assert selection is not None
    assert selection.resolution_mode == "saved_coordinate"
    assert selection.tool_calls[0]["args"]["marker_id"] == 0
    assert selection.markers == [
        {
            "id": 0,
            "bbox": [700, 80, 900, 160],
            "text": "검색",
            "type": "text",
        }
    ]


def test_roi_phash_mismatch_falls_back_to_full_perception(monkeypatch):
    monkeypatch.setattr(
        "agent.recipe.replay.roi_signature_match",
        lambda *_args, **_kwargs: {
            "matched": False,
            "reason": "roi_phash_distance",
        },
    )

    selection, log = select_reflex_replay(
        _state(),
        _context(),
    )

    assert selection is None
    assert log.last_reason == "roi_phash_distance"


def test_wrong_url_rejects_rule_before_roi_comparison(monkeypatch):
    monkeypatch.setattr(
        "agent.recipe.replay.roi_signature_match",
        lambda *_args, **_kwargs: {"matched": True},
    )

    selection, log = select_reflex_replay(
        _state(),
        _context(url="https://example.com/login"),
    )

    assert selection is None
    assert log.last_reason == "url_scope_mismatch"


def test_target_region_effect_ignores_changes_outside_region(tmp_path):
    before = tmp_path / "before.png"
    outside = tmp_path / "outside.png"
    inside = tmp_path / "inside.png"
    Image.new("RGB", (200, 100), "white").save(before)
    for path, box in ((outside, [0, 0, 80, 100]), (inside, [120, 0, 200, 100])):
        image = Image.new("RGB", (200, 100), "white")
        ImageDraw.Draw(image).rectangle(box, fill="black")
        image.save(path)

    request = {
        "source": "reflex",
        "before_screenshot": str(before),
        "before_url": "https://example.com/search",
        "before_page_role": "search",
        "expected_effect": ExpectedEffect(
            kind="target_region_change",
            description="오른쪽 결과 패널이 바뀐다",
            expected_url_template="example.com/search",
            expected_page_role="search",
            target_region_ratio=[0.5, 0.0, 1.0, 1.0],
        ),
    }

    def state_for(path):
        return worker_state(
            observation={
                "current_screenshot": str(path),
                "current_url": "https://example.com/search",
                "current_page_role": "search",
            }
        )

    outside_match = verify_replay_after_state(request, state_for(outside))
    inside_match = verify_replay_after_state(request, state_for(inside))
    assert outside_match[0] is False
    assert inside_match[0] is True


def test_successful_rule_step_advances_session():
    session = ReplaySession(
        recipe_key="experience-rule10#test",
        current_step_index=0,
        pending_step_index=0,
        step_count=2,
    )
    state = worker_state(replay={"replay_session": session})

    advanced = replay_session_after_transition(state, source="reflex", status="ready")

    assert advanced is not None
    assert advanced.current_step_index == 1
    assert advanced.pending_step_index is None


def test_scroll_step_uses_its_saved_roi_and_coordinate(monkeypatch):
    monkeypatch.setattr(
        "agent.recipe.replay.roi_signature_match",
        lambda *_args, **_kwargs: {"matched": True, "mode": "roi_phash"},
    )
    base = _rule()
    scroll_target = RuleTarget(
        reference={
            "text": "상세",
            "marker_type": "text",
            "bbox_ratio": [0.6, 0.1, 0.95, 0.9],
            "center_ratio": [0.8, 0.5],
        },
        reference_roi_signature={
            "phash": "b" * 16,
            "crop_rect_ratio": [0.55, 0.05, 1.0, 0.95],
        },
    )
    step = ExperienceRuleStep(
        step_id="step-2",
        source_transition_seqs=[4],
        before=base.steps[0].before,
        actions=[
            RuleAction(
                source_seq=4,
                action="scroll",
                target=scroll_target,
                param={"direction": "down", "amount": "page"},
            )
        ],
        intent="상세 패널을 더 읽는다",
        expected_effect=ExpectedEffect(
            kind="target_region_change",
            description="오른쪽 상세 본문이 이동한다",
            expected_url_template="example.com/search",
            expected_page_role="search",
            target_region_ratio=[0.55, 0.05, 1.0, 0.95],
        ),
    )
    rule = base.model_copy(update={"steps": [base.steps[0], step]})
    session = ReplaySession(
        recipe_key="experience-rule10#scroll",
        current_step_index=1,
        pending_step_index=None,
        step_count=2,
    )
    context = _context(_marker())
    context = ReflexReplayContext(
        **{
            **context.__dict__,
            "markers": [],
            "replay_session": session,
            "rule_candidates": [("experience-rule10#scroll", rule)],
        }
    )

    selection, _log = select_reflex_replay(_state(), context)

    assert selection is not None
    assert selection.tool_calls[0]["name"] == "scroll"
    assert selection.tool_call_traces[next(iter(selection.tool_call_traces))][
        "match_mode"
    ] == "saved_target_coordinate"

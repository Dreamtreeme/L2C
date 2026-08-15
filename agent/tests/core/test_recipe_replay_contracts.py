from PIL import Image, ImageDraw

from agent.application.experience_rule_resolver import resolve_rule_targets
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
    InteractionRegionHandle,
    ReplaySession,
    RuleAction,
    RuleApplication,
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
                            description="검색 버튼",
                            role="button",
                            component="search_panel",
                            spatial_relation="검색 입력창 오른쪽",
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
                applicable_when="검색 버튼이 하나 보인다",
                decline_when="검색 버튼이 여러 개다",
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


def _context(marker, *, url="https://example.com/search"):
    return ReflexReplayContext(
        markers=[marker],
        inputs=ReplayInputs(),
        task_category="검색",
        site="wanted",
        current_image_path="screen.png",
        current_page_role="search",
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


def test_exact_roi_and_marker_match_skips_semantic_resolver(monkeypatch):
    monkeypatch.setattr(
        "agent.recipe.replay.roi_signature_match",
        lambda *_args, **_kwargs: {"matched": True, "mode": "roi_phash"},
    )

    def resolver(*_args):
        raise AssertionError("정확히 일치한 대상은 모델에 다시 물으면 안 됩니다.")

    selection, _log = select_reflex_replay(
        _state(),
        _context(_marker()),
        lambda _action: [],
        resolver,
    )

    assert selection is not None
    assert selection.resolution_mode == "exact"
    assert selection.tool_calls[0]["args"]["marker_id"] == 7


def test_ocr_text_change_uses_semantic_resolver_once(monkeypatch):
    monkeypatch.setattr(
        "agent.recipe.replay.roi_signature_match",
        lambda *_args, **_kwargs: {"matched": False, "reason": "roi_phash_distance"},
    )
    calls = []

    def resolver(_step, _markers, _image_path):
        calls.append(1)
        return RuleApplication(
            decision="apply",
            reason="같은 검색 패널의 버튼",
            target_bindings=[{"source_action_seq": 3, "marker_id": 9}],
        )

    selection, _log = select_reflex_replay(
        _state(),
        _context(_marker(9, "돋보기")),
        lambda _action: [],
        resolver,
    )

    assert selection is not None
    assert selection.resolution_mode == "semantic"
    assert len(calls) == 1
    assert selection.tool_calls[0]["args"]["marker_id"] == 9


def test_wrong_url_rejects_rule_without_model_call(monkeypatch):
    monkeypatch.setattr(
        "agent.recipe.replay.roi_signature_match",
        lambda *_args, **_kwargs: {"matched": True},
    )

    def resolver(*_args):
        raise AssertionError("URL 범위가 다르면 의미 판단을 호출하면 안 됩니다.")

    selection, log = select_reflex_replay(
        _state(),
        _context(_marker(), url="https://example.com/login"),
        lambda _action: [],
        resolver,
    )

    assert selection is None
    assert log.last_reason == "url_scope_mismatch"


def test_resolver_accepts_only_current_allowed_marker_ids():
    step = _rule().steps[0]
    application = resolve_rule_targets(
        step,
        [_marker(4)],
        "missing.png",
        resolver=lambda _payload: {
            "decision": "apply",
            "reason": "검색 버튼",
            "target_bindings": [{"source_action_seq": 3, "marker_id": 4}],
        },
    )
    assert application.target_bindings[0].marker_id == 4

    try:
        resolve_rule_targets(
            step,
            [_marker(4)],
            "missing.png",
            resolver=lambda _payload: {
                "decision": "apply",
                "reason": "허용되지 않은 ID",
                "target_bindings": [{"source_action_seq": 3, "marker_id": 99}],
            },
        )
    except ValueError as exc:
        assert "allowed_markers" in str(exc)
    else:
        raise AssertionError("현재 화면에 없는 마커 ID를 허용했습니다.")


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


def test_scroll_step_reuses_session_interaction_point_without_ocr_target():
    base = _rule()
    scroll_target = RuleTarget(
        description="오른쪽 상세 패널",
        role="detail_panel",
        component="job_detail_panel",
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
        applicable_when="오른쪽 상세 패널이 열려 있다",
        decline_when="상세 패널이 닫혀 있다",
        expected_effect=ExpectedEffect(
            kind="target_region_change",
            description="오른쪽 상세 본문이 이동한다",
            expected_url_template="example.com/search",
            expected_page_role="search",
            target_region_ratio=[0.55, 0.05, 1.0, 0.95],
        ),
    )
    rule = base.model_copy(update={"steps": [base.steps[0], step]})
    key = "job_detail_panel|detail_panel|오른쪽 상세 패널"
    session = ReplaySession(
        recipe_key="experience-rule10#scroll",
        current_step_index=1,
        pending_step_index=None,
        step_count=2,
        interaction_handles={
            key: InteractionRegionHandle(
                marker_id=8,
                center_ratio=[0.8, 0.5],
                bbox_ratio=[0.79, 0.49, 0.81, 0.51],
                effect_region_ratio=[0.55, 0.05, 1.0, 0.95],
            )
        },
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

    def unexpected(*_args):
        raise AssertionError("저장된 물리 지점은 OCR이나 모델로 다시 찾으면 안 됩니다.")

    selection, _log = select_reflex_replay(
        _state(),
        context,
        unexpected,
        unexpected,
    )

    assert selection is not None
    assert selection.tool_calls[0]["name"] == "scroll"
    assert selection.tool_call_traces[next(iter(selection.tool_call_traces))][
        "match_mode"
    ] == "session_interaction_point"

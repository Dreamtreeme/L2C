from pathlib import Path


def _state(image_path: Path) -> dict:
    return {
        "current_page_role": "search",
        "current_url": "https://example.com/search?q=ios",
        "marked_image": str(image_path),
        "current_markers": [
            {"id": 10, "type": "text", "text": "iOS 개발자", "bbox": [10, 10, 80, 30]},
            {"id": 20, "type": "text", "text": "백엔드 개발자", "bbox": [10, 50, 90, 70]},
            {"id": 30, "type": "text", "text": "합격보상금", "bbox": [100, 10, 150, 30]},
        ],
        "recipe_params": {"query": "iOS 개발자", "target_count": 2},
        "result_card_queue": [],
        "active_result_card": {},
        "extracted_jd": {},
    }


def test_result_card_selector_builds_queue_and_first_click(tmp_path, monkeypatch):
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.ResultCardSelection(
                is_result_list=True,
                cards=[
                    selector.VisibleResultCard(marker_id=10, title="iOS 개발자", company="회사 A"),
                    selector.VisibleResultCard(marker_id=20, title="백엔드 개발자", company="회사 B"),
                    selector.VisibleResultCard(marker_id=999, title="없는 카드", company="회사 C"),
                ],
            )

    monkeypatch.setattr(selector, "_get_result_card_selector_model", lambda: FakeModel())

    message, trace = selector.select_result_cards(_state(image_path))

    assert trace["reason"] == "cards_selected"
    assert trace["marker_ids"] == [10, 20]
    assert [call["name"] for call in message.tool_calls] == ["set_result_card_queue", "click_marker"]
    assert message.tool_calls[0]["args"]["cards"][0]["company"] == "회사 A"
    assert message.tool_calls[1]["args"]["marker_id"] == 10


def test_result_card_selector_carries_generic_result_count_evidence(tmp_path, monkeypatch):
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.ResultCardSelection(
                is_result_list=True,
                available_result_count=4,
                count_evidence="검색 결과 4개",
                count_confidence=0.96,
                cards=[selector.VisibleResultCard(marker_id=10, title="iOS 개발자")],
            )

    monkeypatch.setattr(selector, "_get_result_card_selector_model", lambda: FakeModel())

    message, trace = selector.select_result_cards(_state(image_path))

    assert trace["available_result_count"] == 4
    assert trace["count_evidence"] == "검색 결과 4개"
    assert message.tool_calls[0]["args"]["available_result_count"] == 4


def test_result_card_selector_ignores_low_confidence_result_count(tmp_path, monkeypatch):
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.ResultCardSelection(
                is_result_list=True,
                available_result_count=4,
                count_evidence="숫자 4",
                count_confidence=0.4,
                cards=[selector.VisibleResultCard(marker_id=10, title="iOS 개발자")],
            )

    monkeypatch.setattr(selector, "_get_result_card_selector_model", lambda: FakeModel())

    message, trace = selector.select_result_cards(_state(image_path))

    assert "available_result_count" not in trace
    assert "available_result_count" not in message.tool_calls[0]["args"]


def test_result_count_hint_is_presented_as_current_condition_context():
    from agent.graph import nodes

    context = nodes._compact_result_availability_context(
        {
            "result_availability": {
                "available_result_count": 4,
                "count_evidence": "전체 채용 4건",
                "count_confidence": 0.95,
            }
        }
    )

    assert "현재 검색 조건의 전체 결과 수: 4" in context
    assert "사이트 전체의 최대치가 아닙니다" in context
    assert "finish_task로 부분 완료" in context


def test_result_card_selector_handles_visible_all_without_fixed_count(tmp_path, monkeypatch):
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    state = _state(image_path)
    state["recipe_params"].update({"target_count": 0, "count_mode": "visible_all"})

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.ResultCardSelection(
                is_result_list=True,
                cards=[
                    selector.VisibleResultCard(marker_id=10, title="iOS 개발자", company="회사 A"),
                    selector.VisibleResultCard(marker_id=20, title="백엔드 개발자", company="회사 B"),
                ],
            )

    monkeypatch.setattr(selector, "_get_result_card_selector_model", lambda: FakeModel())

    message, trace = selector.select_result_cards(state)

    assert trace["reason"] == "cards_selected"
    assert trace["card_count"] == 2
    assert [call["name"] for call in message.tool_calls] == ["set_result_card_queue", "click_marker"]


def test_result_card_selector_handles_visible_all_enum(tmp_path):
    from PIL import Image
    from agent.runtime import result_card_selector as selector
    from shared.schema.collection_intent import CollectionCountMode

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    state = _state(image_path)
    state["recipe_params"].update(
        {"target_count": 0, "count_mode": CollectionCountMode.VISIBLE_ALL}
    )

    assert selector.should_select_result_cards(state) is True


def test_result_card_selector_skips_unspecified_zero_count(tmp_path):
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    state = _state(image_path)
    state["recipe_params"].update({"target_count": 0, "count_mode": "unspecified"})

    assert selector.should_select_result_cards(state) is False


def test_result_card_selector_skips_non_result_screen(tmp_path, monkeypatch):
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.ResultCardSelection(is_result_list=False, cards=[])

    monkeypatch.setattr(selector, "_get_result_card_selector_model", lambda: FakeModel())

    message, trace = selector.select_result_cards(_state(image_path))

    assert message is None
    assert trace == {"attempted": True, "reason": "not_result_list"}


def test_result_card_selector_requests_refinement_instead_of_adjacent_role(tmp_path, monkeypatch):
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.ResultCardSelection(
                is_result_list=True,
                needs_refinement=True,
                refinement_reason="정확한 iOS 공고가 한 개뿐이고 기술스택 필터가 보입니다.",
                cards=[],
            )

    monkeypatch.setattr(selector, "_get_result_card_selector_model", lambda: FakeModel())

    message, trace = selector.select_result_cards(_state(image_path))

    assert message is None
    assert trace == {
        "attempted": True,
        "reason": "result_refinement_needed",
        "refinement_reason": "정확한 iOS 공고가 한 개뿐이고 기술스택 필터가 보입니다.",
    }


def test_result_card_selector_clicks_model_selected_filter(tmp_path, monkeypatch):
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    state = _state(image_path)
    state["current_markers"].append(
        {"id": 40, "type": "text", "text": "기술스택", "bbox": [100, 50, 160, 70]}
    )

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.ResultCardSelection(
                is_result_list=True,
                needs_refinement=True,
                refinement_reason="정확한 iOS 공고가 부족해 기술 필터를 엽니다.",
                refinement_action="click",
                refinement_marker_id=40,
                refinement_label="기술스택",
                cards=[],
            )

    monkeypatch.setattr(selector, "_get_result_card_selector_model", lambda: FakeModel())

    message, trace = selector.select_result_cards(state)

    assert trace["reason"] == "result_refinement_action"
    assert trace["marker_id"] == 40
    assert [call["name"] for call in message.tool_calls] == ["click_marker"]
    assert message.tool_calls[0]["args"]["target_component"] == "result_filter"
    assert message.tool_calls[0]["args"]["target_label"] == "기술스택"


def test_result_card_selector_types_filter_query_without_general_reasoning(tmp_path, monkeypatch):
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    state = _state(image_path)
    state["current_markers"].append(
        {"id": 57, "type": "text", "text": "기술스택 검색", "bbox": [100, 50, 160, 70]}
    )

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.ResultCardSelection(
                is_result_list=True,
                needs_refinement=True,
                refinement_reason="iOS 기술 옵션을 찾습니다.",
                refinement_action="type",
                refinement_marker_id=57,
                refinement_label="기술스택 검색",
                refinement_text="iOS",
                cards=[],
            )

    monkeypatch.setattr(selector, "_get_result_card_selector_model", lambda: FakeModel())

    message, trace = selector.select_result_cards(state)

    assert trace["action"] == "type_in_marker"
    assert [call["name"] for call in message.tool_calls] == ["type_in_marker"]
    assert message.tool_calls[0]["args"]["text"] == "iOS"
    assert message.tool_calls[0]["args"]["slot_name"] == "result_filter_query"


def test_result_card_selector_rejects_marker_when_model_label_disagrees_with_ocr(tmp_path, monkeypatch):
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    state = _state(image_path)
    state["current_markers"].append(
        {"id": 54, "type": "text", "text": "shoplive", "bbox": [100, 50, 160, 70]}
    )

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.ResultCardSelection(
                is_result_list=True,
                needs_refinement=True,
                refinement_reason="iOS 기술 옵션을 찾습니다.",
                refinement_action="type",
                refinement_marker_id=54,
                refinement_label="기술스택 검색창",
                refinement_text="iOS",
                cards=[],
            )

    monkeypatch.setattr(selector, "_get_result_card_selector_model", lambda: FakeModel())

    message, trace = selector.select_result_cards(state)

    assert message is None
    assert trace["reason"] == "result_refinement_needed"


def test_result_card_selector_types_into_wide_marker_without_ocr_text(tmp_path, monkeypatch):
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    state = _state(image_path)
    state["current_markers"].append(
        {"id": 54, "type": "icon", "text": "", "bbox": [20, 20, 180, 45]}
    )

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.ResultCardSelection(
                is_result_list=True,
                needs_refinement=True,
                refinement_reason="필터 검색창에 기술명을 입력합니다.",
                refinement_action="type",
                refinement_marker_id=54,
                refinement_label="기술스택 검색창",
                refinement_text="iOS",
                cards=[],
            )

    monkeypatch.setattr(selector, "_get_result_card_selector_model", lambda: FakeModel())

    message, trace = selector.select_result_cards(state)

    assert trace["action"] == "type_in_marker"
    assert message.tool_calls[0]["args"]["marker_id"] == 54
    assert message.tool_calls[0]["args"]["target_label"] == "기술스택 검색창"


def test_compact_markers_includes_only_wide_empty_icon_as_input_candidate():
    from agent.runtime import result_card_selector as selector

    compact = selector._compact_markers(
        [
            {"id": 1, "type": "text", "text": "iOS", "bbox": [0, 0, 20, 10]},
            {"id": 2, "type": "icon", "text": "", "bbox": [0, 0, 120, 20]},
            {"id": 3, "type": "icon", "text": "", "bbox": [0, 0, 20, 20]},
        ]
    )

    assert [marker["id"] for marker in compact] == [1, 2]
    assert compact[1]["input_candidate"] is True


def test_result_card_selector_includes_search_query_in_model_context(tmp_path):
    import json
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)

    messages = selector._selection_messages(_state(image_path), remaining_count=2)
    payload = json.loads(messages[1].content[0]["text"])

    assert payload["search_query"] == "iOS 개발자"
    assert payload["remaining_count"] == 2


def test_result_card_selector_marks_visible_all_in_model_context(tmp_path):
    import json
    from PIL import Image
    from agent.runtime import result_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    state = _state(image_path)
    state["recipe_params"].update({"target_count": 0, "count_mode": "visible_all"})

    messages = selector._selection_messages(state, remaining_count=3)
    payload = json.loads(messages[1].content[0]["text"])

    assert payload["count_mode"] == "visible_all"
    assert "첫 안정 검색 결과 화면" in messages[0].content


def test_reasoning_prompt_uses_visible_filter_after_selector_requests_refinement():
    from agent.graph import nodes

    messages = nodes._build_reasoning_messages(
        {
            "goal": "iOS 공고 두 개 수집",
            "plan": [],
            "current_plan_step": 0,
            "extracted_jd": {},
            "ui_context": "[id: 1] 기술스택\n[id: 2] Software Engineer (iOS)",
            "current_url": "https://www.wanted.co.kr/search",
            "marked_image": "",
            "recipe_params": {"query": "iOS", "target_count": 2},
            "action_history": [],
        },
        "",
        {
            "attempted": True,
            "reason": "result_refinement_needed",
            "refinement_reason": "정확한 후보가 한 개뿐입니다.",
        },
    )

    human_text = messages[-1].content
    assert "검색 결과 정제 필요" in human_text
    assert "화면 필터가 있으면 적용" in human_text
    assert "비슷한 직무로 개수를 채우지 마십시오" in human_text


def test_reasoning_node_uses_card_selector_without_general_model(monkeypatch):
    from agent.graph import nodes
    from agent.graph.action_request import build_action_message

    message = build_action_message(
        "card_selector",
        "selected",
        [{"name": "click_marker", "args": {"marker_id": 10}, "id": "click"}],
    )
    monkeypatch.setattr(
        nodes,
        "_select_result_cards",
        lambda state: (message, {"attempted": True, "reason": "cards_selected"}),
    )
    monkeypatch.setattr(
        nodes,
        "_get_ui_llm_with_tools",
        lambda: (_ for _ in ()).throw(AssertionError("카드 선택 후 범용 모델을 호출함")),
    )

    result = nodes.reasoning_node({"action_history": []})

    assert result["last_action_result"] is message
    assert result["step_durations"][0]["reasoning_mode"] == "card_selection"
    assert result["reflex_trace"]["source"] == "card_selector"

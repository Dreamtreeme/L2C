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

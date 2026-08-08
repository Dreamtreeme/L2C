"""채용공고 카드 선택과 재생 계약 테스트."""

from pathlib import Path
from types import SimpleNamespace

from agent.graph import worker_reasoning
from agent.tests.worker_test_support import node_runtime, worker_state


def _state(image_path: Path) -> dict:
    return worker_state(
        observation={
            "current_page_role": "search",
            "current_url": "https://example.com/search?q=ios",
            "marked_image": str(image_path),
            "current_markers": [
            {
                "id": 10,
                "type": "text",
                "text": "iOS 개발자",
                "bbox": [10, 10, 80, 30],
            },
            {
                "id": 20,
                "type": "text",
                "text": "백엔드 개발자",
                "bbox": [10, 50, 90, 70],
            },
            ],
        },
        request={
            "recipe_params": {"query": "iOS 개발자", "target_count": 2}
        },
    )


def test_selector_builds_queue_only_from_visible_markers(tmp_path, monkeypatch):
    from PIL import Image

    from agent.runtime import job_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.JobCardSelection(
                is_job_results_page=True,
                cards=[
                    selector.VisibleJobCard(
                        marker_id=10,
                        title="iOS 개발자",
                        company="회사 A",
                    ),
                    selector.VisibleJobCard(
                        marker_id=999,
                        title="화면에 없는 공고",
                        company="회사 B",
                    ),
                ],
            )

    monkeypatch.setattr(
        selector,
        "_get_job_card_selector_model",
        lambda: FakeModel(),
    )
    request, trace = selector.select_job_cards(_state(image_path))

    assert trace["reason"] == "cards_selected"
    assert trace["marker_ids"] == [10]
    assert request.tool_calls[0].name == "set_job_card_queue"
    assert len(request.tool_calls[0].args["cards"]) == 1


def test_selector_rejects_model_label_that_disagrees_with_ocr(tmp_path, monkeypatch):
    from PIL import Image

    from agent.runtime import job_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    state = _state(image_path)
    state["observation"]["current_markers"].append(
        {
            "id": 54,
            "type": "text",
            "text": "shoplive",
            "bbox": [100, 50, 160, 70],
        }
    )

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.JobCardSelection(
                is_job_results_page=True,
                needs_refinement=True,
                refinement_reason="iOS 기술 옵션을 찾습니다.",
                refinement_action="type",
                refinement_marker_id=54,
                refinement_label="기술스택 검색창",
                refinement_text="iOS",
                cards=[],
            )

    monkeypatch.setattr(
        selector,
        "_get_job_card_selector_model",
        lambda: FakeModel(),
    )
    request, trace = selector.select_job_cards(state)

    assert request is None
    assert trace["reason"] == "job_results_refinement_needed"


def test_loading_result_recaptures_without_general_reasoning(tmp_path, monkeypatch):
    from PIL import Image

    from agent.runtime import job_card_selector as selector

    image_path = tmp_path / "marked.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)

    class FakeModel:
        def invoke(self, inputs, config=None):
            return selector.JobCardSelection(is_loading=True)

    monkeypatch.setattr(
        selector,
        "_get_job_card_selector_model",
        lambda: FakeModel(),
    )
    monkeypatch.setattr(
        worker_reasoning,
        "_get_ui_llm_with_tools",
        lambda _runtime: (_ for _ in ()).throw(
            AssertionError("로딩 화면을 범용 모델에 보내면 안 됩니다.")
        ),
    )

    result = worker_reasoning.reasoning_node(
        _state(image_path),
        node_runtime(),
    )

    assert result["decision"]["pending_action"] is None
    assert (
        result["decision"]["job_card_selection_trace"]["reason"]
        == "screen_loading"
    )


def test_general_reasoning_converts_model_tool_call(monkeypatch):
    from agent.observability import run_context

    monkeypatch.setattr(
        worker_reasoning,
        "select_job_cards",
        lambda _state: (None, {"attempted": False}),
    )
    monkeypatch.setattr(
        worker_reasoning,
        "build_reasoning_messages",
        lambda *_args: ["현재 화면"],
    )
    monkeypatch.setattr(
        worker_reasoning,
        "_get_ui_llm_with_tools",
        lambda _runtime: object(),
    )
    monkeypatch.setattr(
        run_context,
        "invoke_with_metrics",
        lambda *_args: SimpleNamespace(
            content="아래 내용을 확인합니다.",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "scroll",
                    "args": {"direction": "down"},
                }
            ],
        ),
    )

    result = worker_reasoning.reasoning_node(
        worker_state(),
        node_runtime(),
    )

    assert result["decision"]["pending_action"].tool_calls[0].name == "scroll"
    assert result["replay"]["reflex_trace"]["source"] == "reasoning"


def test_resolved_card_count_only_includes_collected_or_database_confirmed():
    from agent.runtime.job_card_queue import (
        job_card_queue_scope_complete,
        resolved_job_card_count,
    )

    queue = [
        {"status": "done"},
        {"status": "skipped", "job_id": 7},
        {"status": "skipped", "skip_reason": "invalid_marker"},
    ]

    assert resolved_job_card_count(queue) == 2
    assert job_card_queue_scope_complete(
        queue,
        count_mode="explicit",
        target_count=2,
    )

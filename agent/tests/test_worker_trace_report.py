from __future__ import annotations

import json

from agent.observability.worker_trace_report import (
    build_worker_trace,
    render_worker_trace,
)
from agent.recipe.submission_store import SubmissionStore
from scripts.inspect_worker_trace import main


def _submission_payload(review_attempt: int = 0) -> dict:
    capture_1 = "worker-run-1:attempt:00:capture:0001"
    capture_2 = "worker-run-1:attempt:00:capture:0002"
    return {
        "run_id": "worker-run-1",
        "review_attempt": review_attempt,
        "site": "wanted",
        "goal": "iOS 개발자 공고 수집",
        "run_status": "finished",
        "recorded_steps": [
            {
                "seq": 1,
                "action": "click_marker",
                "decision_capture_id": capture_1,
                "intent": "검색 버튼을 누른다",
                "target": {
                    "semantic_label": "검색 아이콘",
                    "text": "Q",
                },
            },
            {
                "seq": 3,
                "action": "press_key",
                "decision_capture_id": capture_2,
                "value": "ENTER",
            },
        ],
        "feedback_episodes": [
            {
                "seq": 1,
                "proposal": {
                    "action": "click_marker",
                    "reason": "검색 UI 열기",
                    "target_label": "검색 아이콘",
                },
                "observation": {
                    "before": {
                        "capture_id": capture_1,
                        "screenshot": "before.png",
                    },
                    "result": {"action_source": "llm"},
                },
                "feedback": {"label": "partial", "reason": "화면 변화 대기"},
            },
            {
                "seq": 2,
                "proposal": {
                    "action": "type_in_marker",
                    "args": {"text": "iOS 개발자"},
                    "target_label": "검색 입력창",
                },
                "observation": {
                    "before": {
                        "capture_id": capture_2,
                        "screenshot": "input.png",
                    },
                    "result": {"action_source": "llm"},
                },
                "feedback": {"label": "no_effect", "reason": "입력 동작"},
            },
        ],
        "transition_observations": [
            {
                "action_seq": 1,
                "action": "click_marker",
                "attempt": 1,
                "from_capture_id": capture_1,
                "to_capture_id": "",
                "status": "pending",
                "reason": "화면 변화 대기",
            },
            {
                "action_seq": 1,
                "action": "click_marker",
                "attempt": 2,
                "from_capture_id": capture_1,
                "to_capture_id": capture_2,
                "status": "ready",
                "reason": "pHash 화면 변화 확인",
                "screenshot": "after.png",
            },
        ],
    }


def test_submission_store_gets_exact_and_latest_attempt(tmp_path):
    store = SubmissionStore(tmp_path / "jobs.db")
    first_id = store.commit_submission(
        _submission_payload(0),
        {"decision": "revise", "confidence": 0.4},
    )
    latest_id = store.commit_submission(
        _submission_payload(1),
        {"decision": "accept", "confidence": 0.9},
    )

    assert store.get_submission(first_id)["review_attempt"] == 0
    assert store.get_run_attempt("worker-run-1", 0)["submission_id"] == first_id
    assert store.get_run_attempt("worker-run-1")["submission_id"] == latest_id


def test_build_worker_trace_merges_actions_and_uses_latest_transition():
    trace = build_worker_trace(
        {
            "submission_id": "worker-run-1:0",
            "review_decision": "accept",
            "payload": _submission_payload(),
        }
    )

    assert [step["seq"] for step in trace["steps"]] == [1, 2, 3]
    assert trace["step_count"] == 3
    assert trace["recipe_step_count"] == 2
    assert trace["transition_count"] == 2
    assert trace["capture_count"] == 2

    first = trace["steps"][0]
    assert first["target_label"] == "검색 아이콘"
    assert first["transition_attempt"] == 2
    assert first["transition_attempt_count"] == 2
    assert first["transition_status"] == "ready"
    assert first["to_capture_id"].endswith("capture:0002")
    assert first["capture_consistent"] is True

    feedback_only = trace["steps"][1]
    assert feedback_only["action"] == "type_in_marker"
    assert feedback_only["recorded_for_recipe"] is False
    assert feedback_only["decision_capture_id"].endswith("capture:0002")

    rendered = render_worker_trace(trace)
    assert "capture:0001 -- click_marker [검색 아이콘] --> capture:0002" in rendered
    assert "전환=ready" in rendered


def test_worker_trace_cli_outputs_structured_json(tmp_path, capsys):
    db_path = tmp_path / "jobs.db"
    store = SubmissionStore(db_path)
    store.commit_submission(
        _submission_payload(),
        {"decision": "accept", "confidence": 0.9},
    )

    assert main(["--db", str(db_path), "--run-id", "worker-run-1", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["run_id"] == "worker-run-1"
    assert output["review_decision"] == "accept"
    assert output["steps"][0]["from_capture_id"].endswith("capture:0001")

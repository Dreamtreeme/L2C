"""저장된 성공 화면에서 지휘자 모델의 첫 물리 행동을 비교한다."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.application.model_clients import get_google_chat_model
from agent.application.run_context import invoke_with_metrics, run_context
from agent.graph.tool_schema import ACTION_TOOL_SCHEMAS
from agent.graph.worker_reasoning import _build_reasoning_messages
from agent.runtime.site_context import infer_site_page_role
from benchmark.bench_openai_detail import _load_dotenv


DEFAULT_SUBMISSION_ID = "worker-20260714122243-27719ba9:0"
DEFAULT_MODELS = ("gemini-3.5-flash", "gemini-3.6-flash")
DEFAULT_SEQUENCES = (5, 9, 11, 12, 14, 16, 17, 21)
TOKEN_PRICES = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.6-flash": (1.50, 7.50),
}


def _load_episodes(
    submission_id: str,
    db_path: Path,
    sequences: set[int],
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT payload_json FROM worker_submissions WHERE submission_id=?",
            (submission_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise SystemExit(f"worker submission not found: {submission_id}")
    payload = json.loads(row["payload_json"])
    return [
        dict(episode)
        for episode in payload.get("feedback_episodes") or []
        if int(episode.get("seq") or 0) in sequences
    ]


def _episode_state(episode: dict[str, Any]) -> dict[str, Any]:
    before = ((episode.get("observation") or {}).get("before") or {})
    marker_texts = list(before.get("marker_texts") or [])
    current_url = str(before.get("url") or "")
    ui_context = "\n".join(
        f"[{index}] {text}"
        for index, text in enumerate(marker_texts)
        if str(text).strip()
    )
    return {
        "goal": str(episode.get("goal") or ""),
        "current_url": current_url,
        "current_page_role": infer_site_page_role(current_url, marker_texts),
        "marked_image": str(before.get("marked_image") or ""),
        "ui_context": ui_context,
        "current_markers": [
            {"id": index, "text": str(text), "type": "text"}
            for index, text in enumerate(marker_texts)
        ],
        "recipe_params": {"target_count": 2, "query": "iOS"},
        "action_history": [],
        "plan": [],
        "extracted_jd": {},
    }


def _tool_calls(response: Any) -> list[dict[str, Any]]:
    calls = list(getattr(response, "tool_calls", None) or [])
    return [
        {
            "name": str(call.get("name") or ""),
            "args": dict(call.get("args") or {}),
        }
        for call in calls
    ]


def _matches_reference(calls: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, bool]:
    matching_calls = [
        call for call in calls if call.get("name") == reference.get("action")
    ]
    action_match = bool(matching_calls)
    expected_args = dict(reference.get("args") or {})
    expected_marker = expected_args.get("marker_id")
    marker_match = (
        expected_marker is None
        or any(
            dict(call.get("args") or {}).get("marker_id") == expected_marker
            for call in matching_calls
        )
    )
    return {
        "action": action_match,
        "marker_when_required": marker_match,
        "exact_target": action_match and marker_match,
    }


def _estimated_cost(model: str, usage: dict[str, Any]) -> float:
    input_price, output_price = TOKEN_PRICES[model]
    return round(
        (
            int(usage.get("input_tokens") or 0) * input_price
            + int(usage.get("output_tokens") or 0) * output_price
        )
        / 1_000_000,
        8,
    )


def _run_case(
    model: str,
    episode: dict[str, Any],
    *,
    thinking_level: str | None = None,
) -> dict[str, Any]:
    state = _episode_state(episode)
    tools = list(ACTION_TOOL_SCHEMAS.values())
    llm = get_google_chat_model(
        model,
        temperature=0.1,
        thinking_level=thinking_level,
    ).bind_tools(tools)
    with run_context(query="screen action comparison", prefix="bench-action") as (
        context,
        _created,
    ):
        started = time.perf_counter()
        response = invoke_with_metrics(
            llm,
            _build_reasoning_messages(state, ""),
            "screen_action_comparison",
        )
        duration = time.perf_counter() - started
        metrics = context.snapshot()
    usage = dict(((metrics.get("llm") or {}).get("totals") or {}))
    reference = dict(episode.get("proposal") or {})
    calls = _tool_calls(response)
    return {
        "seq": int(episode.get("seq") or 0),
        "url": state["current_url"],
        "reference": {
            "action": reference.get("action"),
            "marker_id": (reference.get("args") or {}).get("marker_id"),
            "target_label": (reference.get("args") or {}).get("target_label"),
        },
        "prediction": calls,
        "match": _matches_reference(calls, reference),
        "duration_sec": round(duration, 3),
        "usage": usage,
        "estimated_cost_usd": _estimated_cost(model, usage),
        "thinking_level": thinking_level,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-id", default=DEFAULT_SUBMISSION_ID)
    parser.add_argument("--db", type=Path, default=Path("data/jobs.db"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--thinking-levels",
        nargs="+",
        choices=("minimal", "low", "medium", "high"),
    )
    parser.add_argument("--sequences", nargs="+", type=int, default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    episodes = _load_episodes(args.submission_id, args.db, set(args.sequences))
    report: dict[str, Any] = {
        "submission_id": args.submission_id,
        "case_count": len(episodes),
        "models": {},
        "caveat": "저장된 성공 행동과의 일치율이며 다른 유효 행동 가능성은 수동 검토한다.",
    }
    variants = [
        (model, thinking_level)
        for model in args.models
        for thinking_level in (args.thinking_levels or [None])
    ]
    for model, thinking_level in variants:
        variant_name = f"{model}:{thinking_level}" if thinking_level else model
        cases = [
            _run_case(model, episode, thinking_level=thinking_level)
            for episode in episodes
        ]
        report["models"][variant_name] = {
            "thinking_level": thinking_level,
            "cases": cases,
            "action_match_count": sum(1 for case in cases if case["match"]["action"]),
            "exact_target_count": sum(
                1 for case in cases if case["match"]["exact_target"]
            ),
            "duration_sec": round(sum(case["duration_sec"] for case in cases), 3),
            "input_tokens": sum(
                int(case["usage"].get("input_tokens") or 0) for case in cases
            ),
            "output_tokens": sum(
                int(case["usage"].get("output_tokens") or 0) for case in cases
            ),
            "estimated_cost_usd": round(
                sum(case["estimated_cost_usd"] for case in cases), 8
            ),
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

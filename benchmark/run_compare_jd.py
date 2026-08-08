"""Classic DOM 추출과 Vision 추출을 같은 공고에서 비교한다."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(ROOT_DIR / ".env")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.runtime.worker_contracts import create_worker_state
from agent.graph.workflow import build_graph
from agent.utils.logger import logger
from benchmark.quality_eval import evaluate_job_records
from classic.automation.capture import capture_and_extract_dom
from classic.extractor.llm_engine import LLMEngine


def _run_classic(target_url: str) -> tuple[dict[str, Any], str, float]:
    started = time.perf_counter()
    try:
        dom_raw = capture_and_extract_dom(target_url)
        full_text = dom_raw.get("full_text", "")
        if not full_text:
            raise ValueError("Playwright DOM에서 텍스트를 추출하지 못했습니다.")
        return LLMEngine().extract_from_text(full_text), "", time.perf_counter() - started
    except Exception as exc:
        return {}, str(exc), time.perf_counter() - started


def _parse_collected_item(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, re.DOTALL)
        if not match:
            return {"raw_text": value}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {"raw_text": value}


def _run_vision(target_url: str) -> tuple[dict[str, Any], str, float]:
    started = time.perf_counter()
    goal = (
        f"현재 바탕화면입니다. 채용공고 페이지 '{target_url}'에 물리 입력으로 직접 접속하세요. "
        "회사명, 직무명, 주요업무, 자격요건, 우대사항, 혜택을 화면과 OCR로 끝까지 수집해 "
        "구조화된 JSON으로 반환하세요."
    )
    try:
        app = build_graph()
        final_state: dict[str, Any] = {}
        for output in app.stream(create_worker_state(goal), {"recursion_limit": 100}):
            for value in output.values():
                if isinstance(value, dict):
                    final_state.update(value)
        collected = final_state.get("collected_data") or []
        if not collected:
            raise ValueError("Vision Agent가 수집 결과를 반환하지 않았습니다.")
        return _parse_collected_item(collected[0]), "", time.perf_counter() - started
    except Exception as exc:
        return {}, str(exc), time.perf_counter() - started


def _write_report(payload: dict[str, Any], path: Path) -> None:
    quality = payload.get("quality") or {}
    lines = [
        "# Classic vs Vision 추출 비교",
        "",
        f"- 대상 URL: {payload['target_url']}",
        f"- 실행 시각: {payload['started_at']}",
        f"- 전체 상태: {payload['status']}",
        f"- Classic 실행시간: {payload['classic']['duration_sec']:.3f}초",
        f"- Vision 실행시간: {payload['vision']['duration_sec']:.3f}초",
        "",
        "## 실행 오류",
        "",
        f"- Classic: {payload['classic']['error'] or '없음'}",
        f"- Vision: {payload['vision']['error'] or '없음'}",
        "",
        "## 결정론적 품질 지표",
        "",
        "```json",
        json.dumps(quality, ensure_ascii=False, indent=2),
        "```",
        "",
        "Classic 결과는 정답 데이터가 아니라 비교 기준입니다. 실패한 실행에는 대체 데이터를 넣지 않습니다.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Classic DOM and Vision extraction.")
    parser.add_argument("--url", default="https://www.wanted.co.kr/wd/350432")
    parser.add_argument("--output", type=Path, default=Path("benchmark/jd_comparison_run.json"))
    parser.add_argument("--report", type=Path, default=Path("benchmark/jd_comparison_report.md"))
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("Classic comparison run started", target_url=args.url)
    classic_data, classic_error, classic_duration = _run_classic(args.url)
    logger.info("Vision comparison run started", target_url=args.url)
    vision_data, vision_error, vision_duration = _run_vision(args.url)
    status = "completed" if not classic_error and not vision_error else "failed"
    quality = (
        evaluate_job_records(vision_data, classic_data)
        if classic_data and vision_data
        else {}
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at,
        "target_url": args.url,
        "classic": {
            "duration_sec": round(classic_duration, 6),
            "error": classic_error,
            "result": classic_data,
        },
        "vision": {
            "duration_sec": round(vision_duration, 6),
            "error": vision_error,
            "result": vision_data,
        },
        "quality": quality,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(payload, args.report)
    logger.info(
        "Comparison run completed",
        status=status,
        output=str(args.output),
        report=str(args.report),
    )
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

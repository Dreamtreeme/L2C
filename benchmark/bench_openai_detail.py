"""OpenAI nano 모델로 상세 OCR 텍스트 정제만 단독 벤치합니다."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prompts.detail_extraction import build_detail_extraction_system_prompt

from shared.schema.jd_schema import JobPosting


DEFAULT_SUBMISSION_ID = "worker-20260701010743-25ecd359:0"
DEFAULT_MODEL = "gpt-5.4-nano"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _dedupe_marker_texts(submission_id: str, db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT payload_json FROM worker_submissions WHERE submission_id=?",
        (submission_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"worker submission not found: {submission_id}")

    payload = json.loads(row["payload_json"])
    lines_by_url: dict[str, list[str]] = defaultdict(list)
    seen_by_url: dict[str, set[str]] = defaultdict(set)
    card_by_url: dict[str, dict[str, str]] = {}

    for episode in payload.get("feedback_episodes") or []:
        before = ((episode.get("observation") or {}).get("before") or {})
        url = str(before.get("url") or "")
        if "/wd/" not in url:
            continue

        proposal = episode.get("proposal") or {}
        active_card = proposal.get("active_result_card") or {}
        if active_card and url not in card_by_url:
            card_by_url[url] = {
                "title": active_card.get("title") or active_card.get("target_label") or "",
                "company": active_card.get("company") or "",
            }

        for raw_text in before.get("marker_texts") or []:
            text = str(raw_text or "").strip()
            if not text:
                continue
            if "상호작용 가능한 요소" in text:
                continue
            if text in seen_by_url[url]:
                continue
            seen_by_url[url].add(text)
            lines_by_url[url].append(text)

    return [
        {
            "url": url,
            "active_result_card": card_by_url.get(url, {}),
            "ocr_text": "\n".join(lines),
            "line_count": len(lines),
        }
        for url, lines in lines_by_url.items()
    ]


def _extract_output_text(response_json: dict[str, Any]) -> str:
    if isinstance(response_json.get("output_text"), str):
        return response_json["output_text"]
    parts: list[str] = []
    for item in response_json.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "\n".join(parts)


def _parse_output(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
        job = JobPosting.model_validate(parsed)
        return job.model_dump(exclude_none=True), ""
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, str(exc)


def _request_openai(
    *,
    api_key: str,
    model: str,
    document: dict[str, Any],
    timeout: int,
    max_output_tokens: int,
) -> tuple[str, dict[str, Any]]:
    schema = JobPosting.model_json_schema()
    properties = dict(schema.get("properties") or {})
    for noisy_field in ("raw_ocr_text", "content_hash"):
        properties.pop(noisy_field, None)
    schema["properties"] = properties
    if isinstance(schema.get("required"), list):
        schema["required"] = [
            key for key in schema["required"] if key in properties
        ]
    schema["additionalProperties"] = False
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": build_detail_extraction_system_prompt(
                    "누적 OCR 본문에서 채용공고 1건을 JobPosting JSON으로 정리하십시오. "
                    "OCR에 없는 사실은 만들지 말고, 알 수 없는 필드는 null 또는 빈 배열로 두십시오. "
                    "현재 상세 URL은 보존하십시오."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_url": document["url"],
                        "ocr_text": document["ocr_text"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        "max_output_tokens": max_output_tokens,
        "temperature": 0,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "job_posting",
                "schema": schema,
                "strict": False,
            }
        },
        "store": False,
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    try:
        response_json = response.json()
    except ValueError:
        response_json = {"raw_text": response.text}
    if response.status_code >= 400:
        raise RuntimeError(json.dumps(response_json, ensure_ascii=False)[:2000])
    return _extract_output_text(response_json), response_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--submission-id", default=DEFAULT_SUBMISSION_ID)
    parser.add_argument("--db", default="data/jobs.db")
    parser.add_argument("--max-docs", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    documents = _dedupe_marker_texts(args.submission_id, Path(args.db))[: args.max_docs]
    if not documents:
        raise SystemExit("no detail documents reconstructed")

    print(
        json.dumps(
            {
                "model": args.model,
                "submission_id": args.submission_id,
                "doc_count": len(documents),
            },
            ensure_ascii=False,
        )
    )

    results = []
    for index, document in enumerate(documents):
        started = time.perf_counter()
        output_text, response_json = _request_openai(
            api_key=api_key,
            model=args.model,
            document=document,
            timeout=args.timeout,
            max_output_tokens=args.max_output_tokens,
        )
        duration = time.perf_counter() - started
        parsed, error = _parse_output(output_text)
        usage = response_json.get("usage") or {}
        result = {
            "index": index,
            "url": document["url"],
            "line_count": document["line_count"],
            "char_count": len(document["ocr_text"]),
            "duration_sec": round(duration, 3),
            "parse_ok": parsed is not None,
            "usage": usage,
            "error": error[:500],
            "raw_output": output_text[:3000],
            "parsed": parsed,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print(
        json.dumps(
            {
                "total_infer_sec": round(sum(item["duration_sec"] for item in results), 3),
                "parse_ok_count": sum(1 for item in results if item["parse_ok"]),
                "input_tokens": sum((item.get("usage") or {}).get("input_tokens", 0) for item in results),
                "output_tokens": sum((item.get("usage") or {}).get("output_tokens", 0) for item in results),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
